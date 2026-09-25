"""
src/rl/trainer.py

Roadmap step 4: Real-Time Optimization & Auto-Tuning.

Wires `AdamW` + `ReduceLROnPlateau` around `SixStageDuelingDQN` for
active simulation-loop training, using a Double DQN target (reduces the
well-known DQN overestimation bias, cheap to add given we already
maintain a target network for stability) and Huber loss (robust to the
reward-scale/outlier spikes that "gradient volatility" in the roadmap is
presumably referring to).

"Dynamically balance loss variance and gradient volatility" is
implemented as: `ReduceLROnPlateau` watches a rolling *smoothed* loss
(EMA, not the raw per-batch loss) so a single high-variance batch can't
trigger a premature LR cut, and drops the learning rate only once that
smoothed signal genuinely plateaus -- i.e. the auto-tuning reacts to
sustained volatility, not single-batch noise.

NOT RUNTIME-VERIFIED here (torch dependency; see dueling_dqn.py's
docstring for why). `ReplayBuffer.sample`'s indexing/shape logic is
plain-Python/numpy though, and is covered by tests/test_rules.py's
sibling test module, tests/test_replay_buffer.py.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from src.rl.dueling_dqn import SixStageDuelingDQN
from src.rl.types import Action


class Transition(NamedTuple):
    state_features: Tuple[float, ...]
    context_summary: Optional[Tuple[float, ...]]
    action: Action
    reward: float
    next_state_features: Tuple[float, ...]
    next_context_summary: Optional[Tuple[float, ...]]
    done: bool


class ReplayBuffer:
    """Fixed-capacity uniform replay buffer. Deliberately simple (no
    prioritization) -- this is the auto-tuning roadmap item's concern
    (optimizer/scheduler dynamics), not the sampling-strategy one; swap
    in prioritized replay later without touching `DQNTrainer` by giving
    it any object satisfying `push`/`sample`/`__len__`.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size > len(self._buffer):
            raise ValueError(f"Requested batch_size={batch_size} but buffer only has {len(self._buffer)} transitions.")
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


@dataclass
class TrainerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4  # AdamW's decoupled weight decay
    gamma: float = 0.99
    target_update_every_steps: int = 500
    huber_delta: float = 1.0
    grad_clip_norm: float = 10.0
    loss_ema_alpha: float = 0.05  # smoothing factor feeding the scheduler
    scheduler_factor: float = 0.5
    scheduler_patience: int = 20  # in "scheduler.step()" calls, i.e. optimizer steps
    scheduler_min_lr: float = 1e-6


class DQNTrainer:
    """Double DQN training loop with AdamW + ReduceLROnPlateau.

    Args:
        policy_net: the network being trained.
        target_net: a separate `SixStageDuelingDQN` instance (same
            architecture) used for bootstrapped targets; caller
            initializes it as a copy of `policy_net`'s weights.
        config: hyperparameters; see `TrainerConfig`.
    """

    def __init__(
        self,
        policy_net: SixStageDuelingDQN,
        target_net: SixStageDuelingDQN,
        config: Optional[TrainerConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = torch.device(device)
        self.policy_net = policy_net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        # `mode="min"` on the smoothed loss: drop LR when training has
        # genuinely stalled, not on a single noisy uptick -- see module
        # docstring on why we feed it an EMA rather than the raw loss.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.scheduler_min_lr,
        )

        self._loss_ema: Optional[float] = None
        self._step_count = 0

    def _to_tensor(self, rows: List[Tuple[float, ...]]) -> torch.Tensor:
        return torch.tensor(np.asarray(rows, dtype=np.float32), device=self.device)

    def _context_tensor(self, rows: List[Optional[Tuple[float, ...]]]) -> Optional[torch.Tensor]:
        if all(r is None for r in rows):
            return None
        hidden_dim = self.policy_net.hidden_dim
        filled = [r if r is not None else (0.0,) * hidden_dim for r in rows]
        return self._to_tensor(filled)

    def train_step(self, batch: List[Transition]) -> float:
        """One gradient step over a sampled batch. Returns the raw
        (un-smoothed) loss value for logging; the smoothed EMA used by
        the scheduler is tracked internally and applied in `end_of_epoch`.
        """
        states = self._to_tensor([t.state_features for t in batch])
        contexts = self._context_tensor([t.context_summary for t in batch])
        actions = torch.tensor([int(t.action) for t in batch], device=self.device, dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], device=self.device, dtype=torch.float32)
        next_states = self._to_tensor([t.next_state_features for t in batch])
        next_contexts = self._context_tensor([t.next_context_summary for t in batch])
        dones = torch.tensor([t.done for t in batch], device=self.device, dtype=torch.float32)

        q_values = self.policy_net(states, context_summary=contexts)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: action selection from the *online* network,
            # value estimation from the *target* network -- decouples the
            # two to curb overestimation bias vs. vanilla DQN.
            next_online_q = self.policy_net(next_states, context_summary=next_contexts)
            next_actions = next_online_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_states, context_summary=next_contexts)
            next_q = next_target_q.gather(1, next_actions).squeeze(1)
            targets = rewards + self.config.gamma * next_q * (1.0 - dones)

        loss = F.huber_loss(q_taken, targets, delta=self.config.huber_delta)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.config.target_update_every_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_value = float(loss.item())
        alpha = self.config.loss_ema_alpha
        self._loss_ema = loss_value if self._loss_ema is None else (1 - alpha) * self._loss_ema + alpha * loss_value
        return loss_value

    def end_of_epoch(self) -> Optional[float]:
        """Call once per epoch (or once every N `train_step` calls) to
        advance `ReduceLROnPlateau` on the smoothed loss and return the
        current smoothed value for logging. Returns None if no
        `train_step` has run yet."""
        if self._loss_ema is None:
            return None
        self.scheduler.step(self._loss_ema)
        return self._loss_ema

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
