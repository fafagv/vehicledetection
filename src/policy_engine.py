"""
src/rl/policy_engine.py

Roadmap step 3 (glue): feeds vision-tracked vehicles into the 6-Stage
Dueling DQN, computes each vehicle's multi-agent context summary,
enforces `rules.py`'s hard constraints as a Q-value mask, and picks the
final per-vehicle action -- the piece that actually turns one frame's
`StateVector`s into `Action`s.

NOT RUNTIME-VERIFIED here (depends on `dueling_dqn.py` -> torch; see that
module's docstring). `RuleEnforcer`, which this module also depends on,
*is* unit-tested (see tests/test_rules.py) since it's pure numpy/stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from src.rl.dueling_dqn import SixStageDuelingDQN
from src.rl.rules import RuleEnforcer, StopLine
from src.rl.types import Action, RuleViolation, StateVector


@dataclass(frozen=True)
class PolicyDecision:
    track_id: int
    action: Action
    q_values: Dict[Action, float]
    violations: List[RuleViolation]


class MultiAgentPolicyEngine:
    """Centralized-inference, decentralized-execution policy: one shared
    `SixStageDuelingDQN` is queried once per agent per frame, each call
    conditioned on a mean-pooled summary of every *other* agent present
    that frame (stage 5's multi-agent context), then `RuleEnforcer`
    masks out illegal actions before the final argmax.

    Args:
        network: the shared Dueling DQN. Callers own its device
            placement/eval-vs-train mode; this class always wraps
            inference in `torch.no_grad()`.
        rule_enforcer: hard-constraint mask source.
        device: torch device for feature tensors.
    """

    def __init__(
        self,
        network: SixStageDuelingDQN,
        rule_enforcer: Optional[RuleEnforcer] = None,
        device: str = "cpu",
    ) -> None:
        self.network = network
        self.rule_enforcer = rule_enforcer or RuleEnforcer()
        self.device = torch.device(device)

    def _context_summaries(self, states: Sequence[StateVector]) -> torch.Tensor:
        """For each agent i, mean-pool stage-4 fusion embeddings of every
        *other* agent j != i into a [N, hidden_dim] tensor -- agent i's
        `context_summary` input to `SixStageDuelingDQN.forward`.

        O(N) fusion-embedding calls (one batched call, actually) plus an
        O(N^2) pooling step; fine at intersection-scale N (tens of
        agents), matching the same complexity tradeoff noted in
        `rules.py::RuleEnforcer.evaluate`.
        """
        n = len(states)
        features = torch.tensor([s.to_feature_tuple() for s in states], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            fusion = self.network.fusion_embedding(features)  # [N, hidden_dim]

        if n <= 1:
            return torch.zeros(n, self.network.hidden_dim, device=self.device)

        total = fusion.sum(dim=0, keepdim=True)  # [1, hidden_dim]
        # Mean of all *other* agents = (sum - self) / (n - 1).
        others_sum = total - fusion
        return others_sum / (n - 1)

    def decide(
        self,
        states: Sequence[StateVector],
        stop_lines: Sequence[StopLine] = (),
        corridor_heading_rad: Optional[float] = None,
    ) -> List[PolicyDecision]:
        """One decision per vehicle in `states`, jointly aware of every
        other vehicle present (via context) and every active rule
        violation (via masking)."""
        if not states:
            return []

        violations = self.rule_enforcer.evaluate(states, stop_lines, corridor_heading_rad)

        features = torch.tensor([s.to_feature_tuple() for s in states], dtype=torch.float32, device=self.device)
        context = self._context_summaries(states)

        with torch.no_grad():
            q_values = self.network(features, context_summary=context)  # [N, NUM_ACTIONS]

        decisions: List[PolicyDecision] = []
        for i, state in enumerate(states):
            mask = self.rule_enforcer.action_mask(state, violations)
            masked_q = q_values[i].clone()
            masked_q[~torch.from_numpy(mask)] = float("-inf")
            best_action = Action(int(torch.argmax(masked_q).item()))

            own_violations = [v for v in violations if v.track_id == state.track_id]
            decisions.append(
                PolicyDecision(
                    track_id=state.track_id,
                    action=best_action,
                    q_values={Action(a): float(q_values[i, a]) for a in range(q_values.size(-1))},
                    violations=own_violations,
                )
            )
        return decisions
