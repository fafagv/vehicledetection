"""
src/rl/env.py

A minimal, synthetic single-lane car-following environment used by
`train_policy.py` to exercise the full stack (state -> action -> reward
-> next state) end-to-end. This is a toy kinematic simulator for
wiring/integration purposes, NOT a traffic simulator -- it models one
ego vehicle following one lead vehicle with a randomly varying speed
profile, plus a fixed stop line, and uses `RuleEnforcer` for reward
shaping (rule violations are penalized, a hard SAT collision ends the
episode). Swap this out for a real simulator (SUMO, CARLA, an actual
replayed-video environment, ...) by implementing the same
`reset()`/`step()` contract; nothing else in `src/rl` needs to change --
`policy_engine.py` and `trainer.py` only ever see `StateVector`s and
`Action`s.

Pure stdlib -- no torch, no numpy even -- so (unlike most of the rest of
`src/rl`) this module IS runtime-tested in this sandbox
(`tests/test_env.py`).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.rl.rules import RuleEnforcer, StopLine, sat_overlap
from src.rl.types import Action, OrientedBoundingBox, StateVector

# Simple constant-acceleration model per action -- deliberately crude
# (a real vehicle dynamics model would have jerk limits, drivetrain lag,
# etc.) since the point of this env is exercising the RL plumbing, not
# physical fidelity.
ACTION_ACCEL_MPS2: Dict[Action, float] = {
    Action.HARD_BRAKE: -6.0,
    Action.DECELERATE: -2.5,
    Action.MAINTAIN_SPEED: 0.0,
    Action.ACCELERATE: 1.5,
    Action.LANE_CHANGE_LEFT: 0.0,  # no-op: this demo env is single-lane
    Action.LANE_CHANGE_RIGHT: 0.0,  # no-op: this demo env is single-lane
}


@dataclass
class EnvConfig:
    dt_s: float = 0.2
    max_steps: int = 300
    ego_length_m: float = 4.5
    ego_width_m: float = 1.8
    lead_length_m: float = 4.5
    lead_width_m: float = 1.8
    stop_line_distance_m: float = 120.0
    lead_min_speed_kmh: float = 10.0
    lead_max_speed_kmh: float = 60.0
    lead_speed_change_prob: float = 0.05
    initial_ego_speed_kmh: float = 30.0
    initial_gap_m: float = 40.0


class SyntheticFollowingEnv:
    """One ego vehicle, one lead vehicle, one stop line, on a straight
    single lane. See module docstring for scope/limitations."""

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        rule_enforcer: Optional[RuleEnforcer] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or EnvConfig()
        self.rule_enforcer = rule_enforcer or RuleEnforcer()
        self._rng = random.Random(seed)
        self.step_count = 0
        self.ego_position_m = 0.0
        self.ego_speed_kmh = self.config.initial_ego_speed_kmh
        self.lead_position_m = self.config.initial_gap_m
        self.lead_speed_kmh = self.config.initial_ego_speed_kmh
        self.stop_line = StopLine(
            point_a_m=(-2.0, self.config.stop_line_distance_m),
            point_b_m=(2.0, self.config.stop_line_distance_m),
            approach_heading_rad=math.pi / 2,
        )
        self.reset()

    def reset(self) -> StateVector:
        self.step_count = 0
        self.ego_position_m = 0.0
        self.ego_speed_kmh = self.config.initial_ego_speed_kmh
        self.lead_position_m = self.config.initial_gap_m
        self.lead_speed_kmh = self._rng.uniform(self.config.lead_min_speed_kmh, self.config.lead_max_speed_kmh)
        return self._ego_state()

    def _ego_state(self) -> StateVector:
        dist_to_stop = self.config.stop_line_distance_m - self.ego_position_m
        return StateVector(
            track_id=0,
            position_m=(0.0, self.ego_position_m),
            velocity_mps=(0.0, self.ego_speed_kmh / 3.6),
            heading_rad=math.pi / 2,
            speed_kmh=self.ego_speed_kmh,
            length_m=self.config.ego_length_m,
            width_m=self.config.ego_width_m,
            class_id=2,
            class_name="car",
            is_heavy_vehicle=False,
            dist_to_stop_line_m=dist_to_stop if dist_to_stop >= 0 else None,
        )

    def _lead_state(self) -> StateVector:
        return StateVector(
            track_id=1,
            position_m=(0.0, self.lead_position_m),
            velocity_mps=(0.0, self.lead_speed_kmh / 3.6),
            heading_rad=math.pi / 2,
            speed_kmh=self.lead_speed_kmh,
            length_m=self.config.lead_length_m,
            width_m=self.config.lead_width_m,
            class_id=2,
            class_name="car",
            is_heavy_vehicle=False,
        )

    def step(self, action: Action) -> Tuple[StateVector, float, bool, dict]:
        dt = self.config.dt_s
        accel = ACTION_ACCEL_MPS2[action]
        ego_speed_mps = max(0.0, self.ego_speed_kmh / 3.6 + accel * dt)
        self.ego_speed_kmh = ego_speed_mps * 3.6
        self.ego_position_m += ego_speed_mps * dt

        if self._rng.random() < self.config.lead_speed_change_prob:
            self.lead_speed_kmh = self._rng.uniform(self.config.lead_min_speed_kmh, self.config.lead_max_speed_kmh)
        self.lead_position_m += (self.lead_speed_kmh / 3.6) * dt

        ego_state = self._ego_state()
        lead_state = self._lead_state()

        violations = self.rule_enforcer.evaluate([ego_state, lead_state], stop_lines=[self.stop_line])
        ego_violations = [v for v in violations if v.track_id == 0]

        collided, _ = sat_overlap(OrientedBoundingBox.from_state(ego_state), OrientedBoundingBox.from_state(lead_state))

        # Reward shaping: small per-step progress reward, minus rule
        # violation severity, minus a large one-off collision penalty.
        # This is intentionally simple -- see module docstring.
        reward = 0.1 - sum(v.severity for v in ego_violations)
        if collided:
            reward -= 10.0

        self.step_count += 1
        past_finish = self.ego_position_m > self.config.stop_line_distance_m + 50.0
        done = collided or self.step_count >= self.config.max_steps or past_finish

        info = {"violations": ego_violations, "collided": collided, "lead_speed_kmh": self.lead_speed_kmh}
        return ego_state, reward, done, info
