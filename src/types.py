"""
src/rl/types.py

Framework-agnostic dataclasses for the RL subsystem, kept dependency-free
(stdlib + plain tuples/floats only) for the same reason as
`src.core.types`: they need to be importable and unit-testable without
torch, and losslessly convertible to whatever tensor library
`dueling_dqn.py` uses.

Data flow:
    TrackedVehicle (src.core.types)  --state_bridge-->  StateVector
    StateVector (+ LidarScan)        --dueling_dqn-->   Q-values
    StateVector + neighbors          --rules-->         RuleViolation*
    Q-values + violations            --policy_engine--> Action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, Tuple


class Action(IntEnum):
    """Discrete action space consumed by the Dueling DQN's advantage head.

    Ordering is fixed and load-bearing: `dueling_dqn.py`'s output layer
    width equals `len(Action)`, and `rules.py::action_mask` returns a
    boolean vector indexed positionally by these values.
    """

    HARD_BRAKE = 0
    DECELERATE = 1
    MAINTAIN_SPEED = 2
    ACCELERATE = 3
    LANE_CHANGE_LEFT = 4
    LANE_CHANGE_RIGHT = 5


class RuleName(str, Enum):
    STOP_LINE = "stop_line_violation"
    HEADING_CORRIDOR = "heading_corridor_violation"
    ELASTIC_SEPARATION = "elastic_separation_violation"


@dataclass(frozen=True)
class LidarScan:
    """Simulated 9-ray LiDAR sweep, ego-centric.

    `ray_angles_rad` are absolute world-frame angles (ego heading already
    applied), not relative offsets, so a consumer never needs the ego
    heading to interpret a hit direction.
    """

    ray_angles_rad: Tuple[float, ...]
    ranges_m: Tuple[float, ...]
    hit_track_ids: Tuple[Optional[int], ...]
    max_range_m: float

    def __post_init__(self) -> None:
        n = len(self.ray_angles_rad)
        if not (len(self.ranges_m) == len(self.hit_track_ids) == n):
            raise ValueError(
                "ray_angles_rad, ranges_m, and hit_track_ids must be the "
                f"same length; got {n}, {len(self.ranges_m)}, "
                f"{len(self.hit_track_ids)}."
            )

    @property
    def num_rays(self) -> int:
        return len(self.ray_angles_rad)

    def normalized_ranges(self) -> Tuple[float, ...]:
        """Ranges in [0, 1], 1.0 meaning "no obstacle within max range" --
        the encoding `dueling_dqn.py`'s input stage expects."""
        return tuple(min(r, self.max_range_m) / self.max_range_m for r in self.ranges_m)


@dataclass(frozen=True)
class StateVector:
    """One vehicle's fused perception state at a single timestep -- the
    $S_t$ fed into the Dueling DQN's input embedding stage.

    Positions/velocities are in ground-plane meters (post vision->state
    bridging), not raw pixels, so this type is calibration-independent.
    """

    track_id: int
    position_m: Tuple[float, float]
    velocity_mps: Tuple[float, float]
    heading_rad: float
    speed_kmh: float
    length_m: float
    width_m: float
    class_id: int
    class_name: str
    is_heavy_vehicle: bool
    lidar: Optional[LidarScan] = None
    dist_to_stop_line_m: Optional[float] = None
    timestamp_s: float = 0.0

    def to_feature_tuple(self) -> Tuple[float, ...]:
        """Flat, fixed-order numeric encoding for the network's input
        stage. Kept as a plain tuple (not a tensor) so this module never
        needs to import torch."""
        lidar_features = self.lidar.normalized_ranges() if self.lidar is not None else (1.0,) * 9
        return (
            self.position_m[0],
            self.position_m[1],
            self.velocity_mps[0],
            self.velocity_mps[1],
            self.heading_rad,
            self.speed_kmh / 120.0,  # rough normalization, ~highway max
            self.length_m,
            self.width_m,
            float(self.is_heavy_vehicle),
            (self.dist_to_stop_line_m or 999.0) / 100.0,
            *lidar_features,
        )


@dataclass(frozen=True)
class RuleViolation:
    track_id: int
    rule: RuleName
    severity: float  # 0..1, used to weight action-mask strictness
    detail: str = ""


@dataclass(frozen=True)
class OrientedBoundingBox:
    """Ground-plane oriented bounding box (center + heading + extents),
    used by `rules.py`'s SAT separation check. Distinct from
    `src.core.types.BoundingBox`, which is axis-aligned pixel space."""

    center_m: Tuple[float, float]
    heading_rad: float
    length_m: float
    width_m: float

    @classmethod
    def from_state(cls, state: StateVector) -> "OrientedBoundingBox":
        return cls(
            center_m=state.position_m,
            heading_rad=state.heading_rad,
            length_m=state.length_m,
            width_m=state.width_m,
        )
