"""
src/rl/rules.py

Roadmap step 3 (rules half): hard physical/legal constraints the policy
must never violate, independent of what the Dueling DQN's Q-values say.
`policy_engine.py` applies these as an *action mask* (illegal actions get
-inf Q-value before argmax) rather than a reward penalty, since a
penalty only discourages a violation statistically over training whereas
a mask forbids it deterministically at inference time -- the right
choice for constraints with real-world safety consequences (stop-line
running, wrong-way driving, rear-ending a heavy vehicle).

Three rules, matching the roadmap:
    1. StopLineRule            -- zebra-crosswalk stop-line compliance.
    2. HeadingCorridorRule     -- heading-based lane/corridor detection.
    3. ElasticSATSeparation    -- SAT-based bumper-to-bumper minimum gap,
                                   with an "elastic" (speed-scaled) buffer
                                   for heavy vehicles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.rl.types import Action, OrientedBoundingBox, RuleName, RuleViolation, StateVector


@dataclass(frozen=True)
class StopLine:
    """A zebra-crosswalk stop-line, defined by its two endpoints on the
    ground plane plus the heading vehicles are expected to be traveling
    when approaching it (so "before" vs. "past" the line is well-defined
    for two-way roads)."""

    point_a_m: Tuple[float, float]
    point_b_m: Tuple[float, float]
    approach_heading_rad: float

    def signed_distance_ahead(self, position_m: Tuple[float, float]) -> float:
        """Distance from `position_m` to the line, positive = before the
        line (hasn't crossed yet), negative = past it. Projects onto the
        approach-heading axis rather than perpendicular distance to the
        segment, since what matters for stop-line compliance is
        along-lane progress, not lateral offset."""
        # Line's own along-line point (its midpoint) as the reference origin.
        mid = ((self.point_a_m[0] + self.point_b_m[0]) / 2.0, (self.point_a_m[1] + self.point_b_m[1]) / 2.0)
        dx, dy = position_m[0] - mid[0], position_m[1] - mid[1]
        heading_vec = (math.cos(self.approach_heading_rad), math.sin(self.approach_heading_rad))
        # Vehicle is "ahead of" the line (hasn't reached it) if it's
        # behind along the approach heading, i.e. negative projection.
        projection = dx * heading_vec[0] + dy * heading_vec[1]
        return -projection


class StopLineRule:
    """Zebra-crosswalk stop-line compliance: a vehicle within
    `approach_zone_m` of a stop line that hasn't dropped below
    `max_approach_speed_kmh` is flagged -- it's on track to run the line.
    """

    def __init__(self, approach_zone_m: float = 15.0, max_approach_speed_kmh: float = 15.0) -> None:
        self.approach_zone_m = approach_zone_m
        self.max_approach_speed_kmh = max_approach_speed_kmh

    def check(self, state: StateVector, stop_lines: Sequence[StopLine]) -> Optional[RuleViolation]:
        for line in stop_lines:
            dist_ahead = line.signed_distance_ahead(state.position_m)
            if 0.0 <= dist_ahead <= self.approach_zone_m and state.speed_kmh > self.max_approach_speed_kmh:
                severity = min(1.0, state.speed_kmh / (self.max_approach_speed_kmh * 3.0))
                return RuleViolation(
                    track_id=state.track_id,
                    rule=RuleName.STOP_LINE,
                    severity=severity,
                    detail=f"{dist_ahead:.1f}m from stop line at {state.speed_kmh:.1f} km/h",
                )
        return None


class HeadingCorridorRule:
    """Heading-based corridor/lane detection: flags a vehicle whose
    heading deviates from the corridor's allowed direction by more than
    `max_deviation_rad` -- catches wrong-way driving and lane departures
    without needing lane-line detection, just the corridor's nominal
    heading and a tolerance.
    """

    def __init__(self, max_deviation_rad: float = math.radians(45.0)) -> None:
        self.max_deviation_rad = max_deviation_rad

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Smallest signed difference between two angles, wrapped to
        [-pi, pi] so e.g. 179 deg vs. -179 deg reads as 2 deg apart, not
        358."""
        d = (a - b + math.pi) % (2 * math.pi) - math.pi
        return d

    def check(self, state: StateVector, corridor_heading_rad: float) -> Optional[RuleViolation]:
        if state.speed_kmh < 2.0:
            return None  # near-stationary vehicles have unreliable heading
        deviation = abs(self._angle_diff(state.heading_rad, corridor_heading_rad))
        if deviation > self.max_deviation_rad:
            severity = min(1.0, deviation / math.pi)
            return RuleViolation(
                track_id=state.track_id,
                rule=RuleName.HEADING_CORRIDOR,
                severity=severity,
                detail=f"heading off corridor by {math.degrees(deviation):.0f} deg",
            )
        return None


def _obb_axes(obb: OrientedBoundingBox) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """The two unit separating-axis candidates for an OBB under SAT: its
    own length-axis and width-axis directions."""
    c, s = math.cos(obb.heading_rad), math.sin(obb.heading_rad)
    return (c, s), (-s, c)


def _obb_corners(obb: OrientedBoundingBox) -> List[Tuple[float, float]]:
    cx, cy = obb.center_m
    hl, hw = obb.length_m / 2.0, obb.width_m / 2.0
    c, s = math.cos(obb.heading_rad), math.sin(obb.heading_rad)
    local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    return [(cx + lx * c - ly * s, cy + lx * s + ly * c) for lx, ly in local]


def sat_overlap(obb_a: OrientedBoundingBox, obb_b: OrientedBoundingBox) -> Tuple[bool, float]:
    """Separating Axis Theorem test for two oriented rectangles.

    Returns (is_overlapping, penetration_depth). `penetration_depth` is
    0.0 when not overlapping and the minimum-translation-vector magnitude
    (the smallest push needed to separate them) when they are -- used by
    `ElasticSATSeparation` as the raw "how bad is this gap violation" signal.
    """
    axes = [*_obb_axes(obb_a), *_obb_axes(obb_b)]
    corners_a = _obb_corners(obb_a)
    corners_b = _obb_corners(obb_b)

    min_overlap = math.inf
    for ax, ay in axes:
        proj_a = [px * ax + py * ay for px, py in corners_a]
        proj_b = [px * ax + py * ay for px, py in corners_b]
        a_min, a_max = min(proj_a), max(proj_a)
        b_min, b_max = min(proj_b), max(proj_b)
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap <= 0:
            return False, 0.0
        min_overlap = min(min_overlap, overlap)
    return True, min_overlap


def gap_along_heading(obb_a: OrientedBoundingBox, obb_b: OrientedBoundingBox) -> float:
    """Bumper-to-bumper gap along `obb_a`'s heading axis (the "following
    distance" a driver/policy cares about), not full 2D center distance.
    Positive when `obb_b` is ahead and clear, negative/zero implies
    contact or overlap along that axis."""
    c, s = math.cos(obb_a.heading_rad), math.sin(obb_a.heading_rad)
    dx = obb_b.center_m[0] - obb_a.center_m[0]
    dy = obb_b.center_m[1] - obb_a.center_m[1]
    center_dist_along_heading = dx * c + dy * s
    return center_dist_along_heading - (obb_a.length_m / 2.0) - (obb_b.length_m / 2.0)


class ElasticSATSeparation:
    """Bumper-to-bumper minimum separation, enforced via SAT overlap plus
    an "elastic" required gap that stretches with the following vehicle's
    speed (a time-headway model: the faster you're going, the larger the
    mandatory buffer, like a spring under load) -- with a larger base
    buffer and headway for heavy vehicles (buses/trucks), which need
    longer stopping distances.

    Args:
        base_gap_m: minimum static gap at near-zero speed.
        time_headway_s: seconds of following distance added per m/s of
            speed -- the "elastic" term.
        heavy_vehicle_multiplier: scales both `base_gap_m` and
            `time_headway_s` when the following vehicle is heavy.
    """

    def __init__(
        self,
        base_gap_m: float = 2.0,
        time_headway_s: float = 1.5,
        heavy_vehicle_multiplier: float = 1.8,
    ) -> None:
        self.base_gap_m = base_gap_m
        self.time_headway_s = time_headway_s
        self.heavy_vehicle_multiplier = heavy_vehicle_multiplier

    def required_gap_m(self, following: StateVector) -> float:
        speed_mps = following.speed_kmh / 3.6
        multiplier = self.heavy_vehicle_multiplier if following.is_heavy_vehicle else 1.0
        return multiplier * (self.base_gap_m + self.time_headway_s * speed_mps)

    def check_pair(self, following: StateVector, lead: StateVector) -> Optional[RuleViolation]:
        obb_follow = OrientedBoundingBox.from_state(following)
        obb_lead = OrientedBoundingBox.from_state(lead)

        overlapping, penetration = sat_overlap(obb_follow, obb_lead)
        gap = gap_along_heading(obb_follow, obb_lead)
        required = self.required_gap_m(following)

        if overlapping or gap < required:
            deficit = required - gap if not overlapping else required + penetration
            severity = float(np.clip(deficit / max(required, 1e-6), 0.0, 1.0))
            return RuleViolation(
                track_id=following.track_id,
                rule=RuleName.ELASTIC_SEPARATION,
                severity=severity,
                detail=f"gap {gap:.1f}m < required {required:.1f}m behind track {lead.track_id}",
            )
        return None


class RuleEnforcer:
    """Aggregates all three rules and turns their output into both a flat
    violation list (for logging/reward shaping) and a hard action mask
    (for `policy_engine.py`'s Q-value masking).
    """

    def __init__(
        self,
        stop_line_rule: Optional[StopLineRule] = None,
        corridor_rule: Optional[HeadingCorridorRule] = None,
        separation_rule: Optional[ElasticSATSeparation] = None,
    ) -> None:
        self.stop_line_rule = stop_line_rule or StopLineRule()
        self.corridor_rule = corridor_rule or HeadingCorridorRule()
        self.separation_rule = separation_rule or ElasticSATSeparation()

    def evaluate(
        self,
        states: Sequence[StateVector],
        stop_lines: Sequence[StopLine] = (),
        corridor_heading_rad: Optional[float] = None,
    ) -> List[RuleViolation]:
        violations: List[RuleViolation] = []
        for state in states:
            if stop_lines:
                v = self.stop_line_rule.check(state, stop_lines)
                if v:
                    violations.append(v)
            if corridor_heading_rad is not None:
                v = self.corridor_rule.check(state, corridor_heading_rad)
                if v:
                    violations.append(v)

        # Pairwise separation check: naive O(n^2), acceptable at
        # intersection-scale vehicle counts (tens, not thousands); for
        # larger scenes, pre-filter pairs with the same broad-phase cull
        # idea used in lidar_sim.py before calling check_pair.
        for i, following in enumerate(states):
            for lead in states:
                if lead.track_id == following.track_id:
                    continue
                # Only check vehicles roughly ahead along `following`'s
                # heading -- avoids flagging side-by-side or oncoming
                # traffic as a following-gap violation.
                if gap_along_heading(OrientedBoundingBox.from_state(following), OrientedBoundingBox.from_state(lead)) < 0 and \
                   gap_along_heading(OrientedBoundingBox.from_state(lead), OrientedBoundingBox.from_state(following)) < 0:
                    continue  # side-by-side / overlapping heading axes both ways: not a following pair
                v = self.separation_rule.check_pair(following, lead)
                if v:
                    violations.append(v)
        return violations

    def action_mask(self, state: StateVector, violations: Sequence[RuleViolation]) -> np.ndarray:
        """Boolean mask over `Action`, True = allowed. A vehicle with any
        violation concerning it loses ACCELERATE and lane-change actions
        (don't compound an active violation by speeding up or swerving);
        a severe (>0.66) violation additionally forces HARD_BRAKE as the
        only legal action.
        """
        mask = np.ones(len(Action), dtype=bool)
        own_violations = [v for v in violations if v.track_id == state.track_id]
        if not own_violations:
            return mask

        mask[Action.ACCELERATE] = False
        mask[Action.LANE_CHANGE_LEFT] = False
        mask[Action.LANE_CHANGE_RIGHT] = False

        max_severity = max(v.severity for v in own_violations)
        if max_severity > 0.66:
            mask[:] = False
            mask[Action.HARD_BRAKE] = True
        return mask
