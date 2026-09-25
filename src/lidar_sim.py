"""
src/rl/lidar_sim.py

Roadmap step 2: Perception + Simulation Sensor Fusion.

Simulates a 9-ray LiDAR sweep against the *same* ground-plane obstacle
set produced by vision (`StateVector`s from `state_bridge.py`), then
fuses the two modalities: a track that's both seen by the camera and
struck by a nearby ray is "confirmed", a ray hit with no matching vision
track is "lidar-only" (e.g. camera occlusion), and a vision track with no
corresponding ray hit is "vision-only" (e.g. outside the simulated LiDAR
FOV/range). This mirrors why real AV stacks fuse camera + LiDAR: each
modality's blind spots are the other's strength.

Performance: both raycasting and fusion apply bounding-box culling
before any per-ray/per-track geometry test, which is what keeps this
cheap enough for a 60 FPS decision loop -- see `_broad_phase_cull`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.rl.types import LidarScan, OrientedBoundingBox, StateVector


def _obb_aabb(obb: OrientedBoundingBox) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding radius box around an OBB (conservative: uses
    the diagonal as the half-extent in both axes), used only for the cheap
    broad-phase cull below -- never for the actual hit test."""
    half_diag = math.hypot(obb.length_m, obb.width_m) / 2.0
    cx, cy = obb.center_m
    return (cx - half_diag, cy - half_diag, cx + half_diag, cy + half_diag)


def _broad_phase_cull(
    origin_m: Tuple[float, float],
    max_range_m: float,
    obstacles: Sequence[OrientedBoundingBox],
) -> List[int]:
    """Bounding-box culling: return indices of obstacles whose AABB
    intersects the ego's max-range square, skipping the expensive
    per-ray slab/edge intersection test for everything else. This is the
    step that keeps `cast()` and `fuse()` cheap enough for a 60 FPS loop
    with dozens of tracked vehicles -- without it, cost is O(rays *
    obstacles) exact geometry tests every frame; with it, distant
    obstacles are rejected with 4 float comparisons instead.
    """
    ox, oy = origin_m
    ex1, ey1, ex2, ey2 = ox - max_range_m, oy - max_range_m, ox + max_range_m, oy + max_range_m
    kept: List[int] = []
    for i, obs in enumerate(obstacles):
        bx1, by1, bx2, by2 = _obb_aabb(obs)
        if bx2 < ex1 or bx1 > ex2 or by2 < ey1 or by1 > ey2:
            continue
        kept.append(i)
    return kept


def _obb_edges(obb: OrientedBoundingBox) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """The 4 edges of an oriented rectangle as (start, end) point pairs,
    for ray-segment intersection."""
    cx, cy = obb.center_m
    hl, hw = obb.length_m / 2.0, obb.width_m / 2.0
    cos_h, sin_h = math.cos(obb.heading_rad), math.sin(obb.heading_rad)
    local_corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    corners = [(cx + lx * cos_h - ly * sin_h, cy + lx * sin_h + ly * cos_h) for lx, ly in local_corners]
    return [(corners[i], corners[(i + 1) % 4]) for i in range(4)]


def _ray_segment_intersection(
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    seg_start: Tuple[float, float],
    seg_end: Tuple[float, float],
) -> Optional[float]:
    """Standard 2D ray/segment intersection. Returns the distance along
    `direction` (unit vector) to the intersection, or None if the ray
    (t >= 0) doesn't cross the segment (0 <= u <= 1)."""
    ox, oy = origin
    dx, dy = direction
    x1, y1 = seg_start
    x2, y2 = seg_end
    sx, sy = x2 - x1, y2 - y1

    denom = dx * sy - dy * sx
    if abs(denom) < 1e-9:
        return None  # parallel

    ex, ey = x1 - ox, y1 - oy
    t = (ex * sy - ey * sx) / denom
    u = (ex * dy - ey * dx) / denom
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


@dataclass(frozen=True)
class FusedTrack:
    """One track's fused vision + LiDAR confirmation status."""

    track_id: int
    vision_seen: bool
    lidar_seen: bool
    lidar_range_m: Optional[float]

    @property
    def confirmed(self) -> bool:
        """Both modalities agree this obstacle exists -- highest-confidence
        case, safe to weight most heavily downstream (e.g. in
        `rules.py`'s separation check)."""
        return self.vision_seen and self.lidar_seen


class LidarRaycaster:
    """Simulates a fixed-FOV, fixed-range 9-ray LiDAR sweep against a set
    of ground-plane oriented bounding boxes.

    Args:
        num_rays: ray count. Roadmap specifies 9.
        fov_deg: total field of view, centered on the ego heading.
        max_range_m: rays beyond this report no hit (`ranges_m` entry ==
            `max_range_m`, matching real LiDAR "no return" convention).
    """

    def __init__(self, num_rays: int = 9, fov_deg: float = 180.0, max_range_m: float = 60.0) -> None:
        if num_rays < 1:
            raise ValueError("num_rays must be >= 1.")
        if max_range_m <= 0:
            raise ValueError("max_range_m must be positive.")
        self.num_rays = num_rays
        self.fov_rad = math.radians(fov_deg)
        self.max_range_m = max_range_m

    def _ray_offsets(self) -> Tuple[float, ...]:
        """Relative angle offsets from ego heading, evenly spaced across
        the FOV (a single centered ray when num_rays == 1)."""
        if self.num_rays == 1:
            return (0.0,)
        half = self.fov_rad / 2.0
        step = self.fov_rad / (self.num_rays - 1)
        return tuple(-half + i * step for i in range(self.num_rays))

    def cast(
        self,
        origin_m: Tuple[float, float],
        heading_rad: float,
        obstacles: Sequence[OrientedBoundingBox],
        obstacle_track_ids: Sequence[int],
    ) -> LidarScan:
        """Cast all rays from `origin_m` (typically the ego/AV's own
        position) and return the nearest hit per ray.

        `obstacles` and `obstacle_track_ids` must be parallel sequences
        (index i's box belongs to track id i).
        """
        if len(obstacles) != len(obstacle_track_ids):
            raise ValueError("obstacles and obstacle_track_ids must be the same length.")

        candidate_idx = _broad_phase_cull(origin_m, self.max_range_m, obstacles)
        candidate_edges = [(i, _obb_edges(obstacles[i])) for i in candidate_idx]

        angles: List[float] = []
        ranges: List[float] = []
        hit_ids: List[Optional[int]] = []

        for offset in self._ray_offsets():
            angle = heading_rad + offset
            direction = (math.cos(angle), math.sin(angle))
            best_dist = self.max_range_m
            best_track_id: Optional[int] = None
            for idx, edges in candidate_edges:
                for seg_start, seg_end in edges:
                    dist = _ray_segment_intersection(origin_m, direction, seg_start, seg_end)
                    if dist is not None and dist < best_dist:
                        best_dist = dist
                        best_track_id = obstacle_track_ids[idx]
            angles.append(angle)
            ranges.append(best_dist)
            hit_ids.append(best_track_id)

        return LidarScan(
            ray_angles_rad=tuple(angles),
            ranges_m=tuple(ranges),
            hit_track_ids=tuple(hit_ids),
            max_range_m=self.max_range_m,
        )


class VisionLidarFusion:
    """Confirms or flags each vision track against a simulated LiDAR scan.

    Args:
        confirmation_tolerance_m: max discrepancy between a ray's reported
            hit range and the vision-derived distance to that same track
            for the two modalities to be considered "in agreement".
    """

    def __init__(self, confirmation_tolerance_m: float = 1.5) -> None:
        self.confirmation_tolerance_m = confirmation_tolerance_m

    def fuse(
        self,
        origin_m: Tuple[float, float],
        vision_states: Sequence[StateVector],
        scan: LidarScan,
    ) -> Dict[int, FusedTrack]:
        lidar_range_by_track: Dict[int, float] = {}
        for track_id, rng in zip(scan.hit_track_ids, scan.ranges_m):
            if track_id is not None:
                lidar_range_by_track[track_id] = min(lidar_range_by_track.get(track_id, rng), rng)

        fused: Dict[int, FusedTrack] = {}
        for state in vision_states:
            vision_dist = math.hypot(state.position_m[0] - origin_m[0], state.position_m[1] - origin_m[1])
            lidar_range = lidar_range_by_track.get(state.track_id)
            lidar_seen = lidar_range is not None and abs(lidar_range - vision_dist) <= self.confirmation_tolerance_m
            fused[state.track_id] = FusedTrack(
                track_id=state.track_id,
                vision_seen=True,
                lidar_seen=lidar_seen,
                lidar_range_m=lidar_range,
            )
        # Any LiDAR hit not matched to a vision track (occluded/outside
        # camera FOV) is still reported, with vision_seen=False, so the
        # policy engine can react to it even though no bbox exists for it.
        for track_id, lidar_range in lidar_range_by_track.items():
            if track_id not in fused:
                fused[track_id] = FusedTrack(
                    track_id=track_id, vision_seen=False, lidar_seen=True, lidar_range_m=lidar_range
                )
        return fused
