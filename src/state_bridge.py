"""
src/rl/state_bridge.py

Roadmap step 1: Vision-to-State Feature Bridge.

Turns this platform's existing ground-truth perception outputs
(`TrackedVehicle`/`FrameResult` from `src.pipelines.stream_tracker`,
carrying `Track.bbox` in pixel space plus `Track.velocity_px_per_frame`
from `ByteTrackTracker`'s Kalman filter) into `StateVector`s -- the $S_t$
inputs consumed by `dueling_dqn.py`.

Deliberately reuses `src.tracking.speed_estimator.BaseSpeedEstimator`
rather than re-deriving a pixel->meter conversion here: both
`SimpleSpeedEstimator` (fixed overhead camera) and
`HomographySpeedEstimator` (oblique CCTV) already solve "pixel velocity
-> real-world magnitude"; this module only adds the parts they don't
cover -- ground-plane *position* (not just velocity magnitude), heading,
and physical vehicle extents -- and packages everything into one
`StateVector` per tracked vehicle.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.types import FrameResult, TrackedVehicle
from src.rl.types import LidarScan, StateVector
from src.tracking.speed_estimator import BaseSpeedEstimator

# COCO/Ultralytics-style class names this platform's YOLO configs use for
# heavy vehicles -- see configs/data/vehicles.yaml. Kept as a module-level
# default (overridable per bridge instance) rather than hardcoded inside
# `is_heavy`, since a retrained detector may use different class names.
DEFAULT_HEAVY_CLASS_NAMES = frozenset({"bus", "truck"})

# Rough physical vehicle footprints (length_m, width_m) used when a
# per-class calibrated size table isn't supplied -- adequate for OBB/SAT
# separation checks, which care more about "roughly car-sized vs.
# roughly truck-sized" than centimeter accuracy.
DEFAULT_CLASS_EXTENTS_M: Dict[str, Tuple[float, float]] = {
    "car": (4.5, 1.8),
    "motorcycle": (2.0, 0.8),
    "bus": (12.0, 2.55),
    "truck": (8.0, 2.5),
}
_FALLBACK_EXTENTS_M = (4.5, 1.8)  # treat unknown classes as car-sized


class VisionStateBridge:
    """Maps camera/image-coordinate perception output into the ground-plane
    `StateVector` state space fed into the Dueling DQN.

    Args:
        speed_estimator: pixel-velocity -> km/h converter. Use
            `SimpleSpeedEstimator` for a fronto-parallel/overhead camera or
            `HomographySpeedEstimator` for oblique CCTV (see
            `src/tracking/speed_estimator.py`'s module docstring for when
            each applies).
        fps: source video frame rate, needed because `Track.velocity_px_per_frame`
            is a per-frame (not per-second) displacement.
        pixels_per_meter: required when `speed_estimator` is a
            `SimpleSpeedEstimator`, so this bridge can also convert bbox
            *position* (not just velocity) to ground-plane meters using the
            same fronto-parallel assumption. Ignored if `homography` is given.
        homography: optional 3x3 pixel->ground-plane homography (the same
            matrix passed to `HomographySpeedEstimator`). When given, this
            bridge maps bbox centers through it directly for position,
            which is exact for the oblique-camera case rather than the
            fronto-parallel approximation `pixels_per_meter` implies.
        heavy_class_names: class names treated as "heavy vehicle" for the
            SAT elastic-separation rule in `rules.py`.
        class_extents_m: per-class (length_m, width_m) lookup for the
            oriented bounding boxes used by the separation rule.
    """

    def __init__(
        self,
        speed_estimator: BaseSpeedEstimator,
        fps: float,
        pixels_per_meter: Optional[float] = None,
        homography: Optional[Sequence[Sequence[float]]] = None,
        heavy_class_names: frozenset = DEFAULT_HEAVY_CLASS_NAMES,
        class_extents_m: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive.")
        if pixels_per_meter is None and homography is None:
            raise ValueError(
                "VisionStateBridge needs either pixels_per_meter or a "
                "homography matrix to convert pixel positions to ground-"
                "plane meters."
            )
        self.speed_estimator = speed_estimator
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self._H = None
        if homography is not None:
            # Imported lazily / only when used, matching
            # HomographySpeedEstimator's own lazy-numpy pattern so this
            # module stays importable without numpy for the
            # pixels_per_meter-only path.
            import numpy as np

            self._H = np.asarray(homography, dtype=float)
            if self._H.shape != (3, 3):
                raise ValueError(f"homography must be 3x3, got {self._H.shape}.")
        self.heavy_class_names = heavy_class_names
        self.class_extents_m = class_extents_m or DEFAULT_CLASS_EXTENTS_M

    def _pixel_to_ground(self, point_px: Tuple[float, float]) -> Tuple[float, float]:
        if self._H is not None:
            import numpy as np

            vec = self._H @ np.array([point_px[0], point_px[1], 1.0])
            return float(vec[0] / vec[2]), float(vec[1] / vec[2])
        # Fronto-parallel fallback: same assumption SimpleSpeedEstimator
        # makes, just applied to position instead of velocity magnitude.
        return point_px[0] / self.pixels_per_meter, point_px[1] / self.pixels_per_meter

    def build_state(
        self,
        vehicle: TrackedVehicle,
        timestamp_s: float = 0.0,
        lidar: Optional[LidarScan] = None,
        dist_to_stop_line_m: Optional[float] = None,
    ) -> StateVector:
        """Convert one `TrackedVehicle` into a `StateVector`.

        Heading is derived from the tracker's pixel-space velocity vector
        (available whenever `Track.velocity_px_per_frame` is set, i.e.
        after the Kalman filter has at least two associated frames);
        vehicles with no velocity yet (age_frames == 0) get heading 0.0
        and speed 0.0 rather than raising, since "just appeared, unknown
        heading" is a normal transient state the policy engine should
        still be able to act on (e.g. via the LiDAR/position features
        alone).
        """
        track = vehicle.track
        cx_px, cy_px = track.bbox.center
        position_m = self._pixel_to_ground((cx_px, cy_px))

        if track.velocity_px_per_frame is not None:
            vx_px, vy_px = track.velocity_px_per_frame
            speed_kmh = self.speed_estimator.estimate_kmh(track.velocity_px_per_frame, self.fps)
            # Ground-plane heading: reuse the same pixel->ground mapping
            # applied to a point one frame-step ahead, so an oblique
            # homography's shear doesn't distort the angle silently.
            ahead_px = (cx_px + vx_px, cy_px + vy_px)
            ahead_m = self._pixel_to_ground(ahead_px)
            dx, dy = ahead_m[0] - position_m[0], ahead_m[1] - position_m[1]
            heading_rad = math.atan2(dy, dx) if (dx or dy) else 0.0
            speed_mps = speed_kmh / 3.6
            velocity_mps = (speed_mps * math.cos(heading_rad), speed_mps * math.sin(heading_rad))
        else:
            heading_rad = 0.0
            speed_kmh = 0.0
            velocity_mps = (0.0, 0.0)

        length_m, width_m = self.class_extents_m.get(track.class_name, _FALLBACK_EXTENTS_M)

        return StateVector(
            track_id=track.track_id,
            position_m=position_m,
            velocity_mps=velocity_mps,
            heading_rad=heading_rad,
            speed_kmh=speed_kmh,
            length_m=length_m,
            width_m=width_m,
            class_id=track.class_id,
            class_name=track.class_name,
            is_heavy_vehicle=track.class_name in self.heavy_class_names,
            lidar=lidar,
            dist_to_stop_line_m=dist_to_stop_line_m,
            timestamp_s=timestamp_s,
        )

    def build_states(
        self,
        frame: FrameResult,
        lidar_by_track_id: Optional[Dict[int, LidarScan]] = None,
        dist_to_stop_line_by_track_id: Optional[Dict[int, float]] = None,
    ) -> List[StateVector]:
        """Batch form of `build_state` over every vehicle in a `FrameResult`
        -- the usual call site (once per processed frame, feeding
        `policy_engine.py`)."""
        lidar_by_track_id = lidar_by_track_id or {}
        dist_map = dist_to_stop_line_by_track_id or {}
        return [
            self.build_state(
                vehicle,
                timestamp_s=frame.timestamp_s,
                lidar=lidar_by_track_id.get(vehicle.track.track_id),
                dist_to_stop_line_m=dist_map.get(vehicle.track.track_id),
            )
            for vehicle in frame.vehicles
        ]
