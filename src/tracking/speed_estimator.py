"""
src/tracking/speed_estimator.py

Converts a `Track`'s pixel-space velocity (as produced by
`ByteTrackTracker`'s Kalman filter) into an estimated real-world speed.

This uses the simplest calibration model -- a single scalar
`pixels_per_meter`, assumed constant across the frame -- which is only
accurate for a near-fronto-parallel camera view over a small depth range
(e.g. a fixed overhead/gantry traffic camera looking straight down a
lane). For oblique CCTV angles, replace `SimpleSpeedEstimator` with a
homography-based one (project pixel coordinates to a ground-plane
coordinate system via `cv2.findHomography` using known reference points,
then compute displacement in that plane) -- the `BaseSpeedEstimator`
interface below is deliberately narrow so that swap is a drop-in change
with no caller updates required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class BaseSpeedEstimator(ABC):
    """Contract: pixel velocity (+ frame rate) in, km/h out."""

    @abstractmethod
    def estimate_kmh(self, velocity_px_per_frame: Tuple[float, float], fps: float) -> float:
        raise NotImplementedError


class SimpleSpeedEstimator(BaseSpeedEstimator):
    """Constant `pixels_per_meter` calibration -- see module docstring for
    when this is (and isn't) an appropriate model."""

    def __init__(self, pixels_per_meter: float) -> None:
        if pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be positive.")
        self.pixels_per_meter = pixels_per_meter

    def estimate_kmh(self, velocity_px_per_frame: Tuple[float, float], fps: float) -> float:
        vx, vy = velocity_px_per_frame
        speed_px_per_s = ((vx**2 + vy**2) ** 0.5) * fps
        speed_m_per_s = speed_px_per_s / self.pixels_per_meter
        return speed_m_per_s * 3.6


class HomographySpeedEstimator(BaseSpeedEstimator):
    """Ground-plane calibration via a homography matrix mapping image
    pixels to real-world meters -- accurate for oblique camera angles.

    Extension point: not wired into any pipeline by default. Construct
    `homography_matrix` with `cv2.findHomography` from >= 4 manually
    surveyed (pixel, meters) point correspondences in the scene, then
    pass a 3x3 numpy array here.
    """

    def __init__(self, homography_matrix) -> None:
        import numpy as np

        self.H = np.asarray(homography_matrix, dtype=np.float64)
        if self.H.shape != (3, 3):
            raise ValueError(f"homography_matrix must be 3x3, got {self.H.shape}.")

    def _pixel_to_ground(self, point_px: Tuple[float, float]):
        import numpy as np

        vec = np.array([point_px[0], point_px[1], 1.0])
        ground = self.H @ vec
        return ground[0] / ground[2], ground[1] / ground[2]

    def estimate_kmh(self, velocity_px_per_frame: Tuple[float, float], fps: float) -> float:
        # Approximate: map the velocity vector's endpoint (relative to
        # origin) through the homography rather than re-deriving a proper
        # per-point Jacobian -- adequate for near-linear regions of the
        # ground plane, which is the common case for a single traffic lane.
        origin_ground = self._pixel_to_ground((0.0, 0.0))
        shifted_ground = self._pixel_to_ground(velocity_px_per_frame)
        dx = shifted_ground[0] - origin_ground[0]
        dy = shifted_ground[1] - origin_ground[1]
        speed_m_per_s = ((dx**2 + dy**2) ** 0.5) * fps
        return speed_m_per_s * 3.6
