"""
src/tracking/botsort_tracker.py

BoT-SORT (Aharon et al., 2022) extends ByteTrack with two additions:
    1. Camera motion compensation (global motion estimated via sparse
       optical flow / ECC image alignment between consecutive frames)
       before IoU matching, so a panning/vibrating camera doesn't
       masquerade as object motion.
    2. Appearance (Re-ID) embedding similarity, fused with IoU cost, to
       re-identify objects across longer occlusions than motion alone
       can bridge.

This module is an ARCHITECTURAL EXTENSION POINT, not a fully-verified
implementation: it inherits ByteTrack's two-stage IoU matching as-is and
adds the camera-motion-compensation hook, but the Re-ID appearance branch
requires a trained embedding network (e.g. a small ResNet trained with a
triplet/ArcFace loss on vehicle re-identification data) that is outside
the scope of this baseline. `_extract_appearance_embedding` is the single
method to implement to complete it -- once it returns real embeddings,
`_appearance_cost` will already fuse them into the matching cost.

Registered under `("tracker", "botsort")` so it's selectable via
`configs/tracker/botsort.yaml` once completed, without touching
`stream_tracker.py`.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from src.core.registry import register
from src.core.types import Detection, Track
from src.tracking.bytetrack_tracker import ByteTrackTracker

logger = logging.getLogger(__name__)


@register("tracker", "botsort")
class BotSortTracker(ByteTrackTracker):
    """BoT-SORT tracker. See module docstring for implementation status."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer_frames: int = 30,
        min_hits_to_confirm: int = 3,
        use_camera_motion_compensation: bool = True,
        appearance_weight: float = 0.0,
    ) -> None:
        super().__init__(
            track_thresh=track_thresh,
            low_thresh=low_thresh,
            match_thresh=match_thresh,
            track_buffer_frames=track_buffer_frames,
            min_hits_to_confirm=min_hits_to_confirm,
        )
        self.use_camera_motion_compensation = use_camera_motion_compensation
        # 0.0 disables the (unimplemented) appearance branch entirely,
        # making this behave as plain ByteTrack until Re-ID is wired in.
        self.appearance_weight = appearance_weight
        self._prev_gray_frame: Optional[np.ndarray] = None

    def update(self, detections: List[Detection], frame=None) -> List[Track]:
        if self.use_camera_motion_compensation and frame is not None:
            self._compensate_camera_motion(frame)
        if self.appearance_weight > 0.0 and frame is not None:
            logger.warning(
                "BotSortTracker.appearance_weight > 0 but no Re-ID model is "
                "wired in; falling back to motion-only matching. Implement "
                "`_extract_appearance_embedding` to enable the appearance branch."
            )
        return super().update(detections, frame=frame)

    def _compensate_camera_motion(self, frame: np.ndarray) -> None:
        """Estimate global camera motion via ECC image alignment and shift
        every track's Kalman state by the inverse transform, so subsequent
        IoU matching compares against motion-compensated predicted boxes.
        """
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._prev_gray_frame is None:
            self._prev_gray_frame = gray
            return

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            _, warp_matrix = cv2.findTransformECC(
                self._prev_gray_frame, gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
            )
        except cv2.error:
            logger.debug("ECC motion compensation failed to converge this frame; skipping.")
            self._prev_gray_frame = gray
            return

        dx, dy = warp_matrix[0, 2], warp_matrix[1, 2]
        for track in self._tracks:
            track.kalman.x[0] -= dx
            track.kalman.x[1] -= dy

        self._prev_gray_frame = gray

    def _extract_appearance_embedding(self, frame: np.ndarray, detection: Detection) -> Optional[np.ndarray]:
        """Extension point: crop `detection.bbox` from `frame` and run a
        Re-ID embedding network, returning an L2-normalized feature
        vector. Returning `None` (the default) disables the appearance
        branch for that detection."""
        return None
