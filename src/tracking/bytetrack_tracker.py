"""
src/tracking/bytetrack_tracker.py

A self-contained implementation of the ByteTrack association strategy
(Zhang et al., 2022 -- https://arxiv.org/abs/2110.06864): a constant-
velocity Kalman filter per track, plus **two-stage** IoU matching that
first associates high-confidence detections, then makes a second pass
matching remaining low-confidence detections against tracks the first
pass couldn't explain -- ByteTrack's key insight for recovering
partially-occluded/blurred objects that a single-threshold tracker would
simply drop.

Deliberately NOT a wrapper around `ultralytics.trackers.BYTETracker`:
that class is an internal implementation detail of the `ultralytics`
package (tightly coupled to its own `Results`/`Boxes` objects, with an
API that has changed across releases and isn't published as a stable
public interface). Depending on it here would silently break this
platform on an unrelated `ultralytics` version bump. This module
implements the same published algorithm against our own stable
`Detection`/`Track` types instead, and is registered under the
`"tracker"` namespace as `"bytetrack"` (see `src.core.registry`) so it's
selected purely by `configs/tracker/bytetrack.yaml` -- swapping in a
different backend later never requires touching `stream_tracker.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import count
from typing import List, Optional, Tuple

import numpy as np

from src.core.registry import register
from src.core.types import BoundingBox, Detection, Track
from src.tracking.base_tracker import BaseTracker

logger = logging.getLogger(__name__)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two [N, 4] and [M, 4] arrays of (x1,y1,x2,y2)
    boxes, returned as an [N, M] matrix. Pure numpy -- no torch/torchvision
    dependency, so this tracker works even in a torch-free deployment
    (e.g. a CPU-only ONNX Runtime inference service)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _linear_assignment(cost: np.ndarray, cost_threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian-algorithm assignment with a cost cutoff.

    Returns:
        matches: list of (row_idx, col_idx) pairs kept (cost <= threshold).
        unmatched_rows: row indices with no acceptable match.
        unmatched_cols: column indices with no acceptable match.
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    from scipy.optimize import linear_sum_assignment

    row_idx, col_idx = linear_sum_assignment(cost)
    matches: List[Tuple[int, int]] = []
    unmatched_rows = set(range(cost.shape[0]))
    unmatched_cols = set(range(cost.shape[1]))

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= cost_threshold:
            matches.append((r, c))
            unmatched_rows.discard(r)
            unmatched_cols.discard(c)

    return matches, sorted(unmatched_rows), sorted(unmatched_cols)


class _ConstantVelocityKalmanBox:
    """8-state constant-velocity Kalman filter tracking a single box as
    `[cx, cy, w, h, vcx, vcy, vw, vh]`. Deliberately hand-rolled with plain
    numpy (rather than a `filterpy` dependency) -- the model is a textbook
    linear KF and small enough (8x8) that a dependency isn't worth it.
    """

    def __init__(self, bbox: BoundingBox, process_noise: float = 1.0, measurement_noise: float = 1.0) -> None:
        cx, cy = bbox.center
        w, h = bbox.width, bbox.height

        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 10.0

        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # position += velocity * dt (dt folded into velocity units)

        self.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

        self.Q = np.eye(8, dtype=np.float64) * process_noise
        self.R = np.eye(4, dtype=np.float64) * measurement_noise

    def predict(self) -> BoundingBox:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._state_to_bbox()

    def update(self, bbox: BoundingBox) -> None:
        cx, cy = bbox.center
        z = np.array([cx, cy, bbox.width, bbox.height], dtype=np.float64)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def velocity_px_per_frame(self) -> Tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))

    def _state_to_bbox(self) -> BoundingBox:
        cx, cy, w, h = self.x[0], self.x[1], max(self.x[2], 1e-3), max(self.x[3], 1e-3)
        return BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


@dataclass
class _TrackState:
    track_id: int
    kalman: _ConstantVelocityKalmanBox
    class_id: int
    class_name: str
    confidence: float
    age_frames: int = 0
    frames_since_update: int = 0
    hit_streak: int = 0
    confirmed: bool = False


@register("tracker", "bytetrack")
class ByteTrackTracker(BaseTracker):
    """ByteTrack-style multi-object tracker (see module docstring).

    Args (mirrors `configs/tracker/bytetrack.yaml`):
        track_thresh: detections at/above this confidence are "high
            score" and used for first-pass matching + spawning new tracks.
        low_thresh: detections between `low_thresh` and `track_thresh`
            are "low score" and only used in the second-pass recovery
            match (never spawn a new track on their own -- this is what
            keeps ByteTrack robust to noisy low-confidence boxes).
        match_thresh: minimum IoU to accept a match in either stage.
        track_buffer_frames: how many consecutive frames a track may go
            unmatched before it's dropped.
        min_hits_to_confirm: consecutive matched frames required before a
            new track is reported (suppresses one-frame false-positive
            detections from ever appearing as a track).
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer_frames: int = 30,
        min_hits_to_confirm: int = 3,
    ) -> None:
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer_frames = track_buffer_frames
        self.min_hits_to_confirm = min_hits_to_confirm

        self._tracks: List[_TrackState] = []
        self._id_counter = count(1)

    def reset(self) -> None:
        self._tracks = []
        self._id_counter = count(1)

    def update(self, detections: List[Detection], frame=None) -> List[Track]:
        # 1. Predict every existing track forward one frame.
        predicted_boxes = [t.kalman.predict() for t in self._tracks]

        high_dets = [d for d in detections if d.confidence >= self.track_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.track_thresh]

        track_boxes_arr = np.array(
            [b.to_xyxy() for b in predicted_boxes], dtype=np.float32
        ).reshape(-1, 4)

        # --- Stage 1: match HIGH-confidence detections to all tracks. ---
        high_boxes_arr = np.array([d.bbox.to_xyxy() for d in high_dets], dtype=np.float32).reshape(-1, 4)
        iou1 = _iou_matrix(track_boxes_arr, high_boxes_arr)
        matches1, unmatched_tracks, unmatched_high = _linear_assignment(
            1.0 - iou1, cost_threshold=1.0 - self.match_thresh
        )

        for track_idx, det_idx in matches1:
            self._apply_match(self._tracks[track_idx], high_dets[det_idx])

        # --- Stage 2: try to recover still-unmatched tracks using LOW-confidence detections. ---
        remaining_track_boxes = track_boxes_arr[unmatched_tracks] if unmatched_tracks else np.zeros((0, 4), dtype=np.float32)
        low_boxes_arr = np.array([d.bbox.to_xyxy() for d in low_dets], dtype=np.float32).reshape(-1, 4)
        iou2 = _iou_matrix(remaining_track_boxes, low_boxes_arr)
        matches2, still_unmatched_local, unmatched_low = _linear_assignment(
            1.0 - iou2, cost_threshold=1.0 - self.match_thresh
        )

        matched_track_indices = {t for t, _ in matches1}
        for local_idx, det_idx in matches2:
            track_idx = unmatched_tracks[local_idx]
            self._apply_match(self._tracks[track_idx], low_dets[det_idx])
            matched_track_indices.add(track_idx)

        # --- Age out tracks that stayed unmatched this frame. ---
        for i, track in enumerate(self._tracks):
            if i not in matched_track_indices:
                track.frames_since_update += 1
                track.hit_streak = 0

        self._tracks = [t for t in self._tracks if t.frames_since_update <= self.track_buffer_frames]

        # --- Spawn new tracks from HIGH-confidence detections nothing matched. ---
        for det_idx in unmatched_high:
            self._spawn_track(high_dets[det_idx])

        return self._collect_confirmed_tracks()

    def _apply_match(self, track: _TrackState, detection: Detection) -> None:
        track.kalman.update(detection.bbox)
        track.confidence = detection.confidence
        track.class_id = detection.class_id
        track.class_name = detection.class_name
        track.age_frames += 1
        track.frames_since_update = 0
        track.hit_streak += 1
        if track.hit_streak >= self.min_hits_to_confirm:
            track.confirmed = True

    def _spawn_track(self, detection: Detection) -> None:
        track = _TrackState(
            track_id=next(self._id_counter),
            kalman=_ConstantVelocityKalmanBox(detection.bbox),
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            age_frames=1,
            hit_streak=1,
            confirmed=self.min_hits_to_confirm <= 1,
        )
        self._tracks.append(track)

    def _collect_confirmed_tracks(self) -> List[Track]:
        results: List[Track] = []
        for t in self._tracks:
            if not t.confirmed or t.frames_since_update > 0:
                continue  # only report tracks matched THIS frame, once confirmed
            results.append(
                Track(
                    track_id=t.track_id,
                    bbox=t.kalman._state_to_bbox(),
                    confidence=t.confidence,
                    class_id=t.class_id,
                    class_name=t.class_name,
                    age_frames=t.age_frames,
                    velocity_px_per_frame=t.kalman.velocity_px_per_frame(),
                )
            )
        return results
