"""
src/utils/video_utils.py

Small, dependency-light helpers shared by the streaming pipeline and the
dashboard: a rolling FPS counter and an overlay renderer for
`TrackedVehicle`s. Kept separate from `stream_tracker.py` so the drawing
code can be unit tested (and reused by the Streamlit dashboard) without
importing threading/queue machinery.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Iterable

import numpy as np

from src.core.types import TrackedVehicle


class FPSCounter:
    """Rolling-window FPS estimator. Call `tick()` once per processed
    frame; `fps` reflects the average over the last `window_size` ticks."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self._timestamps: Deque[float] = deque(maxlen=window_size)

    def tick(self) -> None:
        self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


def draw_tracked_vehicles(frame: np.ndarray, vehicles: Iterable[TrackedVehicle]) -> np.ndarray:
    """Draw bounding boxes, track IDs, speed, and plate text onto a copy of
    `frame`. Returns the annotated copy (input frame is not mutated)."""
    import cv2

    annotated = frame.copy()
    for vehicle in vehicles:
        track = vehicle.track
        x1, y1, x2, y2 = (int(v) for v in track.bbox.to_xyxy())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color=(0, 220, 0), thickness=2)

        label_parts = [f"#{track.track_id}", track.class_name, f"{track.confidence:.2f}"]
        if track.speed_kmh is not None:
            label_parts.append(f"{track.speed_kmh:.0f} km/h")
        if vehicle.plate is not None:
            label_parts.append(vehicle.plate.text)

        label = " | ".join(label_parts)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), (0, 220, 0), thickness=-1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    return annotated
