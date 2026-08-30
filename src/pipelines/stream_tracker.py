"""
src/pipelines/stream_tracker.py

Multi-threaded RTSP/CCTV inference pipeline: decouples frame CAPTURE
(network/IO-bound, must never block on a slow model) from INFERENCE
(CPU/GPU-bound, must never block on a slow/flaky network) using the
classic producer/consumer pattern with a latest-frame-only buffer.

Threading model, per stream:

    [FrameGrabber thread]  --frame-->  [Queue(maxsize=1),
                                          overwrite-on-full]
                                              |
                                              v
    [inference thread]  detect -> track -> speed/counting -> FrameResult
                                              |
                                              v
                                   [results Queue -> consumers]

Rationale for "overwrite-on-full" rather than a normal bounded queue:
for a *live* traffic feed, a stale frame is worse than a dropped one --
if inference briefly falls behind, we want the next inference call to
run on the newest available frame, not work through a backlog and fall
further and further behind real time.

`StreamTracker` manages one video source end-to-end. `MultiStreamManager`
runs several `StreamTracker`s concurrently (e.g. one per camera) and
multiplexes their results, which is what `src/api/main.py` uses to serve
more than one RTSP feed from a single process.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np

from src.core.base_model import BaseDetector, BaseOCR, BaseTracker
from src.core.exceptions import StreamConnectionError
from src.core.types import FrameResult, TrackedVehicle
from src.tracking.speed_estimator import BaseSpeedEstimator
from src.utils.video_utils import FPSCounter

logger = logging.getLogger(__name__)


@dataclass
class StreamTrackerConfig:
    source: str                              # RTSP URL, file path, or webcam index (as str)
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    allowed_classes: Optional[List[int]] = None
    reconnect_initial_backoff_s: float = 1.0
    reconnect_max_backoff_s: float = 30.0
    max_consecutive_read_failures: int = 10  # before treating the source as truly down
    frame_queue_get_timeout_s: float = 2.0
    results_queue_maxsize: int = 64
    counting_line_y: Optional[int] = None    # pixel row; None disables line counting


class _FrameGrabber(threading.Thread):
    """Background thread owning the `cv2.VideoCapture` for one source.
    Continuously reads frames and keeps only the most recent one in a
    size-1 queue, transparently reconnecting on read failures.
    """

    def __init__(self, config: StreamTrackerConfig) -> None:
        super().__init__(daemon=True, name=f"FrameGrabber[{config.source}]")
        self.config = config
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._cap = None
        self._consecutive_failures = 0

    def run(self) -> None:
        import cv2

        backoff = self.config.reconnect_initial_backoff_s
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.config.source)
                if not self._cap.isOpened():
                    logger.warning(
                        "Failed to open stream '%s'; retrying in %.1fs.", self.config.source, backoff
                    )
                    if self._stop_event.wait(backoff):
                        break
                    backoff = min(backoff * 2, self.config.reconnect_max_backoff_s)
                    continue
                logger.info("Opened stream '%s'.", self.config.source)
                backoff = self.config.reconnect_initial_backoff_s
                self._consecutive_failures = 0

            ok, frame = self._cap.read()
            if not ok:
                self._consecutive_failures += 1
                logger.debug(
                    "Frame read failed on '%s' (%d/%d consecutive).",
                    self.config.source,
                    self._consecutive_failures,
                    self.config.max_consecutive_read_failures,
                )
                if self._consecutive_failures >= self.config.max_consecutive_read_failures:
                    logger.warning("Too many read failures on '%s'; reconnecting.", self.config.source)
                    self._cap.release()
                    self._cap = None
                    if self._stop_event.wait(backoff):
                        break
                    backoff = min(backoff * 2, self.config.reconnect_max_backoff_s)
                continue

            self._consecutive_failures = 0
            self._push_latest(frame)

        if self._cap is not None:
            self._cap.release()
        logger.info("FrameGrabber for '%s' stopped.", self.config.source)

    def _push_latest(self, frame: np.ndarray) -> None:
        """Overwrite-on-full: drop the stale frame rather than block, so
        the inference side always sees the newest available frame."""
        try:
            self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:  # pragma: no cover - benign race with the get above
            pass

    def stop(self) -> None:
        self._stop_event.set()


class _LineCounter:
    """Counts a track exactly once, the first time its center crosses a
    fixed horizontal pixel row -- a minimal traffic-density counter.
    Direction-agnostic (counts both directions); swap in a signed
    crossing check if you need per-direction counts."""

    def __init__(self, line_y: int) -> None:
        self.line_y = line_y
        self._last_center_y: Dict[int, float] = {}
        self._counted_track_ids: set = set()
        self.counts_by_class: Dict[str, int] = {}
        self.total_count: int = 0

    def update(self, vehicles: List[TrackedVehicle]) -> None:
        for vehicle in vehicles:
            track = vehicle.track
            _, cy = track.bbox.center
            last_cy = self._last_center_y.get(track.track_id)
            self._last_center_y[track.track_id] = cy

            if (
                last_cy is not None
                and track.track_id not in self._counted_track_ids
                and (last_cy - self.line_y) * (cy - self.line_y) <= 0  # sign change => crossed
            ):
                self._counted_track_ids.add(track.track_id)
                self.total_count += 1
                self.counts_by_class[track.class_name] = self.counts_by_class.get(track.class_name, 0) + 1


class StreamTracker:
    """Owns the full capture -> detect -> track -> enrich pipeline for a
    single video source, running inference on its own background thread.

    Usage:
        tracker = StreamTracker(config, detector, tracker=bytetrack, fps_hint=25)
        tracker.start()
        for result in tracker.results():
            ...  # consume FrameResult objects as they're produced
        tracker.stop()
    """

    def __init__(
        self,
        config: StreamTrackerConfig,
        detector: BaseDetector,
        tracker: BaseTracker,
        fps_hint: float = 25.0,
        speed_estimator: Optional[BaseSpeedEstimator] = None,
        ocr: Optional[BaseOCR] = None,
        on_result: Optional[Callable[[FrameResult], None]] = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.fps_hint = fps_hint
        self.speed_estimator = speed_estimator
        self.ocr = ocr
        self._on_result = on_result

        self._grabber = _FrameGrabber(config)
        self._inference_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._results_queue: "queue.Queue[FrameResult]" = queue.Queue(maxsize=config.results_queue_maxsize)
        self._fps_counter = FPSCounter()
        self._line_counter = _LineCounter(config.counting_line_y) if config.counting_line_y is not None else None
        self._frame_index = 0
        self._last_frame_lock = threading.Lock()
        self._last_raw_frame: Optional[np.ndarray] = None

    def start(self) -> None:
        self.detector.warmup()
        self._grabber.start()
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name=f"Inference[{self.config.source}]"
        )
        self._inference_thread.start()
        logger.info("StreamTracker started for '%s'.", self.config.source)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        self._grabber.stop()
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=timeout_s)
        logger.info("StreamTracker stopped for '%s'.", self.config.source)

    def results(self, timeout_s: float = 1.0) -> Iterator[FrameResult]:
        """Blocking generator yielding `FrameResult`s as they're produced.
        Safe to consume from a different thread than the one that called
        `start()`. Stops yielding once `stop()` has been called and the
        queue is drained."""
        while not self._stop_event.is_set() or not self._results_queue.empty():
            try:
                yield self._results_queue.get(timeout=timeout_s)
            except queue.Empty:
                continue

    @property
    def traffic_counts(self) -> Dict[str, int]:
        if self._line_counter is None:
            return {}
        return dict(self._line_counter.counts_by_class)

    @property
    def current_fps(self) -> float:
        return self._fps_counter.fps

    def get_last_frame(self) -> Optional[np.ndarray]:
        """Thread-safe accessor for the most recent raw (unannotated)
        frame, used by the API's MJPEG endpoint to render live overlays."""
        with self._last_frame_lock:
            return None if self._last_raw_frame is None else self._last_raw_frame.copy()

    def _inference_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._grabber.frame_queue.get(timeout=self.config.frame_queue_get_timeout_s)
            except queue.Empty:
                continue

            try:
                result = self._process_frame(frame)
            except Exception:  # noqa: BLE001 - keep the pipeline alive on a bad frame
                logger.exception("Inference failed on a frame from '%s'; skipping it.", self.config.source)
                continue

            with self._last_frame_lock:
                self._last_raw_frame = frame

            self._fps_counter.tick()
            self._publish(result)

    def _process_frame(self, frame: np.ndarray) -> FrameResult:
        detections = self.detector.predict(
            frame,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            classes=self.config.allowed_classes,
        )
        tracks = self.tracker.update(detections, frame=frame)

        vehicles: List[TrackedVehicle] = []
        for track in tracks:
            if self.speed_estimator is not None and track.velocity_px_per_frame is not None:
                speed_kmh = self.speed_estimator.estimate_kmh(track.velocity_px_per_frame, self.fps_hint)
                track = _with_speed(track, speed_kmh)

            plate = None
            if self.ocr is not None:
                plate = _try_read_plate(self.ocr, frame, track)

            vehicles.append(TrackedVehicle(track=track, plate=plate))

        if self._line_counter is not None:
            self._line_counter.update(vehicles)

        self._frame_index += 1
        h, w = frame.shape[:2]
        return FrameResult(
            frame_index=self._frame_index,
            timestamp_s=time.time(),
            frame_width=w,
            frame_height=h,
            vehicles=tuple(vehicles),
        )

    def _publish(self, result: FrameResult) -> None:
        if self._on_result is not None:
            self._on_result(result)

        # Overwrite-on-full for results too: a live dashboard/API consumer
        # cares about the latest traffic state, not a growing backlog.
        try:
            self._results_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._results_queue.put_nowait(result)
        except queue.Full:  # pragma: no cover - benign race with the get above
            pass


def _with_speed(track, speed_kmh: float):
    """`Track` is a frozen dataclass -- return a copy with `speed_kmh` set
    rather than mutating in place."""
    from dataclasses import replace

    return replace(track, speed_kmh=speed_kmh)


def _try_read_plate(ocr: BaseOCR, frame: np.ndarray, track) -> Optional[object]:
    x1, y1, x2, y2 = (int(v) for v in track.bbox.to_xyxy())
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    plate_regions = ocr.locate_plates(crop)
    if not plate_regions:
        return None

    best = max(plate_regions, key=lambda d: d.confidence)
    px1, py1, px2, py2 = (int(v) for v in best.bbox.to_xyxy())
    plate_crop = crop[max(py1, 0):py2, max(px1, 0):px2]
    if plate_crop.size == 0:
        return None
    return ocr.read_plate(plate_crop)


class MultiStreamManager:
    """Runs several `StreamTracker`s concurrently (e.g. one per camera)
    and exposes a single multiplexed results feed -- what
    `src/api/main.py` uses to back a multi-camera dashboard from one
    process without the API layer knowing anything about threads."""

    def __init__(self) -> None:
        self._trackers: Dict[str, StreamTracker] = {}

    def add_stream(self, stream_id: str, tracker: StreamTracker) -> None:
        if stream_id in self._trackers:
            raise ValueError(f"Stream id '{stream_id}' is already registered.")
        self._trackers[stream_id] = tracker
        tracker.start()

    def remove_stream(self, stream_id: str) -> None:
        tracker = self._trackers.pop(stream_id, None)
        if tracker is not None:
            tracker.stop()

    def get_tracker(self, stream_id: str) -> StreamTracker:
        if stream_id not in self._trackers:
            raise StreamConnectionError(f"No active stream registered under id '{stream_id}'.")
        return self._trackers[stream_id]

    def stream_ids(self) -> List[str]:
        return list(self._trackers)

    def stop_all(self) -> None:
        for stream_id in list(self._trackers):
            self.remove_stream(stream_id)
