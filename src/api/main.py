"""
src/api/main.py

FastAPI microservice replacing the legacy Flask scripts. Endpoints:

    GET  /health                          liveness + model/stream status
    POST /detect                          single-image detection (async, threadpool-offloaded)
    POST /streams                         start tracking an RTSP/video source
    DELETE /streams/{stream_id}           stop a stream
    GET  /streams                         list active streams + live stats
    GET  /streams/{stream_id}/latest      latest FrameResult as JSON
    GET  /streams/{stream_id}/mjpeg       live annotated MJPEG stream

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1

NOTE on `--workers`: stream state (`MultiStreamManager`) lives in process
memory, so this service is designed to run as a single worker process per
deployment unit (scale horizontally behind a load balancer that pins each
stream_id to one instance, e.g. via consistent hashing) rather than with
multiple Uvicorn workers sharing one port.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_detector, get_settings, get_stream_manager
from src.api.schemas import (
    DetectionSchema,
    DetectResponse,
    FrameResultSchema,
    HealthResponse,
    StreamStartRequest,
    StreamStartResponse,
    StreamStatus,
    TrackedVehicleSchema,
)
from src.core.base_model import BaseDetector
from src.core.exceptions import StreamConnectionError, VehicleCVError
from src.core.registry import get_component
from src.pipelines.stream_tracker import MultiStreamManager, StreamTracker, StreamTrackerConfig
from src.utils.logging_utils import configure_logging
from src.utils.video_utils import draw_tracked_vehicles

# Registering these modules attaches their `@register("tracker", ...)`
# decorators to the registry before any `/streams` request needs to look
# a backend up by name.
import src.tracking.bytetrack_tracker  # noqa: F401
import src.tracking.botsort_tracker  # noqa: F401

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Force the detector to load (and warm up) at startup rather than on
    # the first request, so the first real caller doesn't pay that cost.
    get_detector()
    logger.info("API startup complete.")
    yield
    get_stream_manager().stop_all()
    logger.info("API shutdown complete: all streams stopped.")


app = FastAPI(
    title="Vehicle CV Platform API",
    description="Detection, multi-object tracking, ALPR, and traffic analytics for vehicle CCTV/RTSP feeds.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(VehicleCVError)
async def vehicle_cv_error_handler(request, exc: VehicleCVError):
    from fastapi.responses import JSONResponse

    status_code = 404 if isinstance(exc, StreamConnectionError) else 400
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health(
    detector: BaseDetector = Depends(get_detector),
    stream_manager: MultiStreamManager = Depends(get_stream_manager),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        detector_loaded=detector is not None,
        active_streams=len(stream_manager.stream_ids()),
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    detector: BaseDetector = Depends(get_detector),
) -> DetectResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image upload, got '{file.content_type}'.")

    image_bytes = await file.read()
    frame = _decode_image(image_bytes)

    start = time.perf_counter()
    # Detector inference is synchronous/CPU-or-GPU-bound; running it
    # directly in this async def would block the event loop for every
    # other concurrent request, so it's offloaded to FastAPI's threadpool.
    detections = await asyncio.to_thread(
        detector.predict, frame, conf_threshold, iou_threshold, None
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    return DetectResponse(
        detections=[
            DetectionSchema(
                bbox={"x1": d.bbox.x1, "y1": d.bbox.y1, "x2": d.bbox.x2, "y2": d.bbox.y2},
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
            )
            for d in detections
        ],
        inference_time_ms=elapsed_ms,
    )


@app.post("/streams", response_model=StreamStartResponse, status_code=201)
async def start_stream(
    request: StreamStartRequest,
    detector: BaseDetector = Depends(get_detector),
    stream_manager: MultiStreamManager = Depends(get_stream_manager),
) -> StreamStartResponse:
    if request.stream_id in stream_manager.stream_ids():
        raise HTTPException(status_code=409, detail=f"Stream '{request.stream_id}' is already running.")

    tracker_cls = get_component("tracker", request.tracker_backend)
    tracker = tracker_cls()

    config = StreamTrackerConfig(
        source=request.source,
        conf_threshold=request.conf_threshold,
        iou_threshold=request.iou_threshold,
        counting_line_y=request.counting_line_y,
    )
    stream_tracker = StreamTracker(config=config, detector=detector, tracker=tracker)

    # start() opens the video source on a background thread; do this off
    # the event loop so a slow/unreachable RTSP endpoint can't stall the
    # request (the grabber thread itself retries indefinitely afterwards).
    await asyncio.to_thread(stream_manager.add_stream, request.stream_id, stream_tracker)

    return StreamStartResponse(stream_id=request.stream_id)


@app.delete("/streams/{stream_id}", status_code=204)
async def stop_stream(
    stream_id: str, stream_manager: MultiStreamManager = Depends(get_stream_manager)
) -> None:
    if stream_id not in stream_manager.stream_ids():
        raise HTTPException(status_code=404, detail=f"No active stream '{stream_id}'.")
    await asyncio.to_thread(stream_manager.remove_stream, stream_id)


@app.get("/streams", response_model=list[StreamStatus])
def list_streams(stream_manager: MultiStreamManager = Depends(get_stream_manager)) -> list[StreamStatus]:
    statuses = []
    for stream_id in stream_manager.stream_ids():
        tracker = stream_manager.get_tracker(stream_id)
        statuses.append(
            StreamStatus(
                stream_id=stream_id,
                source=tracker.config.source,
                current_fps=tracker.current_fps,
                traffic_counts=tracker.traffic_counts,
            )
        )
    return statuses


@app.get("/streams/{stream_id}/latest", response_model=FrameResultSchema)
async def get_latest_result(
    stream_id: str, stream_manager: MultiStreamManager = Depends(get_stream_manager)
) -> FrameResultSchema:
    tracker = stream_manager.get_tracker(stream_id)  # raises StreamConnectionError -> 404 via handler

    try:
        result = await asyncio.wait_for(_get_latest_async(tracker), timeout=5.0)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Timed out waiting for the next frame result.") from exc

    return FrameResultSchema(
        frame_index=result.frame_index,
        timestamp_s=result.timestamp_s,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        vehicle_count=result.vehicle_count,
        vehicles=[
            TrackedVehicleSchema(
                track={
                    "track_id": v.track.track_id,
                    "bbox": {"x1": v.track.bbox.x1, "y1": v.track.bbox.y1, "x2": v.track.bbox.x2, "y2": v.track.bbox.y2},
                    "confidence": v.track.confidence,
                    "class_id": v.track.class_id,
                    "class_name": v.track.class_name,
                    "age_frames": v.track.age_frames,
                    "speed_kmh": v.track.speed_kmh,
                },
                plate=None
                if v.plate is None
                else {
                    "bbox": {"x1": v.plate.bbox.x1, "y1": v.plate.bbox.y1, "x2": v.plate.bbox.x2, "y2": v.plate.bbox.y2},
                    "text": v.plate.text,
                    "detection_confidence": v.plate.detection_confidence,
                    "ocr_confidence": v.plate.ocr_confidence,
                },
                attributes=None,
            )
            for v in result.vehicles
        ],
    )


@app.get("/streams/{stream_id}/mjpeg")
async def stream_mjpeg(stream_id: str, stream_manager: MultiStreamManager = Depends(get_stream_manager)):
    tracker = stream_manager.get_tracker(stream_id)
    return StreamingResponse(
        _mjpeg_generator(tracker), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
async def _get_latest_async(tracker: StreamTracker):
    """Bridge `StreamTracker`'s blocking, thread-based `results()`
    generator into a single awaitable that returns just the next result,
    without blocking the event loop."""

    def _blocking_get():
        for result in tracker.results(timeout_s=5.0):
            return result
        raise queue.Empty

    return await asyncio.to_thread(_blocking_get)


async def _mjpeg_generator(tracker: StreamTracker):
    """Async generator yielding multipart JPEG frames: the tracker's last
    raw frame (see `StreamTracker.get_last_frame`) with live detection/
    tracking overlays drawn on top via `draw_tracked_vehicles`."""
    while True:
        try:
            result = await _get_latest_async(tracker)
        except queue.Empty:
            await asyncio.sleep(0.1)
            continue

        raw_frame = tracker.get_last_frame()
        if raw_frame is None:
            await asyncio.sleep(0.1)
            continue

        annotated = draw_tracked_vehicles(raw_frame, result.vehicles)
        ok, jpeg = cv2.imencode(".jpg", annotated)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


def _decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded file as an image.")
    return frame
