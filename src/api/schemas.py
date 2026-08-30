"""
src/api/schemas.py

Pydantic mirrors of `src.core.types` plus request/response envelopes for
the FastAPI service. Kept as a straight field-for-field translation (no
business logic) so `src/core/types.py` stays framework-agnostic while the
API still gets full request validation + OpenAPI schema generation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionSchema(BaseModel):
    bbox: BoundingBoxSchema
    confidence: float
    class_id: int
    class_name: str


class TrackSchema(BaseModel):
    track_id: int
    bbox: BoundingBoxSchema
    confidence: float
    class_id: int
    class_name: str
    age_frames: int
    speed_kmh: Optional[float] = None


class PlateResultSchema(BaseModel):
    bbox: BoundingBoxSchema
    text: str
    detection_confidence: float
    ocr_confidence: float


class VehicleAttributesSchema(BaseModel):
    vehicle_type: Optional[str] = None
    vehicle_type_confidence: Optional[float] = None
    color: Optional[str] = None
    color_confidence: Optional[float] = None
    brand: Optional[str] = None
    brand_confidence: Optional[float] = None


class TrackedVehicleSchema(BaseModel):
    track: TrackSchema
    plate: Optional[PlateResultSchema] = None
    attributes: Optional[VehicleAttributesSchema] = None


class FrameResultSchema(BaseModel):
    frame_index: int
    timestamp_s: float
    frame_width: int
    frame_height: int
    vehicles: List[TrackedVehicleSchema]
    vehicle_count: int


# --------------------------------------------------------------------- #
# Request / response envelopes
# --------------------------------------------------------------------- #
class DetectRequestParams(BaseModel):
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)


class DetectResponse(BaseModel):
    detections: List[DetectionSchema]
    inference_time_ms: float


class StreamStartRequest(BaseModel):
    stream_id: str = Field(..., description="Caller-chosen unique id, e.g. 'camera-01'.")
    source: str = Field(..., description="RTSP URL, video file path, or webcam index as a string.")
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    tracker_backend: str = Field("bytetrack", description="Registered tracker name, e.g. 'bytetrack'.")
    counting_line_y: Optional[int] = None


class StreamStartResponse(BaseModel):
    stream_id: str
    status: str = "started"


class StreamStatus(BaseModel):
    stream_id: str
    source: str
    current_fps: float
    traffic_counts: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    active_streams: int
