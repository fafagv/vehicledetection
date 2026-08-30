"""
src/core/types.py

Framework-agnostic typed dataclasses describing the domain objects that
flow between every layer of this platform: a `BaseDetector` produces
`Detection`s, a `BaseTracker` turns a sequence of `Detection`s into
`Track`s, and `stream_tracker.py` enriches each `Track` with an optional
`PlateResult` (ALPR) and `VehicleAttributes` (classification) to build a
`TrackedVehicle`.

Kept dependency-free (no torch/numpy required at the type level beyond
plain floats/tuples) so `src/api/schemas.py` can losslessly mirror these
as Pydantic models without any conversion logic beyond field renaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in absolute pixel coordinates, (x1, y1) top-left to
    (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"Invalid box: ({self.x1}, {self.y1}, {self.x2}, {self.y2}) "
                f"-- x2/y2 must be >= x1/y1."
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.width / 2.0, self.y1 + self.height / 2.0)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Detection:
    """A single, unassociated per-frame detector output (no identity yet)."""

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class Track:
    """A `Detection` that has been associated with a persistent identity by
    a `BaseTracker`, plus optional motion-derived fields."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    age_frames: int = 0
    velocity_px_per_frame: Optional[Tuple[float, float]] = None
    speed_kmh: Optional[float] = None

    @classmethod
    def from_detection(cls, detection: Detection, track_id: int, age_frames: int = 0) -> "Track":
        return cls(
            track_id=track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
            age_frames=age_frames,
        )


@dataclass(frozen=True)
class PlateResult:
    """Output of the ALPR module: a localized plate region plus recognized text."""

    bbox: BoundingBox
    text: str
    detection_confidence: float
    ocr_confidence: float


@dataclass(frozen=True)
class VehicleAttributes:
    """Output of the vehicle classification module. Any field may be `None`
    if that sub-task wasn't run or didn't clear its confidence threshold."""

    vehicle_type: Optional[str] = None
    vehicle_type_confidence: Optional[float] = None
    color: Optional[str] = None
    color_confidence: Optional[float] = None
    brand: Optional[str] = None
    brand_confidence: Optional[float] = None


@dataclass(frozen=True)
class TrackedVehicle:
    """A fully-enriched tracked object for a single frame: identity +
    optional plate read + optional attribute classification."""

    track: Track
    plate: Optional[PlateResult] = None
    attributes: Optional[VehicleAttributes] = None


@dataclass(frozen=True)
class FrameResult:
    """Everything produced for one processed video frame."""

    frame_index: int
    timestamp_s: float
    frame_width: int
    frame_height: int
    vehicles: Tuple[TrackedVehicle, ...] = field(default_factory=tuple)

    @property
    def vehicle_count(self) -> int:
        return len(self.vehicles)
