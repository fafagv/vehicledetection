"""
src/core/base_model.py

The stable, framework-agnostic contracts every concrete backend in this
platform implements. Nothing in this file imports torch, Lightning,
Ultralytics, or OpenCV directly -- it only depends on `src.core.types`
(plain dataclasses) and the standard library, so it can be imported
anywhere (including inside `src/api`) without pulling in heavy ML
dependencies transitively.

Design rationale (SOLID):
    Single Responsibility  -- each ABC covers exactly one capability
        (detect, track, read plates, classify attributes). A concrete
        class may implement more than one if that's how the underlying
        model works (e.g. a single YOLO model doing both detection and
        pose), but the *interfaces* stay separate so callers only depend
        on the capability they need.
    Open/Closed             -- `src/pipelines/stream_tracker.py` and
        `src/api/dependencies.py` are written entirely against these
        interfaces plus `src.core.registry`. Adding a new detector
        backend (say, RT-DETR) never requires touching either of those
        files -- only a new `BaseDetector` subclass + a registry entry.
    Liskov Substitution     -- any `BaseDetector` can replace any other
        inside `stream_tracker.py`'s pipeline; the same guarantee holds
        for `BaseTracker`, `BaseOCR`, and `BaseClassifier`.
    Interface Segregation   -- `BaseTracker.update` takes/returns plain
        `Detection`/`Track` lists rather than a fat "context" object, so
        a minimal IOU tracker doesn't need to know about OCR or
        classification at all.
    Dependency Inversion    -- every pipeline/API module depends on these
        ABCs (and `src.core.registry.get_component`), never on a concrete
        `YoloAdapter` or `ByteTrackTracker` class name directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

from src.core.types import Detection, PlateResult, Track, VehicleAttributes


class BaseDetector(ABC):
    """Contract for anything that turns a single image/frame into a list
    of `Detection`s -- YOLOv8/v11 today, RT-DETR or a future architecture
    tomorrow, without changing any caller.
    """

    @abstractmethod
    def predict(
        self,
        frame,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[Sequence[int]] = None,
    ) -> List[Detection]:
        """Run inference on a single frame (HWC BGR or RGB numpy array,
        implementation-defined -- document your expected format on the
        concrete subclass) and return detections above `conf_threshold`.

        Args:
            frame: a single image as a numpy array.
            conf_threshold: minimum confidence to keep a detection.
            iou_threshold: NMS IoU threshold.
            classes: optional allow-list of class ids to keep (e.g. only
                vehicle classes from a COCO-pretrained checkpoint).
        """
        raise NotImplementedError

    @abstractmethod
    def warmup(self, image_size: int = 640) -> None:
        """Run one dummy forward pass so the first real request doesn't
        pay for lazy CUDA/TensorRT context initialization. Implementations
        should be safe to call multiple times (idempotent)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Ordered class names this detector was trained/configured with."""
        raise NotImplementedError


class BaseTracker(ABC):
    """Contract for multi-object trackers (ByteTrack, BoT-SORT, ...) that
    turn a per-frame list of `Detection`s into identity-persistent
    `Track`s across frames.

    Implementations are inherently stateful (they remember prior frames'
    tracks internally) -- `reset()` exists so a single instance can be
    reused across independent video sources (e.g. per-camera trackers in
    a pool) without reconstructing the backend each time.
    """

    @abstractmethod
    def update(self, detections: List[Detection], frame=None) -> List[Track]:
        """Associate `detections` (from the current frame) with existing
        track identities and return the updated `Track` list.

        Args:
            detections: this frame's raw detector output.
            frame: optional raw frame, needed by appearance-based trackers
                (e.g. BoT-SORT's Re-ID branch); IoU-only trackers (e.g.
                plain ByteTrack) may ignore it.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal track state (e.g. when switching video sources)."""
        raise NotImplementedError


class BaseOCR(ABC):
    """Contract for license-plate localization + text recognition (ALPR).

    Split into two steps (`locate_plates` / `read_plate`) rather than one
    fused `read` method so a caller can batch OCR only on the crops that
    survive some downstream filter (e.g. skip OCR on plates smaller than
    N pixels) without the interface forcing that policy on every backend.
    """

    @abstractmethod
    def locate_plates(self, vehicle_crop) -> List[Detection]:
        """Detect candidate plate regions within a single vehicle crop.
        Returned `Detection.bbox` coordinates are relative to
        `vehicle_crop`, not the original frame."""
        raise NotImplementedError

    @abstractmethod
    def read_plate(self, plate_crop) -> PlateResult:
        """Run OCR on an already-localized, tightly-cropped plate image."""
        raise NotImplementedError


class BaseClassifier(ABC):
    """Contract for vehicle attribute classification (type/color/brand)."""

    @abstractmethod
    def classify(self, vehicle_crop) -> VehicleAttributes:
        """Predict attributes for a single cropped vehicle image."""
        raise NotImplementedError


@runtime_checkable
class Trainable(Protocol):
    """Structural (duck-typed) contract for anything `TrainUseCase`
    (`src/pipelines/use_cases.py`) can train -- deliberately a
    `Protocol`, not an ABC every detector must inherit, since training
    is a capability only some backends have (`YoloAdapter` does via its
    `native_train` method; a hypothetical ONNX-only inference-time
    detector legitimately wouldn't implement this at all).
    """

    def native_train(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class Exportable(Protocol):
    """Structural contract for anything `ExportUseCase`
    (`src/pipelines/use_cases.py`) can export to an inference-optimized
    format (ONNX / TensorRT / OpenVINO). `YoloAdapter.export` implements
    this by delegating to Ultralytics' own native multi-format exporter
    rather than reimplementing ONNX/TensorRT conversion by hand -- see
    that method's docstring.
    """

    def export(self, format: str, output_path: str, **kwargs: Any) -> str: ...
