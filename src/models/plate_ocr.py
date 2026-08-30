"""
src/models/plate_ocr.py

License plate localization + recognition (ALPR), split into two narrow,
independently swappable strategies -- composed via dependency injection
into `AlprPipeline`, which is what actually implements `BaseOCR`
(`src/core/base_model.py`):

    PlateLocator  (locate candidate plate regions within a vehicle crop)
    TextReader    (recognize text within an already-localized plate crop)

This is one level more decomposed than `BaseOCR` strictly requires
(`BaseOCR` only demands `locate_plates`/`read_plate`) -- deliberately so,
because plate localization and text recognition are usually swapped
independently in practice (e.g. keep the same OCR reader while trying a
different detector, or vice versa), and Interface Segregation says each
of those decisions shouldn't force touching the other.

Two concrete implementations are provided:

    HaarCascadePlateLocator  -- OpenCV's bundled
        `haarcascade_russian_plate_number.xml`. Zero extra dependencies
        beyond `opencv`, works out of the box, but is the weakest link
        in this module: Haar cascades are 2000s-era technology, tuned on
        a specific plate style, and will under-perform a trained
        YOLO-based plate detector on non-Russian/EU plate formats or
        difficult angles/lighting. It's the honest, verifiable-today
        default -- swap in a `YoloPlateLocator` (same shape as
        `src/models/yolo_adapter.py`, pointed at a plate-detection
        checkpoint) for production accuracy without changing
        `AlprPipeline` or `BaseOCR` callers at all.

    EasyOcrTextReader  -- wraps the `easyocr` package. NOT
        runtime-verified in this environment (`easyocr` transitively
        requires `torch`, unavailable here) -- reviewed against its
        documented API, not executed. `HaarCascadePlateLocator` above,
        by contrast, IS exercised for real in `tests/test_alpr.py` since
        it only needs `opencv`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from src.core.base_model import BaseOCR
from src.core.exceptions import InferenceError, ModelLoadError
from src.core.registry import register
from src.core.types import BoundingBox, Detection, PlateResult

logger = logging.getLogger(__name__)


class PlateLocator(ABC):
    """Finds candidate plate regions within an already-cropped vehicle image."""

    @abstractmethod
    def locate(self, vehicle_crop: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class TextReader(ABC):
    """Recognizes text within an already-localized, tightly-cropped plate image."""

    @abstractmethod
    def read(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """Return (recognized_text, confidence)."""
        raise NotImplementedError


@register("plate_locator", "haar_cascade")
class HaarCascadePlateLocator(PlateLocator):
    """OpenCV Haar cascade plate detector -- see module docstring for
    accuracy caveats. `cascade_name` must be one of the `.xml` files
    bundled under `cv2.data.haarcascades`.
    """

    def __init__(
        self,
        cascade_name: str = "haarcascade_russian_plate_number.xml",
        scale_factor: float = 1.05,
        min_neighbors: int = 4,
        min_size: Tuple[int, int] = (25, 8),
    ) -> None:
        import os

        import cv2

        cascade_path = os.path.join(cv2.data.haarcascades, cascade_name)
        if not os.path.exists(cascade_path):
            raise ModelLoadError(f"Haar cascade file not found: '{cascade_path}'.")

        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise ModelLoadError(f"Failed to load Haar cascade from '{cascade_path}' (file is invalid or corrupt).")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def locate(self, vehicle_crop: np.ndarray) -> List[Detection]:
        import cv2

        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY) if vehicle_crop.ndim == 3 else vehicle_crop

        try:
            boxes = self._cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
            )
        except cv2.error as exc:
            raise InferenceError(f"Haar cascade detection failed: {exc}") from exc

        detections: List[Detection] = []
        for (x, y, w, h) in boxes:
            # Haar cascades don't produce a confidence score the way a
            # learned detector does; `1.0` is a placeholder constant so
            # `AlprPipeline.locate_plates` can still rank multiple
            # candidates consistently (all tied) -- a YOLO-based
            # `PlateLocator` swapped in later would report real scores here.
            detections.append(
                Detection(
                    bbox=BoundingBox(float(x), float(y), float(x + w), float(y + h)),
                    confidence=1.0,
                    class_id=0,
                    class_name="plate",
                )
            )
        return detections


@register("text_reader", "easyocr")
class EasyOcrTextReader(TextReader):
    """Wraps `easyocr.Reader` for plate text recognition.

    NOT runtime-verified in this environment -- `easyocr` transitively
    depends on `torch`, which isn't installed here. Reviewed against
    EasyOCR's documented `Reader.readtext` API (returns a list of
    `(bbox, text, confidence)` tuples); the model is lazily constructed
    on first use so importing this module doesn't require `easyocr` to
    be installed unless `EasyOcrTextReader` is actually instantiated.
    """

    def __init__(self, languages: List[str] = None, use_gpu: bool = False) -> None:
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self._reader = None  # lazy: avoid importing/loading easyocr until first read()

    def _ensure_loaded(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise ModelLoadError(
                    "easyocr is required for EasyOcrTextReader. Install via `pip install easyocr`."
                ) from exc
            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        return self._reader

    def read(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        reader = self._ensure_loaded()
        try:
            results = reader.readtext(plate_crop)
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"EasyOCR read failed: {exc}") from exc

        if not results:
            return "", 0.0

        # A plate may be segmented into multiple text regions (e.g. a
        # region code separate from the main plate number); concatenate
        # left-to-right by x-coordinate and average confidence, rather
        # than keeping only the single highest-confidence fragment.
        results_sorted = sorted(results, key=lambda r: r[0][0][0])  # sort by top-left x
        text = "".join(r[1] for r in results_sorted).replace(" ", "").upper()
        avg_confidence = float(np.mean([r[2] for r in results_sorted]))
        return text, avg_confidence


class AlprPipeline(BaseOCR):
    """Composes an injected `PlateLocator` + `TextReader` into the
    `BaseOCR` contract consumed by `src/pipelines/stream_tracker.py`.
    """

    def __init__(self, plate_locator: PlateLocator, text_reader: TextReader, min_plate_confidence: float = 0.0) -> None:
        self._plate_locator = plate_locator
        self._text_reader = text_reader
        self.min_plate_confidence = min_plate_confidence

    def locate_plates(self, vehicle_crop: np.ndarray) -> List[Detection]:
        detections = self._plate_locator.locate(vehicle_crop)
        return [d for d in detections if d.confidence >= self.min_plate_confidence]

    def read_plate(self, plate_crop: np.ndarray) -> PlateResult:
        text, ocr_confidence = self._text_reader.read(plate_crop)
        h, w = plate_crop.shape[:2] if plate_crop is not None and plate_crop.size > 0 else (0, 0)
        return PlateResult(
            bbox=BoundingBox(0.0, 0.0, float(w), float(h)),
            text=text,
            detection_confidence=1.0,  # the caller already filtered by locator confidence upstream
            ocr_confidence=ocr_confidence,
        )
