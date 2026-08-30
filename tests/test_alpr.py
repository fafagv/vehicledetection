"""
tests/test_alpr.py

`HaarCascadePlateLocator` needs only `opencv` (verified: runs in this
sandbox with no torch/ultralytics). `EasyOcrTextReader` is NOT exercised
here -- `easyocr` transitively requires `torch`; see its docstring in
`src/models/plate_ocr.py`. `AlprPipeline`'s DI composition is tested
against fakes, independent of either concrete strategy.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.core.types import BoundingBox, Detection
from src.models.plate_ocr import AlprPipeline, HaarCascadePlateLocator, PlateLocator, TextReader


class TestHaarCascadePlateLocator:
    def test_loads_and_runs_without_crashing(self):
        locator = HaarCascadePlateLocator()
        random_image = np.random.randint(0, 255, size=(200, 300, 3), dtype=np.uint8)
        detections = locator.locate(random_image)
        assert isinstance(detections, list)

    def test_handles_empty_crop(self):
        locator = HaarCascadePlateLocator()
        assert locator.locate(np.zeros((0, 0, 3), dtype=np.uint8)) == []

    def test_handles_none_crop(self):
        locator = HaarCascadePlateLocator()
        assert locator.locate(None) == []


class _FakePlateLocator(PlateLocator):
    def locate(self, vehicle_crop):
        return [
            Detection(bbox=BoundingBox(0, 0, 50, 20), confidence=0.9, class_id=0, class_name="plate"),
            Detection(bbox=BoundingBox(60, 60, 90, 80), confidence=0.3, class_id=0, class_name="plate"),
        ]


class _FakeTextReader(TextReader):
    def read(self, plate_crop):
        return "ABC123", 0.95


class TestAlprPipeline:
    def test_locate_plates_filters_by_min_confidence(self):
        pipeline = AlprPipeline(plate_locator=_FakePlateLocator(), text_reader=_FakeTextReader(), min_plate_confidence=0.5)
        plates = pipeline.locate_plates(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(plates) == 1
        assert plates[0].confidence == 0.9

    def test_read_plate_delegates_to_text_reader(self):
        pipeline = AlprPipeline(plate_locator=_FakePlateLocator(), text_reader=_FakeTextReader())
        result = pipeline.read_plate(np.zeros((20, 50, 3), dtype=np.uint8))
        assert result.text == "ABC123"
        assert result.ocr_confidence == 0.95
