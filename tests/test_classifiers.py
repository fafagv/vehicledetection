"""
tests/test_classifiers.py

`HistogramColorClassifier` needs only `opencv`/`numpy` -- verified here
against known ground-truth colors (pure red/green/blue/yellow/white/
black/gray), not just structurally. `TimmBackboneClassifier` is NOT
exercised (`torch`/`timm` unavailable in this sandbox); see its
docstring in `src/models/vehicle_classifier.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.core.base_model import BaseClassifier
from src.core.types import VehicleAttributes
from src.models.vehicle_classifier import CompositeVehicleClassifier, HistogramColorClassifier


def _solid_bgr(b: int, g: int, r: int, size: int = 100) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = (b, g, r)
    return image


class TestHistogramColorClassifier:
    @pytest.mark.parametrize(
        "bgr,expected",
        [
            ((0, 0, 255), "red"),
            ((0, 255, 0), "green"),
            ((255, 0, 0), "blue"),
            ((0, 255, 255), "yellow"),
        ],
    )
    def test_classifies_known_hues(self, bgr, expected):
        classifier = HistogramColorClassifier()
        attrs = classifier.classify(_solid_bgr(*bgr))
        assert attrs.color == expected

    def test_classifies_white(self):
        classifier = HistogramColorClassifier()
        attrs = classifier.classify(np.full((100, 100, 3), 255, dtype=np.uint8))
        assert attrs.color == "white"

    def test_classifies_black(self):
        classifier = HistogramColorClassifier()
        attrs = classifier.classify(np.zeros((100, 100, 3), dtype=np.uint8))
        assert attrs.color == "black"

    def test_classifies_gray(self):
        classifier = HistogramColorClassifier()
        attrs = classifier.classify(np.full((100, 100, 3), 128, dtype=np.uint8))
        assert attrs.color == "gray"

    def test_empty_crop_returns_all_none(self):
        classifier = HistogramColorClassifier()
        attrs = classifier.classify(np.zeros((0, 0, 3), dtype=np.uint8))
        assert attrs == VehicleAttributes()


class _FakeColorClassifier(BaseClassifier):
    def classify(self, crop):
        return VehicleAttributes(color="red", color_confidence=0.9)


class _FakeTypeBrandClassifier(BaseClassifier):
    def classify(self, crop):
        return VehicleAttributes(vehicle_type="sedan", vehicle_type_confidence=0.8, brand="toyota", brand_confidence=0.7)


class _FakeOverlappingColorClassifier(BaseClassifier):
    def classify(self, crop):
        return VehicleAttributes(color="blue", color_confidence=0.99)


class TestCompositeVehicleClassifier:
    def test_merges_fields_with_priority_ordering(self):
        composite = CompositeVehicleClassifier(
            [_FakeColorClassifier(), _FakeTypeBrandClassifier(), _FakeOverlappingColorClassifier()]
        )
        result = composite.classify(np.zeros((10, 10, 3), dtype=np.uint8))
        assert result.color == "red"  # first classifier wins over the later overlapping one
        assert result.vehicle_type == "sedan"
        assert result.brand == "toyota"

    def test_rejects_empty_classifier_list(self):
        with pytest.raises(ValueError):
            CompositeVehicleClassifier([])
