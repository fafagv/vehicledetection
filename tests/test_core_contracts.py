"""
tests/test_core_contracts.py

Unlike the heavier pipeline pieces (YOLO, OpenCV, FastAPI), everything
under `src/core/` has zero third-party dependencies -- these tests run
in ANY environment (no torch/ultralytics/opencv install required) and
are the fastest signal that the domain model and registry are correct.
"""

from __future__ import annotations

import pytest

from src.core.exceptions import ConfigurationError
from src.core.registry import available_components, get_component, register
from src.core.types import (
    BoundingBox,
    Detection,
    FrameResult,
    PlateResult,
    Track,
    TrackedVehicle,
    VehicleAttributes,
)


class TestBoundingBox:
    def test_basic_geometry(self):
        box = BoundingBox(10, 20, 30, 60)
        assert box.width == 20
        assert box.height == 40
        assert box.area == 800
        assert box.center == (20, 40)
        assert box.to_xywh() == (10, 20, 20, 40)
        assert box.to_xyxy() == (10, 20, 30, 60)

    def test_rejects_inverted_box(self):
        with pytest.raises(ValueError):
            BoundingBox(30, 20, 10, 60)  # x2 < x1

    def test_is_frozen(self):
        box = BoundingBox(0, 0, 10, 10)
        with pytest.raises(Exception):
            box.x1 = 5  # frozen dataclasses raise FrozenInstanceError


class TestTrack:
    def test_from_detection(self):
        det = Detection(bbox=BoundingBox(0, 0, 10, 10), confidence=0.9, class_id=2, class_name="bus")
        track = Track.from_detection(det, track_id=7, age_frames=3)
        assert track.track_id == 7
        assert track.age_frames == 3
        assert track.class_name == "bus"
        assert track.speed_kmh is None


class TestFrameResult:
    def test_vehicle_count(self):
        track = Track(track_id=1, bbox=BoundingBox(0, 0, 5, 5), confidence=0.8, class_id=0, class_name="car")
        vehicle = TrackedVehicle(track=track)
        result = FrameResult(frame_index=0, timestamp_s=0.0, frame_width=640, frame_height=480, vehicles=(vehicle,))
        assert result.vehicle_count == 1

    def test_empty_frame(self):
        result = FrameResult(frame_index=0, timestamp_s=0.0, frame_width=640, frame_height=480)
        assert result.vehicle_count == 0


class TestPlateAndAttributes:
    def test_plate_result_construction(self):
        plate = PlateResult(
            bbox=BoundingBox(0, 0, 100, 30), text="ABC123", detection_confidence=0.95, ocr_confidence=0.88
        )
        assert plate.text == "ABC123"

    def test_vehicle_attributes_all_optional(self):
        attrs = VehicleAttributes()
        assert attrs.vehicle_type is None
        assert attrs.color is None
        assert attrs.brand is None


class TestRegistry:
    def test_register_and_get(self):
        @register("test_namespace", "widget")
        class Widget:
            pass

        assert get_component("test_namespace", "widget") is Widget

    def test_duplicate_registration_same_class_is_idempotent(self):
        @register("test_namespace_2", "gadget")
        class Gadget:
            pass

        # Re-registering the SAME class under the same name must not raise.
        register("test_namespace_2", "gadget")(Gadget)
        assert get_component("test_namespace_2", "gadget") is Gadget

    def test_duplicate_registration_different_class_raises(self):
        @register("test_namespace_3", "conflict")
        class First:
            pass

        with pytest.raises(ConfigurationError):

            @register("test_namespace_3", "conflict")
            class Second:
                pass

    def test_unknown_namespace_raises_with_helpful_message(self):
        with pytest.raises(ConfigurationError):
            get_component("nonexistent_namespace", "anything")

    def test_unknown_name_raises_with_helpful_message(self):
        @register("test_namespace_4", "known")
        class Known:
            pass

        with pytest.raises(ConfigurationError):
            get_component("test_namespace_4", "unknown")

    def test_available_components_returns_copy(self):
        @register("test_namespace_5", "a")
        class A:
            pass

        components = available_components("test_namespace_5")
        components["b"] = object  # mutating the returned dict...
        assert "b" not in available_components("test_namespace_5")  # ...must not leak back
