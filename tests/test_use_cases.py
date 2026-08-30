"""
tests/test_use_cases.py

Exercises `DetectUseCase` and `TrainUseCase` against real
`Local*Repository` implementations (writing to a real temp directory) but
a **fake** detector/trainable -- exactly the payoff of the Clean
Architecture refactor: this suite runs with zero torch/ultralytics
install (only `opencv`, `numpy`, and the standard library), because the
use-case layer never imports those directly; only the concrete
`YoloAdapter` composed in `detect.py`/`train.py` does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.core.base_model import BaseDetector
from src.core.types import BoundingBox, Detection
from src.data.repositories import (
    LocalDatasetRepository,
    LocalDetectionResultsRepository,
    LocalRunRepository,
    LocalWeightsRepository,
)
from src.pipelines.use_cases import DetectRequest, DetectUseCase, TrainRequest, TrainUseCase


class FakeDetector(BaseDetector):
    """Returns a fixed, caller-supplied list of detections for every
    frame -- enough to prove `DetectUseCase` correctly iterates sources,
    calls the detector, and persists results, without needing a real
    model."""

    def __init__(self, fixed_detections: List[Detection]) -> None:
        self._fixed_detections = fixed_detections
        self.warmup_called = False
        self.predict_call_count = 0

    def predict(self, frame, conf_threshold=0.25, iou_threshold=0.45, classes=None) -> List[Detection]:
        self.predict_call_count += 1
        return list(self._fixed_detections)

    def warmup(self, image_size: int = 640) -> None:
        self.warmup_called = True

    @property
    def class_names(self) -> List[str]:
        return ["car"]


class FakeTrainable:
    """Records exactly what it was called with -- lets the test assert
    `TrainUseCase` resolved paths correctly and passed the right
    hyperparameters through, without running any real training."""

    def __init__(self) -> None:
        self.calls = []

    def native_train(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "fake-trained"}


def _write_synthetic_image(path: Path, size: int = 32) -> None:
    array = np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), array)


def _write_synthetic_video(path: Path, num_frames: int = 5, size: int = 32) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (size, size))
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture()
def fixed_detections() -> List[Detection]:
    return [
        Detection(bbox=BoundingBox(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="car"),
        Detection(bbox=BoundingBox(5, 5, 15, 15), confidence=0.7, class_id=0, class_name="car"),
    ]


class TestDetectUseCaseOnImageDirectory:
    def test_processes_every_image_and_persists_detections(self, tmp_path: Path, fixed_detections):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for i in range(3):
            _write_synthetic_image(images_dir / f"img_{i}.png")

        detector = FakeDetector(fixed_detections)
        use_case = DetectUseCase(
            detector=detector,
            weights_repository=LocalWeightsRepository(weights_dir=str(tmp_path / "weights")),
            results_repository=LocalDetectionResultsRepository(),
            run_repository=LocalRunRepository(base_dir=str(tmp_path / "runs")),
        )

        result = use_case.execute(DetectRequest(source=str(images_dir), save_results=True))

        assert len(result.detections_by_source) == 3
        assert detector.predict_call_count == 3
        for detections in result.detections_by_source.values():
            assert len(detections) == 2

        assert result.run_dir == str(tmp_path / "runs" / "exp")
        assert len(result.output_paths) == 3
        for output_path in result.output_paths:
            with open(output_path) as f:
                payload = json.load(f)
            assert len(payload) == 2
            assert payload[0]["class_name"] == "car"
            assert payload[0]["confidence"] == pytest.approx(0.9)

    def test_incrementing_run_dirs_across_multiple_executions(self, tmp_path: Path, fixed_detections):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _write_synthetic_image(images_dir / "img_0.png")

        use_case = DetectUseCase(
            detector=FakeDetector(fixed_detections),
            weights_repository=LocalWeightsRepository(weights_dir=str(tmp_path / "weights")),
            results_repository=LocalDetectionResultsRepository(),
            run_repository=LocalRunRepository(base_dir=str(tmp_path / "runs")),
        )

        first = use_case.execute(DetectRequest(source=str(images_dir)))
        second = use_case.execute(DetectRequest(source=str(images_dir)))
        third = use_case.execute(DetectRequest(source=str(images_dir)))

        assert first.run_dir == str(tmp_path / "runs" / "exp")
        assert second.run_dir == str(tmp_path / "runs" / "exp2")
        assert third.run_dir == str(tmp_path / "runs" / "exp3")

    def test_does_not_persist_when_save_results_false(self, tmp_path: Path, fixed_detections):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _write_synthetic_image(images_dir / "img_0.png")

        use_case = DetectUseCase(
            detector=FakeDetector(fixed_detections),
            weights_repository=LocalWeightsRepository(weights_dir=str(tmp_path / "weights")),
            results_repository=LocalDetectionResultsRepository(),
            run_repository=LocalRunRepository(base_dir=str(tmp_path / "runs")),
        )

        result = use_case.execute(DetectRequest(source=str(images_dir), save_results=False))

        assert result.run_dir is None
        assert result.output_paths == []
        assert len(result.detections_by_source) == 1  # detection still ran, just wasn't persisted
        assert not (tmp_path / "runs").exists()


class TestDetectUseCaseOnVideo:
    def test_respects_max_frames_cap(self, tmp_path: Path, fixed_detections):
        video_path = tmp_path / "clip.mp4"
        _write_synthetic_video(video_path, num_frames=10)

        use_case = DetectUseCase(
            detector=FakeDetector(fixed_detections),
            weights_repository=LocalWeightsRepository(weights_dir=str(tmp_path / "weights")),
            results_repository=LocalDetectionResultsRepository(),
            run_repository=LocalRunRepository(base_dir=str(tmp_path / "runs")),
        )

        result = use_case.execute(DetectRequest(source=str(video_path), max_frames=3, save_results=False))

        assert len(result.detections_by_source) == 3


class TestDetectUseCaseErrorHandling:
    def test_raises_on_missing_source(self, tmp_path: Path, fixed_detections):
        from src.core.exceptions import ConfigurationError

        use_case = DetectUseCase(
            detector=FakeDetector(fixed_detections),
            weights_repository=LocalWeightsRepository(weights_dir=str(tmp_path / "weights")),
            results_repository=LocalDetectionResultsRepository(),
            run_repository=LocalRunRepository(base_dir=str(tmp_path / "runs")),
        )

        with pytest.raises(ConfigurationError):
            use_case.execute(DetectRequest(source=str(tmp_path / "does_not_exist.png")))


class TestTrainUseCase:
    def test_resolves_paths_and_forwards_hyperparameters(self, tmp_path: Path):
        datasets_dir = tmp_path / "data"
        datasets_dir.mkdir()
        data_yaml = datasets_dir / "vehicles" / "data.yaml"
        data_yaml.parent.mkdir(parents=True)
        data_yaml.write_text("train: images/train\nval: images/val\nnc: 5\n")

        trainable = FakeTrainable()
        use_case = TrainUseCase(
            trainable=trainable,
            dataset_repository=LocalDatasetRepository(datasets_dir=str(datasets_dir)),
            weights_repository=LocalWeightsRepository(weights_dir=str(tmp_path / "weights")),
            run_repository=LocalRunRepository(base_dir=str(tmp_path / "runs")),
        )

        result = use_case.execute(
            TrainRequest(
                dataset_identifier="vehicles/data.yaml",
                base_weights_identifier="yolov8n.pt",  # not present locally -> pass-through resolution
                epochs=42,
                image_size=512,
                batch_size=8,
                device="cpu",
            )
        )

        assert result.resolved_data_yaml == str(data_yaml)
        assert result.resolved_base_weights == "yolov8n.pt"  # pass-through, as documented
        assert result.run_dir == str(tmp_path / "runs" / "train")
        assert result.native_train_result == {"status": "fake-trained"}

        assert len(trainable.calls) == 1
        call_kwargs = trainable.calls[0]
        assert call_kwargs["data"] == str(data_yaml)
        assert call_kwargs["epochs"] == 42
        assert call_kwargs["imgsz"] == 512
        assert call_kwargs["batch"] == 8
        assert call_kwargs["device"] == "cpu"

    def test_raises_on_unresolvable_dataset(self, tmp_path: Path):
        from src.core.exceptions import ConfigurationError

        use_case = TrainUseCase(
            trainable=FakeTrainable(),
            dataset_repository=LocalDatasetRepository(datasets_dir=str(tmp_path / "data")),
            weights_repository=LocalWeightsRepository(weights_dir=str(tmp_path / "weights")),
            run_repository=LocalRunRepository(base_dir=str(tmp_path / "runs")),
        )

        with pytest.raises(ConfigurationError):
            use_case.execute(
                TrainRequest(dataset_identifier="nonexistent/data.yaml", base_weights_identifier="yolov8n.pt")
            )
