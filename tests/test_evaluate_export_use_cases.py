"""
tests/test_evaluate_export_use_cases.py

`EvaluateUseCase` exercised end-to-end against real image files and real
YOLO-format label files on disk (only `opencv`/`numpy` needed).
`ExportUseCase` exercised entirely against fakes (a `FakeExportable` and
an injected `benchmark_fn`) -- proving the DI seam actually works without
needing a real `ultralytics` install or a real ONNX Runtime session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.core.base_model import BaseDetector
from src.core.types import BoundingBox, Detection
from src.data.repositories import LocalReportRepository, LocalRunRepository
from src.pipelines.use_cases import EvaluateRequest, EvaluateUseCase, ExportRequest, ExportUseCase


class _FixedBoxDetector(BaseDetector):
    """Always returns one detection at a fixed absolute box location,
    regardless of the input image -- lets the test control exactly which
    predictions are TP/FP against known ground truth."""

    def __init__(self, box, conf: float = 0.9) -> None:
        self.box = box
        self.conf = conf
        self.warmup_called = False

    def predict(self, frame, conf_threshold=0.25, iou_threshold=0.45, classes=None) -> List[Detection]:
        return [Detection(bbox=BoundingBox(*self.box), confidence=self.conf, class_id=0, class_name="car")]

    def warmup(self, image_size: int = 640) -> None:
        self.warmup_called = True

    @property
    def class_names(self) -> List[str]:
        return ["car"]


class TestEvaluateUseCase:
    def test_end_to_end_against_real_labeled_dataset(self, tmp_path: Path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        size = 100

        # img1: GT box exactly at (10,10,30,30) -- detector predicts the
        # same box -> TP.
        cv2.imwrite(str(images_dir / "img1.png"), np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
        (labels_dir / "img1.txt").write_text(f"0 {20/size} {20/size} {20/size} {20/size}\n")

        # img2: GT box at (60,60,80,80) -- detector still predicts
        # (10,10,30,30) -> FP, and this GT is never matched.
        cv2.imwrite(str(images_dir / "img2.png"), np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
        (labels_dir / "img2.txt").write_text(f"0 {70/size} {70/size} {20/size} {20/size}\n")

        detector = _FixedBoxDetector(box=(10, 10, 30, 30), conf=0.9)
        use_case = EvaluateUseCase(
            detector=detector,
            run_repository=LocalRunRepository(base_dir=str(tmp_path / "runs")),
            report_repository=LocalReportRepository(),
        )
        result = use_case.execute(EvaluateRequest(images_dir=str(images_dir), labels_dir=str(labels_dir), class_names=["car"]))

        assert result.report.num_images == 2
        assert abs(result.report.mean_average_precision - 0.5) < 1e-6
        assert result.report_path is not None
        with open(result.report_path) as f:
            saved = json.load(f)
        assert abs(saved["mean_average_precision"] - 0.5) < 1e-6

    def test_does_not_call_detector_warmup_itself(self, tmp_path: Path):
        # Warmup is the composition root's job (see val.py), not the use case's.
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        cv2.imwrite(str(images_dir / "img1.png"), np.zeros((50, 50, 3), dtype=np.uint8))
        (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        detector = _FixedBoxDetector(box=(0, 0, 10, 10))
        use_case = EvaluateUseCase(
            detector=detector,
            run_repository=LocalRunRepository(base_dir=str(tmp_path / "runs")),
            report_repository=LocalReportRepository(),
        )
        use_case.execute(EvaluateRequest(images_dir=str(images_dir), labels_dir=str(labels_dir), class_names=["car"]))
        assert detector.warmup_called is False


class _FakeExportable:
    def __init__(self) -> None:
        self.export_calls = []

    def export(self, format: str, output_path: str, **kwargs) -> str:
        self.export_calls.append((format, output_path, kwargs))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("fake-onnx-bytes")
        return output_path


class TestExportUseCase:
    def test_onnx_export_triggers_injected_benchmark(self, tmp_path: Path):
        exportable = _FakeExportable()
        benchmark_calls = []

        def fake_benchmark_fn(onnx_path, image_size, num_warmup, num_runs):
            benchmark_calls.append((onnx_path, image_size, num_warmup, num_runs))
            return {"mean_latency_ms": 5.0, "throughput_fps": 200.0}

        use_case = ExportUseCase(
            exportable=exportable,
            run_repository=LocalRunRepository(base_dir=str(tmp_path / "exports")),
            report_repository=LocalReportRepository(),
            benchmark_fn=fake_benchmark_fn,
        )
        result = use_case.execute(ExportRequest(format="onnx", output_filename="model.onnx", image_size=320))

        assert Path(result.exported_path).exists()
        assert len(exportable.export_calls) == 1
        assert exportable.export_calls[0][2]["imgsz"] == 320
        assert len(benchmark_calls) == 1
        assert result.benchmark_report == {"mean_latency_ms": 5.0, "throughput_fps": 200.0}
        assert result.report_path is not None and Path(result.report_path).exists()

    def test_non_onnx_format_skips_benchmark(self, tmp_path: Path):
        exportable = _FakeExportable()
        calls = []

        use_case = ExportUseCase(
            exportable=exportable,
            run_repository=LocalRunRepository(base_dir=str(tmp_path / "exports")),
            report_repository=LocalReportRepository(),
            benchmark_fn=lambda *a: calls.append(a) or {},
        )
        result = use_case.execute(ExportRequest(format="openvino", output_filename="model.xml", benchmark=True))

        assert result.benchmark_report is None
        assert calls == []
