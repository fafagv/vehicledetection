"""
src/pipelines/use_cases.py

The application/use-case layer that replaces the business logic
previously buried inside the legacy monolithic `detect.py`/`train.py`
scripts. Everything here is constructed via **dependency injection** --
a `BaseDetector`, a `Trainable`, and whichever repository
implementations the caller wants -- and depends only on
`src.core.*` interfaces, never on a concrete `YoloAdapter`,
`LocalWeightsRepository`, Hydra, or Pydantic.

That's what makes this layer trivially unit-testable with fakes (see
`tests/test_use_cases.py`, which exercises both use cases for real using
in-memory fakes -- no torch/ultralytics/opencv required) and what makes
`detect.py`/`train.py` themselves shrink down to thin composition roots:
their only job is to parse config and wire concrete objects together,
never to contain business logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.core.base_model import BaseDetector, Exportable, Trainable
from src.core.exceptions import ConfigurationError, InferenceError
from src.core.repositories import (
    DatasetRepository,
    DetectionResultsRepository,
    ReportRepository,
    RunRepository,
    WeightsRepository,
)
from src.core.types import Detection
from src.evaluation.detection_metrics import EvaluationReport, evaluate_detections

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# --------------------------------------------------------------------- #
# Detect
# --------------------------------------------------------------------- #
@dataclass
class DetectRequest:
    source: str                          # image file, directory of images, or video file
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    save_results: bool = True
    run_name_prefix: str = "exp"
    max_frames: Optional[int] = None     # cap frames processed for a video source (None = all)


@dataclass
class DetectResult:
    run_dir: Optional[str]
    detections_by_source: Dict[str, List[Detection]] = field(default_factory=dict)
    output_paths: List[str] = field(default_factory=list)


class DetectUseCase:
    """Runs detection over an image, a directory of images, or a video,
    optionally persisting results via the injected `DetectionResultsRepository`.
    """

    def __init__(
        self,
        detector: BaseDetector,
        weights_repository: WeightsRepository,
        results_repository: DetectionResultsRepository,
        run_repository: RunRepository,
    ) -> None:
        self._detector = detector
        self._weights_repository = weights_repository
        self._results_repository = results_repository
        self._run_repository = run_repository

    def execute(self, request: DetectRequest) -> DetectResult:
        source_path = Path(request.source)
        if not source_path.exists():
            raise ConfigurationError(f"Detection source '{request.source}' does not exist.")

        run_dir = self._run_repository.new_run_dir(request.run_name_prefix) if request.save_results else None
        result = DetectResult(run_dir=run_dir)

        for source_name, frame in self._iter_frames(source_path, request.max_frames):
            try:
                detections = self._detector.predict(
                    frame, conf_threshold=request.conf_threshold, iou_threshold=request.iou_threshold
                )
            except InferenceError:
                logger.exception("Detection failed on '%s'; skipping.", source_name)
                continue

            result.detections_by_source[source_name] = detections

            if request.save_results and run_dir is not None:
                output_path = self._results_repository.save_detections(run_dir, source_name, detections)
                result.output_paths.append(output_path)

        return result

    def _iter_frames(self, source_path: Path, max_frames: Optional[int]):
        if source_path.is_dir():
            for image_path in sorted(source_path.iterdir()):
                if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                frame = self._read_image(image_path)
                if frame is not None:
                    yield image_path.name, frame

        elif source_path.suffix.lower() in _VIDEO_EXTENSIONS:
            yield from self._iter_video_frames(source_path, max_frames)

        elif source_path.suffix.lower() in _IMAGE_EXTENSIONS:
            frame = self._read_image(source_path)
            if frame is not None:
                yield source_path.name, frame

        else:
            raise ConfigurationError(
                f"Unsupported source type for '{source_path}' "
                f"(expected an image, a directory of images, or a video file)."
            )

    @staticmethod
    def _read_image(path: Path):
        import cv2

        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Could not read image '%s'; skipping.", path)
        return frame

    @staticmethod
    def _iter_video_frames(path: Path, max_frames: Optional[int]):
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ConfigurationError(f"Could not open video source '{path}'.")

        frame_index = 0
        try:
            while max_frames is None or frame_index < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                yield f"{path.stem}_frame{frame_index:06d}{path.suffix}", frame
                frame_index += 1
        finally:
            cap.release()


# --------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------- #
@dataclass
class TrainRequest:
    dataset_identifier: str
    base_weights_identifier: str
    epochs: int = 100
    image_size: int = 640
    batch_size: int = 16
    device: str = "cpu"
    run_name_prefix: str = "train"
    extra_train_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    run_dir: str
    resolved_data_yaml: str
    resolved_base_weights: str
    native_train_result: Any = None


class TrainUseCase:
    """Resolves a dataset + base weights via their repositories, allocates
    a run directory, and delegates the actual training loop to the
    injected `Trainable` (e.g. `YoloAdapter.native_train`).
    """

    def __init__(
        self,
        trainable: Trainable,
        dataset_repository: DatasetRepository,
        weights_repository: WeightsRepository,
        run_repository: RunRepository,
    ) -> None:
        self._trainable = trainable
        self._dataset_repository = dataset_repository
        self._weights_repository = weights_repository
        self._run_repository = run_repository

    def execute(self, request: TrainRequest) -> TrainResult:
        data_yaml = self._dataset_repository.resolve_data_yaml(request.dataset_identifier)
        base_weights = self._weights_repository.resolve(request.base_weights_identifier)
        run_dir = self._run_repository.new_run_dir(request.run_name_prefix)

        logger.info(
            "Starting training: data=%s weights=%s epochs=%d run_dir=%s",
            data_yaml,
            base_weights,
            request.epochs,
            run_dir,
        )

        native_result = self._trainable.native_train(
            data=data_yaml,
            epochs=request.epochs,
            imgsz=request.image_size,
            batch=request.batch_size,
            device=request.device,
            project=str(Path(run_dir).parent),
            name=Path(run_dir).name,
            exist_ok=True,
            **request.extra_train_kwargs,
        )

        return TrainResult(
            run_dir=run_dir,
            resolved_data_yaml=data_yaml,
            resolved_base_weights=base_weights,
            native_train_result=native_result,
        )


# --------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------- #
@dataclass
class EvaluateRequest:
    images_dir: str
    labels_dir: str
    class_names: List[str]
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    map_iou_threshold: float = 0.5
    run_name_prefix: str = "eval"
    save_report: bool = True


@dataclass
class EvaluateResult:
    report: EvaluationReport
    run_dir: Optional[str] = None
    report_path: Optional[str] = None


class EvaluateUseCase:
    """Runs a detector over a YOLO-format labeled dataset and computes
    mAP (via `src.evaluation.detection_metrics.evaluate_detections`).

    `read_yolo_labelled_dataset` is imported lazily inside `execute`
    (rather than at module level) purely to keep this module's own import
    graph free of the `cv2` dependency until an evaluation actually runs
    -- consistent with the rest of this layer depending only on
    `src.core` at import time.
    """

    def __init__(
        self,
        detector: BaseDetector,
        run_repository: RunRepository,
        report_repository: ReportRepository,
    ) -> None:
        self._detector = detector
        self._run_repository = run_repository
        self._report_repository = report_repository

    def execute(self, request: EvaluateRequest) -> EvaluateResult:
        from src.data.yolo_label_reader import read_yolo_labelled_dataset

        image_paths, ground_truths_by_image = read_yolo_labelled_dataset(
            request.images_dir, request.labels_dir, request.class_names
        )

        predictions_by_image: Dict[str, List[Detection]] = {}
        for image_name, image_path in image_paths.items():
            frame = self._read_image(image_path)
            try:
                predictions_by_image[image_name] = self._detector.predict(
                    frame, conf_threshold=request.conf_threshold, iou_threshold=request.iou_threshold
                )
            except InferenceError:
                logger.exception("Detection failed on '%s' during evaluation; treating as zero predictions.", image_path)
                predictions_by_image[image_name] = []

        report = evaluate_detections(
            predictions_by_image, ground_truths_by_image, iou_threshold=request.map_iou_threshold
        )

        run_dir: Optional[str] = None
        report_path: Optional[str] = None
        if request.save_report:
            run_dir = self._run_repository.new_run_dir(request.run_name_prefix)
            report_path = self._report_repository.save_report(run_dir, "evaluation_report", report.to_dict())
            logger.info("Evaluation complete: mAP@%.2f = %.4f. Report: %s", request.map_iou_threshold, report.mean_average_precision, report_path)
        else:
            logger.info("Evaluation complete: mAP@%.2f = %.4f.", request.map_iou_threshold, report.mean_average_precision)

        return EvaluateResult(report=report, run_dir=run_dir, report_path=report_path)

    @staticmethod
    def _read_image(path: str):
        import cv2

        frame = cv2.imread(path)
        if frame is None:
            raise ConfigurationError(f"Could not read image '{path}' during evaluation.")
        return frame


# --------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------- #
BenchmarkFn = Callable[[str, int, int, int], Dict[str, Any]]


@dataclass
class ExportRequest:
    format: str                     # "onnx" | "openvino" | "engine" (TensorRT) | ...
    output_filename: str            # just the filename; use case decides the run dir
    image_size: int = 640
    run_name_prefix: str = "export"
    benchmark: bool = True          # only actually runs for format == "onnx" (see ExportUseCase)
    num_warmup_runs: int = 10
    num_benchmark_runs: int = 50
    extra_export_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportResult:
    exported_path: str
    run_dir: str
    benchmark_report: Optional[Dict[str, Any]] = None
    report_path: Optional[str] = None


class ExportUseCase:
    """Exports a model to an inference-optimized format via the injected
    `Exportable` (e.g. `YoloAdapter.export`, which itself delegates to
    Ultralytics' native exporter -- see that method's docstring), then
    optionally benchmarks the result.

    `benchmark_fn` is itself injected (defaulting to
    `src.pipelines.onnx_benchmark.benchmark_onnx_model`) specifically so
    tests can substitute a fake without needing a real ONNX runtime
    session or a real exported model file -- the same DI principle
    applied to every other use case in this module, extended one level
    further here because "benchmark an ONNX file" is itself an external
    concern (onnxruntime) worth being able to swap/fake independently of
    "export a model".
    """

    def __init__(
        self,
        exportable: Exportable,
        run_repository: RunRepository,
        report_repository: ReportRepository,
        benchmark_fn: Optional[BenchmarkFn] = None,
    ) -> None:
        self._exportable = exportable
        self._run_repository = run_repository
        self._report_repository = report_repository
        if benchmark_fn is not None:
            self._benchmark_fn = benchmark_fn
        else:
            from src.pipelines.onnx_benchmark import benchmark_onnx_model

            self._benchmark_fn = benchmark_onnx_model

    def execute(self, request: ExportRequest) -> ExportResult:
        run_dir = self._run_repository.new_run_dir(request.run_name_prefix)
        output_path = str(Path(run_dir) / request.output_filename)

        exported_path = self._exportable.export(
            format=request.format,
            output_path=output_path,
            imgsz=request.image_size,
            **request.extra_export_kwargs,
        )
        logger.info("Exported model to '%s' (format=%s).", exported_path, request.format)

        benchmark_report: Optional[Dict[str, Any]] = None
        report_path: Optional[str] = None
        if request.benchmark and request.format == "onnx":
            benchmark_report = self._benchmark_fn(
                exported_path, request.image_size, request.num_warmup_runs, request.num_benchmark_runs
            )
            report_path = self._report_repository.save_report(run_dir, "benchmark_report", benchmark_report)
            logger.info("Benchmark: %s", benchmark_report)
        elif request.benchmark:
            logger.info("Skipping benchmark: only 'onnx' exports are benchmarked in-process (got format='%s').", request.format)

        return ExportResult(
            exported_path=exported_path,
            run_dir=run_dir,
            benchmark_report=benchmark_report,
            report_path=report_path,
        )

