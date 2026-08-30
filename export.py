#!/usr/bin/env python3
"""
export.py

Composition root for exporting a trained model to an inference-optimized
format (ONNX / OpenVINO / TensorRT), following the same discipline as
`detect.py`/`train.py`/`val.py`: parse config, wire concrete objects into
`ExportUseCase` (`src/pipelines/use_cases.py`), and nothing else. The
actual export delegates to `YoloAdapter.export`, which itself delegates
to Ultralytics' native multi-format exporter -- see that method's
docstring for why this project doesn't hand-roll ONNX/TensorRT
conversion.

Config strategy -- Pydantic Settings, not Hydra:
    Same reasoning as `detect.py`: exporting one checkpoint to one format
    is a single deterministic action with no sweep/compose need, so
    Pydantic `BaseSettings` (strict typing, `configs/export.yaml` +
    `VEHICLE_CV_EXPORT_*` env overrides) is the better fit than Hydra
    here. `val.py`/`train.py` use Hydra instead because sweeps
    (thresholds, hyperparameters) are a normal workflow for those.

Usage:
    python export.py --config configs/export.yaml
    python export.py --config configs/export.yaml --format openvino --weights runs/train/exp/weights/best.pt
    VEHICLE_CV_EXPORT_FORMAT=engine python export.py   # env var override, e.g. for a TensorRT build machine
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config_loader import load_yaml_settings
from src.data.repositories import LocalReportRepository, LocalRunRepository, LocalWeightsRepository
from src.pipelines.use_cases import ExportRequest, ExportUseCase
from src.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


class ExportSettings(BaseSettings):
    """Strictly-typed schema for `configs/export.yaml`. Any field can
    also be overridden via a `VEHICLE_CV_EXPORT_<FIELD_NAME>` environment
    variable (see `env_prefix` below)."""

    model_config = SettingsConfigDict(env_prefix="VEHICLE_CV_EXPORT_", extra="ignore")

    weights: str
    class_names: List[str]
    weights_dir: str = "weights"
    runs_dir: str = "runs/export"
    format: str = "onnx"
    output_filename: str = "model.onnx"
    image_size: int = Field(640, gt=0)
    run_name_prefix: str = "export"
    benchmark: bool = True
    num_warmup_runs: int = Field(10, ge=0)
    num_benchmark_runs: int = Field(50, ge=1)
    extra_export_kwargs: Dict[str, Any] = {}


def build_export_use_case(settings: ExportSettings) -> ExportUseCase:
    """Composition step -- mirrors `detect.py::build_detect_use_case`:
    the one place that knows concrete class names."""
    from omegaconf import OmegaConf

    from src.models.yolo_adapter import YoloAdapter

    weights_repository = LocalWeightsRepository(weights_dir=settings.weights_dir)
    resolved_weights = weights_repository.resolve(settings.weights)

    model_cfg = OmegaConf.create({"weights": resolved_weights, "class_names": settings.class_names})
    adapter = YoloAdapter(model_cfg)  # implements `Exportable` via .export()

    return ExportUseCase(
        exportable=adapter,
        run_repository=LocalRunRepository(base_dir=settings.runs_dir),
        report_repository=LocalReportRepository(),
        # benchmark_fn defaults to src.pipelines.onnx_benchmark.benchmark_onnx_model
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/export.yaml", help="Path to the export config YAML.")
    parser.add_argument("--format", default=None, help="Override the configured export format (onnx/openvino/engine/...).")
    parser.add_argument("--weights", default=None, help="Override the configured weights identifier.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    settings = load_yaml_settings(
        ExportSettings, args.config, format=args.format, weights=args.weights
    )
    use_case = build_export_use_case(settings)

    request = ExportRequest(
        format=settings.format,
        output_filename=settings.output_filename,
        image_size=settings.image_size,
        run_name_prefix=settings.run_name_prefix,
        benchmark=settings.benchmark,
        num_warmup_runs=settings.num_warmup_runs,
        num_benchmark_runs=settings.num_benchmark_runs,
        extra_export_kwargs=settings.extra_export_kwargs,
    )
    result = use_case.execute(request)

    logger.info("Exported to '%s'.", result.exported_path)
    if result.benchmark_report is not None:
        logger.info("Benchmark: %s", result.benchmark_report)


if __name__ == "__main__":
    main()
