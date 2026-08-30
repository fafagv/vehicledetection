#!/usr/bin/env python3
"""
detect.py

Refactored replacement for the legacy monolithic YOLOv5 `detect.py`.
This file is now a thin **composition root**: it does exactly two things
-- (1) load and validate configuration, (2) construct concrete objects
and inject them into `DetectUseCase` -- and contains zero detection
business logic itself (that all lives in
`src/pipelines/use_cases.py::DetectUseCase`, which is unit-tested against
fakes in `tests/test_use_cases.py` with no dependency on this file at
all).

Config strategy -- Pydantic Settings, not Hydra:
    `detect.py` is a single, direct inference run against one source
    with one set of thresholds -- there's no meaningful "sweep" or
    multi-run composition need the way there is for training. Pydantic
    `BaseSettings` gives us exactly what this script needs (strict
    typing, YAML file + environment variable + CLI-flag layering) with
    less ceremony than Hydra for a single-config script. `train.py`,
    which genuinely benefits from Hydra's override/compose/multirun
    ergonomics, uses Hydra instead -- see that file's docstring.

The only `argparse` left is `--config`/`--source`/`--weights`, which
point AT configuration rather than encode it (Pydantic Settings does the
actual, strictly-typed parsing) -- this is the one legitimate use of
lightweight CLI parsing the refactor keeps, versus the legacy script's
~20 hand-rolled, untyped `argparse` flags.

Usage:
    python detect.py --config configs/model.yaml
    python detect.py --config configs/model.yaml --source path/to/video.mp4
    VEHICLE_CV_DETECT_CONF_THRESHOLD=0.4 python detect.py  # env var override
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config_loader import load_yaml_settings
from src.core.repositories import DatasetRepository  # noqa: F401 (documents the DI seam, unused directly here)
from src.data.repositories import (
    LocalDetectionResultsRepository,
    LocalRunRepository,
    LocalWeightsRepository,
)
from src.pipelines.use_cases import DetectRequest, DetectUseCase
from src.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


class DetectSettings(BaseSettings):
    """Strictly-typed schema for `configs/model.yaml`. Any field can also
    be overridden via a `VEHICLE_CV_DETECT_<FIELD_NAME>` environment
    variable (see `env_prefix` below) without touching the YAML file --
    handy for CI or containerized runs."""

    model_config = SettingsConfigDict(env_prefix="VEHICLE_CV_DETECT_", extra="ignore")

    weights: str
    class_names: List[str]
    weights_dir: str = "weights"
    runs_dir: str = "runs/detect"
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    source: str
    save_results: bool = True
    run_name_prefix: str = "exp"
    max_frames: Optional[int] = None


def build_detect_use_case(settings: DetectSettings) -> DetectUseCase:
    """Composition step: wire concrete adapters into `DetectUseCase`.

    This is the ONLY place in the whole detect path that knows concrete
    class names (`YoloAdapter`, `LocalWeightsRepository`, ...) -- swap any
    of them here (e.g. `S3WeightsRepository` in a cloud deployment)
    without touching `DetectUseCase` or its tests.
    """
    from omegaconf import OmegaConf

    from src.models.yolo_adapter import YoloAdapter

    weights_repository = LocalWeightsRepository(weights_dir=settings.weights_dir)
    resolved_weights = weights_repository.resolve(settings.weights)

    model_cfg = OmegaConf.create({"weights": resolved_weights, "class_names": settings.class_names})
    detector = YoloAdapter(model_cfg)
    detector.warmup()

    return DetectUseCase(
        detector=detector,
        weights_repository=weights_repository,
        results_repository=LocalDetectionResultsRepository(),
        run_repository=LocalRunRepository(base_dir=settings.runs_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model.yaml", help="Path to the detect config YAML.")
    parser.add_argument("--source", default=None, help="Override the configured source (image/dir/video).")
    parser.add_argument("--weights", default=None, help="Override the configured weights identifier.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    settings = load_yaml_settings(
        DetectSettings, args.config, source=args.source, weights=args.weights
    )
    use_case = build_detect_use_case(settings)

    request = DetectRequest(
        source=settings.source,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
        save_results=settings.save_results,
        run_name_prefix=settings.run_name_prefix,
        max_frames=settings.max_frames,
    )
    result = use_case.execute(request)

    total_detections = sum(len(d) for d in result.detections_by_source.values())
    logger.info(
        "Processed %d source(s), %d total detection(s). Run dir: %s",
        len(result.detections_by_source),
        total_detections,
        result.run_dir,
    )


if __name__ == "__main__":
    main()
