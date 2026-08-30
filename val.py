#!/usr/bin/env python3
"""
val.py

Refactored replacement for the legacy YOLOv5 `val.py`. Same composition-
root discipline as `detect.py`/`train.py`: this file parses config and
wires concrete objects into `EvaluateUseCase`
(`src/pipelines/use_cases.py`), which does the actual mAP computation via
`src/evaluation/detection_metrics.py` -- no evaluation logic lives here.

Config strategy -- Hydra, not Pydantic Settings:
    Unlike `detect.py`/`export.py` (single deterministic actions),
    validation genuinely benefits from Hydra's `--multirun`: sweeping
    `conf_threshold`/`iou_threshold` across a grid to build a
    precision-recall curve, or comparing several checkpoints
    (`weights=runs/train/exp/weights/best.pt,runs/train/exp2/weights/best.pt`),
    is a normal validation workflow that Pydantic Settings has no
    equivalent for. This mirrors `train.py`'s reasoning exactly.

Usage:
    python val.py
    python val.py weights=runs/train/exp/weights/best.pt
    python val.py --multirun conf_threshold=0.1,0.25,0.5,0.75
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.repositories import LocalReportRepository, LocalRunRepository, LocalWeightsRepository
from src.pipelines.use_cases import EvaluateRequest, EvaluateUseCase
from src.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def build_evaluate_use_case(cfg: DictConfig) -> EvaluateUseCase:
    """Composition step -- mirrors `detect.py::build_detect_use_case` /
    `train.py::build_train_use_case`: the one place that knows concrete
    class names, everything downstream depends only on `src.core`
    interfaces."""
    from omegaconf import OmegaConf

    from src.models.yolo_adapter import YoloAdapter

    weights_repository = LocalWeightsRepository(weights_dir=cfg.weights_dir)
    resolved_weights = weights_repository.resolve(cfg.weights)

    model_cfg = OmegaConf.create({"weights": resolved_weights, "class_names": list(cfg.class_names)})
    detector = YoloAdapter(model_cfg)
    detector.warmup()

    return EvaluateUseCase(
        detector=detector,
        run_repository=LocalRunRepository(base_dir=cfg.runs_dir),
        report_repository=LocalReportRepository(),
    )


@hydra.main(config_path="configs", config_name="val", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging()
    logger.info("Resolved validation configuration:\n%s", OmegaConf.to_yaml(cfg))

    use_case = build_evaluate_use_case(cfg)
    request = EvaluateRequest(
        images_dir=cfg.images_dir,
        labels_dir=cfg.labels_dir,
        class_names=list(cfg.class_names),
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        map_iou_threshold=cfg.map_iou_threshold,
        run_name_prefix=cfg.run_name_prefix,
        save_report=cfg.save_report,
    )
    result = use_case.execute(request)

    logger.info(
        "Validation complete: mAP@%.2f = %.4f across %d image(s). Report: %s",
        cfg.map_iou_threshold,
        result.report.mean_average_precision,
        result.report.num_images,
        result.report_path,
    )
    for class_ap in result.report.per_class:
        logger.info("  %s: AP=%.4f (gt=%d, pred=%d)", class_ap.class_name, class_ap.average_precision, class_ap.num_ground_truth, class_ap.num_predictions)


if __name__ == "__main__":
    main()
