#!/usr/bin/env python3
"""
train.py

Refactored replacement for the legacy monolithic YOLOv5 `train.py`. Like
`detect.py`, this is now a thin **composition root** -- Hydra resolves
config, this function wires concrete repository/model adapters together,
and `TrainUseCase` (src/pipelines/use_cases.py) does the actual
orchestration. No business logic lives in this file.

Config strategy -- Hydra, not Pydantic Settings:
    Training genuinely benefits from Hydra's config-group composition and
    CLI overrides (`python train.py epochs=50 device=cuda:0`), and from
    `--multirun` sweeps across hyperparameters -- capabilities Pydantic
    Settings doesn't provide. `detect.py` uses Pydantic Settings instead
    because a single inference run doesn't need any of that; see its
    docstring for the full rationale. Both are legitimate, idiomatic
    choices for strictly-typed hierarchical YAML config -- this project
    uses each where it fits rather than forcing one tool everywhere.

Usage:
    python train.py
    python train.py epochs=50 device=cuda:0 batch_size=32
    python train.py dataset_identifier=my_data/data.yaml base_weights_identifier=yolov8s.pt
    python train.py --multirun batch_size=8,16,32     # Hydra sweep
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.repositories import LocalDatasetRepository, LocalRunRepository, LocalWeightsRepository
from src.pipelines.use_cases import TrainRequest, TrainUseCase
from src.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def build_train_use_case(cfg: DictConfig) -> TrainUseCase:
    """Composition step: wire concrete adapters into `TrainUseCase`.

    Mirrors `detect.py::build_detect_use_case` -- this is the one place
    that knows concrete class names, everything downstream depends only
    on `src.core` interfaces.
    """
    from src.models.yolo_adapter import YoloAdapter

    model_cfg = OmegaConf.create(
        {"weights": cfg.base_weights_identifier, "class_names": list(cfg.class_names)}
    )
    trainable = YoloAdapter(model_cfg)

    return TrainUseCase(
        trainable=trainable,
        dataset_repository=LocalDatasetRepository(datasets_dir=cfg.datasets_dir),
        weights_repository=LocalWeightsRepository(weights_dir=cfg.weights_dir),
        run_repository=LocalRunRepository(base_dir=cfg.runs_dir),
    )


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging()
    logger.info("Resolved training configuration:\n%s", OmegaConf.to_yaml(cfg))

    use_case = build_train_use_case(cfg)
    request = TrainRequest(
        dataset_identifier=cfg.dataset_identifier,
        base_weights_identifier=cfg.base_weights_identifier,
        epochs=cfg.epochs,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        device=cfg.device,
        run_name_prefix=cfg.run_name_prefix,
        extra_train_kwargs=OmegaConf.to_container(cfg.extra_train_kwargs, resolve=True),
    )
    result = use_case.execute(request)

    logger.info("Training complete. Run dir: %s", result.run_dir)


if __name__ == "__main__":
    main()
