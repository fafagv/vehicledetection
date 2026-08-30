"""
src/api/dependencies.py

FastAPI dependency-injection wiring. Every heavyweight object (the
detector, the multi-stream manager) is a process-wide singleton built
once via `functools.lru_cache` and handed out through `Depends(...)` --
route handlers in `main.py` never construct these directly, which keeps
them trivially testable (override the dependency with a fake in tests)
and keeps model loading out of the request path.

Configuration is environment-variable driven (via `pydantic-settings`)
rather than Hydra: this process is a deployable service started by an
orchestrator/Dockerfile, not a research CLI, so env vars (or a `.env`
file) are the more conventional configuration surface here. Training
still goes through Hydra (`main.py` at the project root) -- the API only
ever *loads* an already-trained checkpoint.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from omegaconf import OmegaConf
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.base_model import BaseDetector
from src.pipelines.stream_tracker import MultiStreamManager

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VEHICLE_CV_", env_file=".env", extra="ignore")

    detector_weights: str = "yolov8n.pt"
    class_names: List[str] = [
        "car", "motorcycle", "bus", "truck", "bicycle",
    ]
    default_conf_threshold: float = 0.25
    default_iou_threshold: float = 0.45
    warmup_image_size: int = 640


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_detector() -> BaseDetector:
    """Build (once) and return the process-wide detector singleton.

    Uses `YoloAdapter` purely as a `BaseDetector` here (its `predict`/
    `warmup`/`class_names` methods) -- the Lightning-training half of
    that class is irrelevant to a serving process and is simply unused.
    """
    from src.models.yolo_adapter import YoloAdapter

    settings = get_settings()
    model_cfg = OmegaConf.create(
        {
            "weights": settings.detector_weights,
            "class_names": settings.class_names,
        }
    )
    detector = YoloAdapter(model_cfg)
    detector.warmup(image_size=settings.warmup_image_size)
    logger.info("Detector loaded and warmed up (weights=%s).", settings.detector_weights)
    return detector


@lru_cache(maxsize=1)
def get_stream_manager() -> MultiStreamManager:
    return MultiStreamManager()
