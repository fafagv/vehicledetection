"""
src/core/repositories.py

Repository-pattern interfaces separating the legacy `detect.py`/`train.py`
scripts' business logic (now in `src/pipelines/use_cases.py`) from *where*
weights, datasets, and results actually live on disk (or, later, in S3 /
an MLflow model registry / a database -- swapping any of those in is a
new adapter class, never a change to the use cases that depend on these
interfaces).

Zero third-party imports here on purpose -- same rule as
`src/core/base_model.py`: the center of a Clean Architecture never
depends on its outer layers (filesystem, Hydra, Pydantic all live in
`src/data/repositories.py` and the composition roots `detect.py`/
`train.py` instead).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.core.types import Detection


class WeightsRepository(ABC):
    """Resolves a weights *identifier* (a bare name, a relative path, an
    Ultralytics-recognized alias like "yolov8n.pt", ...) to something a
    model loader can actually open, and persists newly-trained weights
    under a chosen identifier."""

    @abstractmethod
    def resolve(self, identifier: str) -> str:
        """Return a path/URI a model loader can open for `identifier`."""
        raise NotImplementedError

    @abstractmethod
    def save(self, source_path: str, identifier: str) -> str:
        """Persist the weights file at `source_path` under `identifier`,
        returning where it now lives."""
        raise NotImplementedError


class DatasetRepository(ABC):
    """Resolves a dataset identifier to an Ultralytics-format `data.yaml`
    path. An adapter backed by DVC or a data catalog would implement the
    same interface without any use-case-layer change."""

    @abstractmethod
    def resolve_data_yaml(self, identifier: str) -> str:
        raise NotImplementedError


class RunRepository(ABC):
    """Allocates a fresh, non-colliding run directory -- the same
    `runs/<kind>/exp`, `exp2`, `exp3`, ... convention the legacy
    YOLOv5 scripts used, so this refactor doesn't change the on-disk
    layout users already have tooling/muscle-memory around."""

    @abstractmethod
    def new_run_dir(self, prefix: str = "exp") -> str:
        raise NotImplementedError


class DetectionResultsRepository(ABC):
    """Persists the detections produced for one processed source (a
    single image or one video's worth of frames)."""

    @abstractmethod
    def save_detections(self, run_dir: str, source_name: str, detections: List[Detection]) -> str:
        """Write `detections` for `source_name` under `run_dir`, returning
        the path written to."""
        raise NotImplementedError


class ReportRepository(ABC):
    """Persists a single structured JSON report -- shared by
    `EvaluateUseCase` (metrics report) and `ExportUseCase` (benchmark
    report) so both get the same on-disk convention without either
    depending on the other's use case.
    """

    @abstractmethod
    def save_report(self, run_dir: str, name: str, data: dict) -> str:
        """Write `data` as `<run_dir>/<name>.json`, returning the path."""
        raise NotImplementedError
