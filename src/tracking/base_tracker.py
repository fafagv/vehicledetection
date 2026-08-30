"""
src/tracking/base_tracker.py

Thin re-export of `src.core.base_model.BaseTracker` so code under
`src/tracking/` and `src/pipelines/` can `from src.tracking.base_tracker
import BaseTracker` without reaching into `src.core` directly -- keeps
the public surface of the tracking subsystem self-contained even though
the actual interface definition (shared with detectors/OCR/classifiers)
lives in `src.core.base_model`.
"""

from __future__ import annotations

from src.core.base_model import BaseTracker

__all__ = ["BaseTracker"]
