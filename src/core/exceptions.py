"""
src/core/exceptions.py

Domain-specific exception hierarchy. Catching `VehicleCVError` at an API
boundary (see `src/api/main.py`) lets us translate any internal failure
into a clean HTTP error without leaking stack traces, while still
distinguishing failure modes where callers need to (e.g. retry a stream
disconnect but not a model-load failure).
"""

from __future__ import annotations


class VehicleCVError(Exception):
    """Base class for every exception raised by this platform."""


class ModelLoadError(VehicleCVError):
    """Raised when a detector/tracker/OCR backend fails to load its weights."""


class InferenceError(VehicleCVError):
    """Raised when a forward pass / prediction call fails."""


class StreamConnectionError(VehicleCVError):
    """Raised when an RTSP/video source cannot be opened or drops unrecoverably."""


class TrackerError(VehicleCVError):
    """Raised when the tracking backend fails to update or reset."""


class ExportError(VehicleCVError):
    """Raised when ONNX/TensorRT/OpenVINO export or benchmarking fails."""


class ConfigurationError(VehicleCVError):
    """Raised when a Hydra-composed config is structurally invalid for its consumer."""
