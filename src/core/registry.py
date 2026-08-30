"""
src/core/registry.py

A small, namespaced component registry used across the platform so new
detectors, trackers, OCR backends, or classifiers can be plugged in via
config alone (Open/Closed Principle) -- no `if/elif` chains in
`src/pipelines/*.py` or `src/api/dependencies.py`.

Namespaces keep unrelated component kinds from colliding on name (e.g. a
"bytetrack" tracker and a hypothetical "bytetrack" detector variant can
coexist):

    @register("tracker", "bytetrack")
    class ByteTrackTracker(BaseTracker):
        ...

    tracker_cls = get_component("tracker", "bytetrack")
    tracker = tracker_cls(**cfg)
"""

from __future__ import annotations

from typing import Callable, Dict, Type, TypeVar

from src.core.exceptions import ConfigurationError

T = TypeVar("T")

_REGISTRY: Dict[str, Dict[str, type]] = {}


def register(namespace: str, name: str) -> Callable[[Type[T]], Type[T]]:
    """Class decorator registering `cls` under `(namespace, name)`.

    Args:
        namespace: component kind, e.g. "detector", "tracker", "ocr",
            "classifier". Freeform but should match the namespaces
            consumed by `get_component` call sites.
        name: the identifier used in config (e.g. `tracker.backend:
            bytetrack`) to select this implementation.
    """

    def _wrap(cls: Type[T]) -> Type[T]:
        bucket = _REGISTRY.setdefault(namespace, {})
        if name in bucket and bucket[name] is not cls:
            raise ConfigurationError(
                f"Component '{name}' is already registered in namespace "
                f"'{namespace}' as {bucket[name]!r}; refusing to overwrite "
                f"with {cls!r}."
            )
        bucket[name] = cls
        return cls

    return _wrap


def get_component(namespace: str, name: str) -> type:
    """Look up a registered class by `(namespace, name)`.

    Raises:
        ConfigurationError: if the namespace or name isn't registered,
            with the list of currently-available names to help fix a typo
            in a Hydra config override.
    """
    bucket = _REGISTRY.get(namespace)
    if bucket is None:
        raise ConfigurationError(
            f"No components have been registered under namespace '{namespace}'. "
            f"Known namespaces: {sorted(_REGISTRY) or '<none>'}."
        )
    if name not in bucket:
        raise ConfigurationError(
            f"Unknown component '{name}' in namespace '{namespace}'. "
            f"Available: {sorted(bucket)}."
        )
    return bucket[name]


def available_components(namespace: str) -> Dict[str, type]:
    """Return a copy of everything registered under `namespace` (for
    introspection, CLI `--list-backends` commands, and tests)."""
    return dict(_REGISTRY.get(namespace, {}))
