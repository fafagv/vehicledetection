"""
src/config_loader.py

A tiny, generic helper for the "strictly-typed, hierarchical YAML"
config pattern used by `detect.py`: load a YAML file into a dict, apply
any explicit overrides (e.g. from a minimal `--source` CLI flag), then
validate/coerce the merged result through a Pydantic model so the rest
of the program works with a typed object instead of a loose dict.

Precedence (highest wins): explicit `overrides` kwargs > YAML file values
> the Pydantic model's own field defaults > (for any field present in
neither the YAML nor the overrides) environment variables, if the model
is a `pydantic_settings.BaseSettings` subclass -- BaseSettings' own env
lookup still applies to fields we didn't explicitly supply.

`train.py` does NOT use this: Hydra already provides the equivalent
(and considerably more capable -- CLI overrides, multirun sweeps, config
composition) hierarchical-YAML mechanism for that composition root,
which is why the two scripts deliberately demonstrate two different,
independently idiomatic config approaches (see the docstrings in
`detect.py`/`train.py` for the rationale).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type, TypeVar

import yaml

T = TypeVar("T")


def load_yaml_settings(model_cls: Type[T], yaml_path: str, **overrides: Any) -> T:
    """Load `yaml_path`, apply `overrides` on top, and validate the result
    against `model_cls` (typically a `pydantic.BaseModel` or
    `pydantic_settings.BaseSettings` subclass)."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: '{path}'.")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected '{path}' to contain a YAML mapping, got {type(data).__name__}.")

    data.update({k: v for k, v in overrides.items() if v is not None})
    return model_cls(**data)
