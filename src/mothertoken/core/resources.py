"""
Helpers for loading bundled mothertoken data from source checkouts or wheels.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


def _find_data_path(filename: str) -> Path:
    """Find a data file in package resources."""
    try:
        resource_path = resources.files("mothertoken.data").joinpath(filename)
    except ModuleNotFoundError:
        resource_path = None
    if resource_path is not None and resource_path.is_file():
        return Path(str(resource_path))

    return Path(__file__).resolve().parent.parent / "data" / filename


def load_benchmark_data() -> dict[str, Any]:
    """Load bundled benchmark.json."""
    benchmark_path = _find_data_path("benchmark.json")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"benchmark.json not found at {benchmark_path}. Reinstall mothertoken.")
    with open(benchmark_path, encoding="utf-8") as f:
        return json.load(f)


def load_models_config() -> dict[str, Any]:
    """Load bundled models.yaml."""
    import yaml

    config_path = _find_data_path("models.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"models.yaml not found at {config_path}. Reinstall mothertoken.")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
