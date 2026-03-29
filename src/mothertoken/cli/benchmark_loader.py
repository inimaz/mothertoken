"""
mothertoken — cli/benchmark_loader.py

Loads the bundled benchmark.json and provides helpers to query it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Walk up from __file__ to find the project root containing data/benchmark.json.
# This works for both editable installs (src layout) and regular installs.
def _find_benchmark_path() -> Path:
    candidate = Path(__file__).resolve()
    for _ in range(8):
        candidate = candidate.parent
        p = candidate / "data" / "benchmark.json"
        if p.exists():
            return p
    # Fallback: standard install places data/ alongside the package root
    return Path(__file__).resolve().parent.parent.parent.parent / "data" / "benchmark.json"


_BENCHMARK_PATH = _find_benchmark_path()


def load_benchmark() -> dict[str, Any]:
    """Load benchmark.json. Raises FileNotFoundError if not found."""
    if not _BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"benchmark.json not found at {_BENCHMARK_PATH}. Run the benchmark runner first to generate it."
        )
    with open(_BENCHMARK_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_languages(data: dict[str, Any]) -> list[str]:
    """Return all language codes present in the benchmark."""
    return list(data.get("metrics", {}).keys())


def get_model_ids(data: dict[str, Any]) -> list[str]:
    """Return all model IDs present in the benchmark."""
    return [m["id"] for m in data.get("models", [])]


def get_model_name(data: dict[str, Any], model_id: str) -> str:
    """Return the display name for a model ID, falling back to the ID itself."""
    for m in data.get("models", []):
        if m["id"] == model_id:
            return m["name"]
    return model_id


def get_language_metrics(data: dict[str, Any], language: str) -> dict[str, Any]:
    """Return the per-model metrics dict for a language code."""
    return data.get("metrics", {}).get(language, {})
