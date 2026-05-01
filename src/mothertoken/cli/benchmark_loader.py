"""
mothertoken — cli/benchmark_loader.py

Loads the bundled benchmark.json and provides helpers to query it.
"""

from __future__ import annotations

from typing import Any

from mothertoken.core.resources import load_benchmark_data


def load_benchmark() -> dict[str, Any]:
    """Load benchmark.json. Raises FileNotFoundError if not found."""
    return load_benchmark_data()


def get_languages(data: dict[str, Any]) -> list[str]:
    """Return all language codes present in the benchmark."""
    return list(data.get("metrics", {}).keys())


def get_model_ids(data: dict[str, Any]) -> list[str]:
    """Return all tokenizer IDs present in the benchmark."""
    return [m["id"] for m in data.get("tokenizers", data.get("models", []))]


def get_model_name(data: dict[str, Any], model_id: str) -> str:
    """Return the display name for a tokenizer ID, falling back to the ID itself."""
    for m in data.get("tokenizers", data.get("models", [])):
        if m["id"] == model_id:
            return m["name"]
    return model_id


def get_language_metrics(data: dict[str, Any], language: str) -> dict[str, Any]:
    """Return the per-model metrics dict for a language code."""
    return data.get("metrics", {}).get(language, {})
