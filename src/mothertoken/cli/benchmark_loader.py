"""
mothertoken — cli/benchmark_loader.py

Loads active benchmark data and provides helpers to query it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mothertoken.core.resources import benchmark_data_path, load_benchmark_data
from mothertoken.core.user_config import get_configured_benchmark_path


@dataclass(frozen=True)
class BenchmarkSource:
    """Explains where active benchmark data came from."""

    source_type: str
    path: Path
    reason: str


def load_benchmark() -> dict[str, Any]:
    """Load the active benchmark. Raises FileNotFoundError if not found."""
    data, _source = resolve_active_benchmark()
    return data


def resolve_active_benchmark(override_path: Path | None = None) -> tuple[dict[str, Any], BenchmarkSource]:
    """Resolve benchmark data using explicit override, user config, then bundled default."""
    if override_path is not None:
        path = override_path.expanduser().resolve()
        data = load_benchmark_file(path)
        return data, BenchmarkSource("command override", path, "--benchmark")

    configured_path = get_configured_benchmark_path()
    if configured_path is not None:
        path = configured_path.expanduser().resolve()
        data = load_benchmark_file(path)
        return data, BenchmarkSource("user config", path, "config.toml benchmark.path")

    path = benchmark_data_path()
    return load_benchmark_data(), BenchmarkSource("bundled default", path, "no user benchmark configured")


def load_benchmark_file(path: Path) -> dict[str, Any]:
    """Load and validate a benchmark JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    validate_benchmark_data(data)
    return data


def validate_benchmark_data(data: dict[str, Any]) -> None:
    """Validate the benchmark fields consumed by the CLI."""
    if not isinstance(data, dict):
        raise ValueError("Benchmark JSON must be an object.")
    if not isinstance(data.get("metrics"), dict) or not data["metrics"]:
        raise ValueError("Benchmark JSON must include a non-empty metrics object.")
    tokenizers = data.get("tokenizers", data.get("models"))
    if not isinstance(tokenizers, list) or not tokenizers:
        raise ValueError("Benchmark JSON must include a non-empty tokenizers or models list.")
    for index, tokenizer in enumerate(tokenizers):
        if not isinstance(tokenizer, dict) or not tokenizer.get("id"):
            raise ValueError(f"Benchmark tokenizer entry at index {index} must include an id.")


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
