"""User-owned mothertoken configuration."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any


def user_config_dir() -> Path:
    """Return the platform-appropriate mothertoken config directory."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "mothertoken"
        return Path.home() / "AppData" / "Roaming" / "mothertoken"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "mothertoken"

    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "mothertoken"
    return Path.home() / ".config" / "mothertoken"


def user_config_path() -> Path:
    """Return the user config TOML path."""
    return user_config_dir() / "config.toml"


def user_benchmark_path() -> Path:
    """Return the default user-owned benchmark JSON path."""
    return user_config_dir() / "benchmark.json"


def load_user_config(path: Path | None = None) -> dict[str, Any]:
    """Load user config, returning an empty config when no file exists."""
    config_path = path or user_config_path()
    if not config_path.exists():
        return {}
    with config_path.open("rb") as file:
        return tomllib.load(file)


def save_user_config(config: dict[str, Any], path: Path | None = None) -> None:
    """Write user config TOML."""
    config_path = path or user_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_dump_config(config), encoding="utf-8")


def get_configured_benchmark_path(config: dict[str, Any] | None = None) -> Path | None:
    """Return the configured active benchmark path, if any."""
    cfg = load_user_config() if config is None else config
    value = (cfg.get("benchmark") or {}).get("path")
    if not value:
        return None

    path = Path(value)
    if path.is_absolute():
        return path
    return user_config_path().parent / path


def set_configured_benchmark_path(benchmark_path: Path) -> Path:
    """Persist an absolute active benchmark path and return it."""
    resolved_path = benchmark_path.expanduser().resolve()
    config = load_user_config()
    benchmark_config = dict(config.get("benchmark") or {})
    benchmark_config["path"] = str(resolved_path)
    config["benchmark"] = benchmark_config
    save_user_config(config)
    return resolved_path


def clear_configured_benchmark_path() -> None:
    """Clear the active benchmark override while preserving other config."""
    config = load_user_config()
    benchmark_config = dict(config.get("benchmark") or {})
    benchmark_config.pop("path", None)
    if benchmark_config:
        config["benchmark"] = benchmark_config
    else:
        config.pop("benchmark", None)
    save_user_config(config)


def _dump_config(config: dict[str, Any]) -> str:
    lines: list[str] = []
    benchmark = config.get("benchmark")
    if isinstance(benchmark, dict) and benchmark:
        lines.append("[benchmark]")
        for key, value in benchmark.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_string(str(value))}")
        lines.append("")
    return "\n".join(lines)


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
