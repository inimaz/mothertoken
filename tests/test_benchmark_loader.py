import json
from pathlib import Path

from mothertoken.cli import benchmark_loader
from mothertoken.core.user_config import (
    clear_configured_benchmark_path,
    get_configured_benchmark_path,
    set_configured_benchmark_path,
    user_benchmark_path,
    user_config_dir,
)


def _benchmark(version: str, model_id: str = "gpt-4o") -> dict:
    return {
        "version": version,
        "models": [{"id": model_id, "name": model_id}],
        "metrics": {"eng_Latn": {model_id: {"rtc": 1.0}}},
    }


def _write_benchmark(path: Path, version: str, model_id: str = "gpt-4o") -> None:
    path.write_text(json.dumps(_benchmark(version, model_id)), encoding="utf-8")


def _isolate_user_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))


def test_user_config_path_uses_platform_config_directory(monkeypatch, tmp_path):
    _isolate_user_config(monkeypatch, tmp_path)
    config_dir = user_config_dir()

    assert config_dir.name == "mothertoken"
    assert tmp_path in config_dir.parents
    assert user_benchmark_path() == config_dir / "benchmark.json"


def test_set_and_clear_configured_benchmark_path(monkeypatch, tmp_path):
    _isolate_user_config(monkeypatch, tmp_path)
    benchmark_path = tmp_path / "benchmark.json"
    _write_benchmark(benchmark_path, "user")

    saved_path = set_configured_benchmark_path(benchmark_path)

    assert saved_path == benchmark_path.resolve()
    assert get_configured_benchmark_path() == benchmark_path.resolve()

    clear_configured_benchmark_path()

    assert get_configured_benchmark_path() is None


def test_relative_configured_benchmark_path_resolves_from_config_dir(monkeypatch, tmp_path):
    _isolate_user_config(monkeypatch, tmp_path)
    config_dir = user_config_dir()
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('[benchmark]\npath = "benchmark.json"\n', encoding="utf-8")

    assert get_configured_benchmark_path() == config_dir / "benchmark.json"


def test_resolve_active_benchmark_uses_bundled_default_without_config(monkeypatch, tmp_path):
    _isolate_user_config(monkeypatch, tmp_path)
    default_path = tmp_path / "default_benchmark.json"
    default_data = _benchmark("default")

    monkeypatch.setattr(benchmark_loader, "benchmark_data_path", lambda: default_path)
    monkeypatch.setattr(benchmark_loader, "load_benchmark_data", lambda: default_data)

    data, source = benchmark_loader.resolve_active_benchmark()

    assert data["version"] == "default"
    assert source.source_type == "bundled default"
    assert source.path == default_path


def test_resolve_active_benchmark_prefers_user_config(monkeypatch, tmp_path):
    _isolate_user_config(monkeypatch, tmp_path)
    user_benchmark = tmp_path / "user_benchmark.json"
    _write_benchmark(user_benchmark, "user")
    set_configured_benchmark_path(user_benchmark)

    data, source = benchmark_loader.resolve_active_benchmark()

    assert data["version"] == "user"
    assert source.source_type == "user config"
    assert source.path == user_benchmark.resolve()


def test_resolve_active_benchmark_prefers_command_override(monkeypatch, tmp_path):
    _isolate_user_config(monkeypatch, tmp_path)
    user_benchmark = tmp_path / "user_benchmark.json"
    override_benchmark = tmp_path / "override_benchmark.json"
    _write_benchmark(user_benchmark, "user")
    _write_benchmark(override_benchmark, "override")
    set_configured_benchmark_path(user_benchmark)

    data, source = benchmark_loader.resolve_active_benchmark(override_benchmark)

    assert data["version"] == "override"
    assert source.source_type == "command override"
    assert source.path == override_benchmark.resolve()


def test_load_benchmark_file_validates_schema(tmp_path):
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps({"version": "bad"}), encoding="utf-8")

    try:
        benchmark_loader.load_benchmark_file(invalid_path)
    except ValueError as error:
        assert "metrics" in str(error)
    else:
        raise AssertionError("Expected invalid benchmark schema to fail")
