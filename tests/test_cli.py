"""
Tests for mothertoken CLI commands.
"""

from unittest.mock import patch

from typer.testing import CliRunner

from mothertoken.cli.app import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_BENCHMARK = {
    "version": "2026-03-26",
    "flores_split": "dev",
    "flores_dataset": "openlanguagedata/flores_plus",
    "baseline_language": "eng_Latn",
    "models": [
        {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "type": "tiktoken",
            "ref": "o200k_base",
            "api_key_env": None,
        },
        {
            "id": "qwen2.5",
            "name": "Qwen 2.5",
            "type": "huggingface",
            "ref": "Qwen/Qwen2.5-7B",
            "api_key_env": None,
        },
        {
            "id": "claude-sonnet",
            "name": "Claude Sonnet",
            "type": "anthropic_api",
            "ref": "claude-sonnet-4-6",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
    ],
    "metrics": {
        "arb_Arab": {
            "gpt-4o": {
                "num_sentences": 997,
                "total_chars": 110574,
                "total_tokens": 35301,
                "total_words": 18759,
                "chars_per_token": 3.132,
                "fertility": 1.882,
                "rtc": 1.565,
            },
            "qwen2.5": {
                "num_sentences": 997,
                "total_chars": 110574,
                "total_tokens": 42779,
                "total_words": 18759,
                "chars_per_token": 2.585,
                "fertility": 2.28,
                "rtc": 1.845,
            },
        },
        "eng_Latn": {
            "gpt-4o": {
                "num_sentences": 997,
                "total_chars": 125194,
                "total_tokens": 25536,
                "total_words": 20954,
                "chars_per_token": 4.903,
                "fertility": 1.219,
                "rtc": 1.0,
            },
        },
    },
    "errors": {},
}

FAKE_MODELS_CONFIG = [
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "type": "tiktoken",
        "ref": "o200k_base",
        "api_key_env": None,
    },
    {
        "id": "qwen2.5",
        "name": "Qwen 2.5",
        "type": "huggingface",
        "ref": "Qwen/Qwen2.5-7B",
        "api_key_env": None,
    },
    {
        "id": "claude-sonnet",
        "name": "Claude Sonnet",
        "type": "anthropic_api",
        "ref": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
]


def _patch_benchmark():
    return patch("mothertoken.cli.app._load_benchmark_or_exit", return_value=FAKE_BENCHMARK)


def _patch_models():
    return patch("mothertoken.cli.app._load_models_config", return_value=FAKE_MODELS_CONFIG)


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------


def test_rank_valid_language():
    with _patch_benchmark():
        result = runner.invoke(app, ["rank", "arb_Arab"])
    assert result.exit_code == 0, result.output
    assert "arb_Arab" in result.output
    assert "GPT-4o" in result.output
    assert "Qwen 2.5" in result.output
    assert "Cost vs English" in result.output
    assert "Fertility" not in result.output


def test_rank_language_alias():
    with _patch_benchmark():
        result = runner.invoke(app, ["rank", "ar"])
    assert result.exit_code == 0, result.output
    assert "arb_Arab" in result.output
    assert "GPT-4o" in result.output


def test_rank_invalid_language():
    with _patch_benchmark():
        result = runner.invoke(app, ["rank", "xyz_Fake"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "not found" in (result.stderr or "").lower()
    assert "arabic" in result.output.lower() or "arabic" in (result.stderr or "").lower()


def test_rank_shows_best_model_tip():
    with _patch_benchmark():
        result = runner.invoke(app, ["rank", "arb_Arab"])
    assert result.exit_code == 0
    assert "GPT-4o" in result.output  # highest chars/token = 3.132 wins


def test_rank_missing_language_arg():
    with _patch_benchmark():
        result = runner.invoke(app, ["rank"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def test_models_lists_supported_models():
    with _patch_models():
        result = runner.invoke(app, ["models"])
    assert result.exit_code == 0, result.output
    assert "gpt-4o" in result.output
    assert "Qwen 2.5" in result.output
    assert "claude-sonnet" in result.output
    assert "local" in result.output
    assert "API" in result.output
    assert "tiktoken / o200k_base" in result.output


def test_models_local_only_hides_api_models():
    with _patch_models():
        result = runner.invoke(app, ["models", "--local-only"])
    assert result.exit_code == 0, result.output
    assert "gpt-4o" in result.output
    assert "qwen2.5" in result.output
    assert "claude-sonnet" not in result.output
    assert "API" not in result.output


def test_models_empty_config_errors():
    with patch("mothertoken.cli.app._load_models_config", return_value=[]):
        result = runner.invoke(app, ["models"])
    assert result.exit_code == 1
    assert "no models" in result.output.lower() or "no models" in (result.stderr or "").lower()


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


def test_tokenize_single_model():
    with _patch_models(), patch("mothertoken.core.tokenizers.tokenize_sentences", return_value=[3]):
        result = runner.invoke(app, ["tokenize", "ChatGPT", "--model", "gpt-4o"])
    assert result.exit_code == 0, result.output
    assert "GPT-4o" in result.output
    assert "3" in result.output
    assert "ChatGPT" in result.output
    assert "Chars/Token" in result.output
    assert "Cost Multiplier" not in result.output


def test_tokenize_all_local_models_excludes_api_models():
    call_log = []

    def fake_tokenize(model_cfg, sentences, cache, dry_run):
        call_log.append(model_cfg["id"])
        return [3]

    with _patch_models(), patch("mothertoken.core.tokenizers.tokenize_sentences", side_effect=fake_tokenize):
        result = runner.invoke(app, ["tokenize", "hello"])

    assert result.exit_code == 0, result.output
    assert call_log == ["gpt-4o", "qwen2.5"]
    assert "Claude Sonnet" not in result.output


def test_tokenize_file_input(tmp_path):
    test_file = tmp_path / "prompt.txt"
    test_file.write_text("This is a test prompt.", encoding="utf-8")
    with _patch_models(), patch("mothertoken.core.tokenizers.tokenize_sentences", return_value=[6]):
        result = runner.invoke(app, ["tokenize", "--file", str(test_file), "--model", "gpt-4o"])
    assert result.exit_code == 0, result.output
    assert "This is a test prompt" in result.output
    assert "6" in result.output


def test_tokenize_missing_input():
    with _patch_models():
        result = runner.invoke(app, ["tokenize"])
    assert result.exit_code == 1
    assert "provide text" in result.output.lower() or "provide text" in (result.stderr or "").lower()


def test_tokenize_rejects_text_and_file(tmp_path):
    test_file = tmp_path / "prompt.txt"
    test_file.write_text("This is a test prompt.", encoding="utf-8")
    with _patch_models():
        result = runner.invoke(app, ["tokenize", "hello", "--file", str(test_file)])
    assert result.exit_code == 1
    assert "not both" in result.output.lower() or "not both" in (result.stderr or "").lower()


def test_tokenize_nonexistent_file():
    with _patch_models():
        result = runner.invoke(app, ["tokenize", "--file", "/nonexistent/path.txt"])
    assert result.exit_code == 1


def test_tokenize_api_model_rejected():
    with _patch_models():
        result = runner.invoke(app, ["tokenize", "hello", "--model", "claude-sonnet"])
    assert result.exit_code == 1
    assert "api-backed" in result.output.lower() or "api-backed" in (result.stderr or "").lower()


def test_tokenize_unknown_model():
    with _patch_models():
        result = runner.invoke(app, ["tokenize", "hello", "--model", "nonexistent-model"])
    assert result.exit_code == 1


def test_tokenize_single_model_tokenizer_error_exits_1():
    with _patch_models(), patch("mothertoken.core.tokenizers.tokenize_sentences", side_effect=Exception("load failed")):
        result = runner.invoke(app, ["tokenize", "hello", "--model", "gpt-4o"])
    assert result.exit_code == 1
    assert "load failed" in result.output


def test_old_token_command_removed():
    result = runner.invoke(app, ["token", "--text", "hello", "--model", "gpt-4o"])
    assert result.exit_code != 0


def test_old_analyze_command_removed():
    result = runner.invoke(app, ["analyze", "--text", "hello"])
    assert result.exit_code != 0
