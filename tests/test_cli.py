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
            "cost_source_id": "gpt-4o",
        },
        {
            "id": "qwen2.5",
            "name": "Qwen 2.5",
            "type": "huggingface",
            "ref": "Qwen/Qwen2.5-7B",
            "api_key_env": None,
            "cost_source_id": "qwen2.5",
        },
        {
            "id": "claude-sonnet",
            "name": "Claude Sonnet",
            "type": "anthropic_api",
            "ref": "claude-sonnet-4-6",
            "api_key_env": "ANTHROPIC_API_KEY",
            "cost_source_id": "claude-3-5-sonnet-20241022",
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
                "cost_per_1m_chars": 0.79813,
                "fair_cost_delta": 0.2882,
            },
            "qwen2.5": {
                "num_sentences": 997,
                "total_chars": 110574,
                "total_tokens": 42779,
                "total_words": 18759,
                "chars_per_token": 2.585,
                "fertility": 2.28,
                "rtc": 1.845,
                "cost_per_1m_chars": 0.0,
                "fair_cost_delta": 0.0,
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
                "cost_per_1m_chars": 0.50993,
                "fair_cost_delta": 0.0,
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
        "cost_source_id": "gpt-4o",
    },
    {
        "id": "qwen2.5",
        "name": "Qwen 2.5",
        "type": "huggingface",
        "ref": "Qwen/Qwen2.5-7B",
        "api_key_env": None,
        "cost_source_id": "qwen2.5",
    },
    {
        "id": "claude-sonnet",
        "name": "Claude Sonnet",
        "type": "anthropic_api",
        "ref": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "cost_source_id": "claude-3-5-sonnet-20241022",
    },
]


def _patch_benchmark():
    return patch("mothertoken.cli.app._load_benchmark_or_exit", return_value=FAKE_BENCHMARK)


def _patch_models():
    return patch("mothertoken.cli.app._load_models_config", return_value=FAKE_MODELS_CONFIG)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_valid_language():
    with _patch_benchmark():
        result = runner.invoke(app, ["compare", "--language", "arb_Arab"])
    assert result.exit_code == 0, result.output
    assert "arb_Arab" in result.output
    assert "GPT-4o" in result.output
    assert "Qwen 2.5" in result.output


def test_compare_invalid_language():
    with _patch_benchmark():
        result = runner.invoke(app, ["compare", "--language", "xyz_Fake"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "not found" in (result.stderr or "").lower()


def test_compare_shows_best_model_tip():
    with _patch_benchmark():
        result = runner.invoke(app, ["compare", "--language", "arb_Arab"])
    assert result.exit_code == 0
    assert "GPT-4o" in result.output  # highest chars/token = 3.132 wins


def test_compare_missing_language_arg():
    with _patch_benchmark():
        result = runner.invoke(app, ["compare"])
    # Should error because --language is required
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# token
# ---------------------------------------------------------------------------


def test_token_tiktoken_model():
    with _patch_models(), patch("mothertoken.core.tokenizers.tokenize_sentences", return_value=[3]):
        result = runner.invoke(app, ["token", "--text", "ChatGPT", "--model", "gpt-4o"])
    assert result.exit_code == 0, result.output
    assert "3" in result.output
    assert "ChatGPT" in result.output


def test_token_api_model_graceful():
    """API-only models should print a helpful message and exit 0."""
    with _patch_models():
        result = runner.invoke(app, ["token", "--text", "hello", "--model", "claude-sonnet"])
    assert result.exit_code == 0
    assert "closed API" in result.output or "local" in result.output.lower()


def test_token_unknown_model():
    with _patch_models():
        result = runner.invoke(app, ["token", "--text", "hello", "--model", "nonexistent-model"])
    assert result.exit_code == 1


def test_token_tokenizer_error():
    """If the tokenizer raises, exit with code 1."""
    with _patch_models(), patch("mothertoken.core.tokenizers.tokenize_sentences", side_effect=Exception("load failed")):
        result = runner.invoke(app, ["token", "--text", "hello", "--model", "gpt-4o"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_text_input():
    mock_counts = [5]  # 5 tokens for "Hello world"
    with (
        _patch_models(),
        _patch_benchmark(),
        patch("mothertoken.core.tokenizers.tokenize_sentences", return_value=mock_counts),
    ):
        result = runner.invoke(app, ["analyze", "--text", "Hello world", "--languages", "eng_Latn"])
    assert result.exit_code == 0, result.output
    assert "GPT-4o" in result.output or "Qwen 2.5" in result.output


def test_analyze_file_input(tmp_path):
    test_file = tmp_path / "prompt.txt"
    test_file.write_text("This is a test prompt.", encoding="utf-8")
    with _patch_models(), _patch_benchmark(), patch("mothertoken.core.tokenizers.tokenize_sentences", return_value=[6]):
        result = runner.invoke(app, ["analyze", "--file", str(test_file), "--languages", "eng_Latn"])
    assert result.exit_code == 0, result.output
    assert "This is a test prompt" in result.output


def test_analyze_missing_input():
    """Must provide either --text or --file."""
    with _patch_models(), _patch_benchmark():
        result = runner.invoke(app, ["analyze", "--languages", "eng_Latn"])
    assert result.exit_code == 1


def test_analyze_nonexistent_file():
    with _patch_models(), _patch_benchmark():
        result = runner.invoke(app, ["analyze", "--file", "/nonexistent/path.txt", "--languages", "eng_Latn"])
    assert result.exit_code == 1


def test_analyze_local_mode_excludes_api_models():
    """In local mode, API models should NOT be called."""
    call_log = []

    def fake_tokenize(model_cfg, sentences, cache, dry_run):
        call_log.append(model_cfg["id"])
        return [3]

    with (
        _patch_models(),
        _patch_benchmark(),
        patch("mothertoken.core.tokenizers.tokenize_sentences", side_effect=fake_tokenize),
    ):
        result = runner.invoke(app, ["analyze", "--text", "hello", "--languages", "eng_Latn", "--mode", "local"])

    assert result.exit_code == 0
    assert "claude-sonnet" not in call_log


def test_analyze_tokenizer_error_shown_gracefully():
    """A tokenizer error should be shown in output, not crash the whole command."""

    def fake_tokenize(model_cfg, sentences, cache, dry_run):
        if model_cfg["id"] == "qwen2.5":
            raise Exception("HuggingFace load error")
        return [4]

    with (
        _patch_models(),
        _patch_benchmark(),
        patch("mothertoken.core.tokenizers.tokenize_sentences", side_effect=fake_tokenize),
    ):
        result = runner.invoke(app, ["analyze", "--text", "hello", "--languages", "eng_Latn", "--mode", "local"])
    assert result.exit_code == 0
    assert "error" in result.output.lower() or "HuggingFace" in result.output
