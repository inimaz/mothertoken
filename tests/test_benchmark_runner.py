import json
from unittest.mock import patch

import pytest

from mothertoken.benchmark.runner import (
    _get_models,
    compute_metrics,
    load_config,
    resolve_benchmark_models,
    run_benchmark,
    save_benchmark,
)

# ---------------------------------------------------------------------------
# compute_metrics Tests
# ---------------------------------------------------------------------------


def test_compute_metrics_basic():
    sentences = ["Hello world", "Test sentence"]
    token_counts = [2, 2]
    english_chars_per_token = 5.0

    metrics = compute_metrics(sentences, token_counts, english_chars_per_token)

    # Total chars: len("Hello world") + len("Test sentence") = 11 + 13 = 24
    assert metrics["total_chars"] == 24
    assert metrics["total_tokens"] == 4
    # Total words: "Hello world" (2) + "Test sentence" (2) = 4
    assert metrics["total_words"] == 4

    # chars_per_token: 24 / 4 = 6.0
    assert metrics["chars_per_token"] == 6.0

    # fertility: 4 / 4 = 1.0
    assert metrics["fertility"] == 1.0

    # rtc: 5.0 / 6.0 = 0.833
    assert metrics["rtc"] == pytest.approx(0.833, abs=1e-3)


def test_compute_metrics_zero_tokens():
    sentences = [""]
    token_counts = [0]
    metrics = compute_metrics(sentences, token_counts, 5.0)

    assert metrics["total_tokens"] == 0
    assert metrics["chars_per_token"] == 0.0
    assert metrics["rtc"] == 0.0


def test_compute_metrics_zero_words():
    sentences = ["   "]
    token_counts = [1]
    metrics = compute_metrics(sentences, token_counts, 5.0)

    assert metrics["total_words"] == 0
    assert metrics["fertility"] == 0.0


# ---------------------------------------------------------------------------
# run_benchmark Tests
# ---------------------------------------------------------------------------


def test_load_config_uses_tokenizer_registry_service():
    fake_config = {"tokenizers": [{"id": "gpt-4o"}]}

    with patch("mothertoken.benchmark.runner.TokenizerRegistryService") as mock_service_class:
        mock_service = mock_service_class.return_value
        mock_service.path_info.return_value = {"path": "/tmp/tokenizers.yaml", "exists": True}
        mock_service.exists.return_value = True
        mock_service.load.return_value = fake_config

        assert load_config() == fake_config
        mock_service_class.assert_called_once_with()
        mock_service.path_info.assert_called_once_with()
        mock_service.exists.assert_called_once_with()
        mock_service.load.assert_called_once_with()


def test_load_config_reports_registry_path_when_missing():
    with patch("mothertoken.benchmark.runner.TokenizerRegistryService") as mock_service_class:
        mock_service = mock_service_class.return_value
        mock_service.path_info.return_value = {"path": "/tmp/missing-tokenizers.yaml", "exists": False}
        mock_service.exists.return_value = False

        with pytest.raises(FileNotFoundError, match="/tmp/missing-tokenizers.yaml"):
            load_config()


def test_get_models_uses_tokenizer_registry_service_list():
    fake_config = {"tokenizers": [{"id": "gpt-4o"}]}
    fake_entries = [{"id": "gpt-4o"}]

    with (
        patch("mothertoken.benchmark.runner._get_config", return_value=fake_config),
        patch("mothertoken.benchmark.runner.TokenizerRegistryService") as mock_service_class,
    ):
        mock_service = mock_service_class.return_value
        mock_service.list.return_value = fake_entries

        assert _get_models() == fake_entries
        mock_service_class.assert_called_once_with()
        mock_service.list.assert_called_once_with(fake_config)


def test_resolve_benchmark_models_accepts_aliases_and_huggingface_refs():
    with patch(
        "mothertoken.benchmark.runner._get_models",
        return_value=[{"id": "gpt-4o", "type": "tiktoken", "ref": "o200k_base"}],
    ):
        selected = resolve_benchmark_models(["gpt-4o", "Qwen/Qwen3-0.6B"])

    assert selected[0]["id"] == "gpt-4o"
    assert selected[1]["id"] == "Qwen/Qwen3-0.6B"
    assert selected[1]["type"] == "huggingface"
    assert selected[1]["ref"] == "Qwen/Qwen3-0.6B"


def test_resolve_benchmark_models_rejects_unknown_one_word_ids():
    with patch("mothertoken.benchmark.runner._get_models", return_value=[{"id": "gpt-4o"}]):
        with pytest.raises(ValueError, match="Unknown model/tokenizer"):
            resolve_benchmark_models(["typo-model"])


@patch("mothertoken.benchmark.runner.load_flores_sentences")
@patch("mothertoken.benchmark.runner.tokenizers.tokenize_sentences")
def test_run_benchmark_flow(mock_tokenize, mock_load_flores):
    # Setup mocks
    mock_load_flores.side_effect = lambda lang, split="dev": [f"Sentence in {lang}"] * 2
    mock_tokenize.return_value = [3, 3]  # 6 tokens total per language/model

    languages = ["eng_Latn", "tha_Thai"]
    model_ids = ["gpt-4o"]

    with patch(
        "mothertoken.benchmark.runner._get_models",
        return_value=[{"id": "gpt-4o", "type": "tiktoken", "ref": "o200k_base"}],
    ):
        results, errors = run_benchmark(languages, model_ids, dry_run=False)

        # Verify calls
        assert "eng_Latn" in results
        assert "tha_Thai" in results
        assert "gpt-4o" in results["eng_Latn"]
        assert "gpt-4o" in results["tha_Thai"]

        # English baseline should have been computed
        # total_chars = len("Sentence in eng_Latn") * 2 = 20 * 2 = 40
        # total_tokens = 6
        # eng_cpt = 40 / 6 = 6.666...

        # Thai results
        # total_chars = len("Sentence in tha_Thai") * 2 = 20 * 2 = 40
        # total_tokens = 6
        # tha_cpt = 40 / 6 = 6.666...
        # rtc = 6.666 / 6.666 = 1.0

        assert results["tha_Thai"]["gpt-4o"]["rtc"] == 1.0
        assert not errors


@patch("mothertoken.benchmark.runner.load_flores_sentences")
@patch("mothertoken.benchmark.runner.tokenizers.tokenize_sentences")
def test_run_benchmark_accepts_direct_huggingface_ref(mock_tokenize, mock_load_flores):
    call_log = []

    def fake_tokenize(model_cfg, sentences, cache, dry_run):
        call_log.append(model_cfg)
        return [3, 3]

    mock_load_flores.return_value = ["hello", "world"]
    mock_tokenize.side_effect = fake_tokenize

    with patch("mothertoken.benchmark.runner._get_models", return_value=[]):
        results, errors = run_benchmark(["eng_Latn"], ["Qwen/Qwen3-0.6B"], dry_run=False)

    assert not errors
    assert "Qwen/Qwen3-0.6B" in results["eng_Latn"]
    assert call_log[0]["type"] == "huggingface"
    assert call_log[0]["ref"] == "Qwen/Qwen3-0.6B"


@patch("mothertoken.benchmark.runner.load_flores_sentences")
@patch("mothertoken.benchmark.runner.tokenizers.tokenize_sentences")
def test_run_benchmark_model_failure(mock_tokenize, mock_load_flores):
    mock_load_flores.return_value = ["test"]
    mock_tokenize.side_effect = Exception("Tokenizer load fail")

    languages = ["eng_Latn"]
    model_ids = ["fail-model"]

    with patch(
        "mothertoken.benchmark.runner._get_models",
        return_value=[{"id": "fail-model", "type": "tiktoken", "ref": "invalid"}],
    ):
        results, errors = run_benchmark(languages, model_ids, dry_run=False)

        assert "fail-model" in errors
        assert errors["fail-model"] == "Tokenizer load fail"
        assert results["eng_Latn"] == {}  # No models were successful


# ---------------------------------------------------------------------------
# save_benchmark Tests
# ---------------------------------------------------------------------------


def test_save_benchmark(tmp_path):
    results = {"eng_Latn": {"gpt-4o": {"rtc": 1.0}}}
    errors = {}
    output_path = tmp_path / "benchmark.json"
    model_ids = ["gpt-4o"]

    with patch("mothertoken.benchmark.runner._get_models", return_value=[{"id": "gpt-4o", "type": "tiktoken"}]):
        save_benchmark(results, errors, output_path, model_ids)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
            assert data["metrics"] == results
            assert data["baseline_language"] == "eng_Latn"
            assert len(data["models"]) == 1


def test_save_benchmark_includes_direct_huggingface_ref(tmp_path):
    results = {"eng_Latn": {"Qwen/Qwen3-0.6B": {"rtc": 1.0}}}
    errors = {}
    output_path = tmp_path / "benchmark.json"
    model_ids = ["Qwen/Qwen3-0.6B"]

    with patch("mothertoken.benchmark.runner._get_models", return_value=[]):
        save_benchmark(results, errors, output_path, model_ids)

    with open(output_path) as f:
        data = json.load(f)

    assert data["models"][0]["id"] == "Qwen/Qwen3-0.6B"
    assert data["models"][0]["type"] == "huggingface"
    assert data["models"][0]["ref"] == "Qwen/Qwen3-0.6B"
