import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from mothertoken.benchmark.runner import compute_metrics, run_benchmark, save_benchmark

# ---------------------------------------------------------------------------
# compute_metrics Tests
# ---------------------------------------------------------------------------

def test_compute_metrics_basic():
    sentences = ["Hello world", "Test sentence"]
    token_counts = [2, 2]
    english_chars_per_token = 5.0
    input_cost_per_token = 0.00001
    
    metrics = compute_metrics(sentences, token_counts, english_chars_per_token, input_cost_per_token)
    
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
    
    # cost_per_1m_chars: (1,000,000 / 6.0) * 0.00001 = 166,666.666 * 0.00001 = 1.66667
    assert metrics["cost_per_1m_chars"] == pytest.approx(1.66667, abs=1e-5)
    
    # fair_cost: (1,000,000 / 5.0) * 0.00001 = 200,000 * 0.00001 = 2.0
    # fair_cost_delta: 1.66667 - 2.0 = -0.33333
    assert metrics["fair_cost_delta"] == pytest.approx(-0.33333, abs=1e-5)

def test_compute_metrics_zero_tokens():
    sentences = [""]
    token_counts = [0]
    metrics = compute_metrics(sentences, token_counts, 5.0, 0.00001)
    
    assert metrics["total_tokens"] == 0
    assert metrics["chars_per_token"] == 0.0
    assert metrics["rtc"] == 0.0
    assert metrics["cost_per_1m_chars"] == 0.0

def test_compute_metrics_zero_words():
    sentences = ["   "]
    token_counts = [1]
    metrics = compute_metrics(sentences, token_counts, 5.0, 0.00001)
    
    assert metrics["total_words"] == 0
    assert metrics["fertility"] == 0.0

# ---------------------------------------------------------------------------
# run_benchmark Tests
# ---------------------------------------------------------------------------

@patch("mothertoken.benchmark.runner.load_flores_sentences")
@patch("mothertoken.benchmark.runner.pricing.fetch_pricing_data")
@patch("mothertoken.benchmark.runner.pricing.get_model_input_cost")
@patch("mothertoken.benchmark.runner.tokenizers.tokenize_sentences")
def test_run_benchmark_flow(mock_tokenize, mock_get_cost, mock_fetch_pricing, mock_load_flores):
    # Setup mocks
    mock_load_flores.side_effect = lambda lang, split="dev": [f"Sentence in {lang}"] * 2
    mock_fetch_pricing.return_value = {"dummy": "data"}
    mock_get_cost.return_value = 0.00002
    mock_tokenize.return_value = [3, 3] # 6 tokens total per language/model
    
    languages = ["eng_Latn", "tha_Thai"]
    model_ids = ["gpt-4o"]
    
    with patch("mothertoken.benchmark.runner.MODELS", [{"id": "gpt-4o", "type": "tiktoken", "ref": "o200k_base"}]):
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
def test_run_benchmark_model_failure(mock_tokenize, mock_load_flores):
    mock_load_flores.return_value = ["test"]
    mock_tokenize.side_effect = Exception("Tokenizer load fail")
    
    languages = ["eng_Latn"]
    model_ids = ["fail-model"]
    
    with patch("mothertoken.benchmark.runner.MODELS", [{"id": "fail-model", "type": "tiktoken", "ref": "invalid"}]):
        results, errors = run_benchmark(languages, model_ids, dry_run=False)
        
        assert "fail-model" in errors
        assert errors["fail-model"] == "Tokenizer load fail"
        assert results["eng_Latn"] == {} # No models were successful

# ---------------------------------------------------------------------------
# save_benchmark Tests
# ---------------------------------------------------------------------------

def test_save_benchmark(tmp_path):
    results = {"eng_Latn": {"gpt-4o": {"rtc": 1.0}}}
    errors = {}
    output_path = tmp_path / "benchmark.json"
    model_ids = ["gpt-4o"]
    
    with patch("mothertoken.benchmark.runner.MODELS", [{"id": "gpt-4o", "type": "tiktoken"}]):
        save_benchmark(results, errors, output_path, model_ids)
        
        assert output_path.exists()
        with open(output_path, "r") as f:
            data = json.load(f)
            assert data["metrics"] == results
            assert data["baseline_language"] == "eng_Latn"
            assert len(data["models"]) == 1
