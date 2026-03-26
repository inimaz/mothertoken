import pytest
from unittest.mock import patch
from mothertoken.core.tokenizers import tokenize_sentences

def test_tokenizer_caching():
    cache = {}
    
    model_a = {
        "id": "model_a",
        "type": "tiktoken",
        "ref": "cl100k_base"
    }

    model_b = {
        "id": "model_b",
        "type": "tiktoken",
        "ref": "cl100k_base"  # Same tokenizer reference
    }

    sentences = ["Hello world"]

    with patch("mothertoken.core.tokenizers.load_tiktoken_tokenizer") as mock_load, \
         patch("mothertoken.core.tokenizers.tokenize_tiktoken", return_value=[2]) as mock_tokenize:

        # First call with model_a
        counts1 = tokenize_sentences(model_a, sentences, cache, dry_run=False)
        assert counts1 == [2]
        mock_load.assert_called_once_with("cl100k_base")
        assert mock_tokenize.call_count == 1
        
        # Second call with model_b (same ref)
        counts2 = tokenize_sentences(model_b, sentences, cache, dry_run=False)
        assert counts2 == [2]
        
        # Under the current implementation, model_b will cause load_tiktoken_tokenizer to be called again 
        # (or at least loaded again in the cache under its own ID)
        # However, the GOAL is that load_tiktoken_tokenizer is only called ONCE for the same ref.
        # And tokenize_tiktoken is ONLY called ONCE for the same (ref, sentences) combination.
        
        # With the new optimization, both assertions below should pass:
        assert mock_load.call_count == 1, "Tokenizer should only be loaded once for the same ref"
        assert mock_tokenize.call_count == 1, "Token counting should be cached for the same text and ref"

def test_dry_run_ignores_cache():
    cache = {}
    model = {"id": "gpt-4o", "type": "tiktoken", "ref": "o200k_base"}
    sentences = ["Test"]

    with patch("mothertoken.core.tokenizers.load_tiktoken_tokenizer") as mock_load:
        counts = tokenize_sentences(model, sentences, cache, dry_run=True)
        assert counts == [5]
        mock_load.assert_not_called()

def test_huggingface_caching():
    cache = {}
    
    model_a = {"id": "llama-a", "type": "huggingface", "ref": "meta-llama/Meta-Llama-3-8B"}
    model_b = {"id": "llama-b", "type": "huggingface", "ref": "meta-llama/Meta-Llama-3-8B"}

    sentences = ["Hello world"]

    with patch("mothertoken.core.tokenizers.load_hf_tokenizer") as mock_load, \
         patch("mothertoken.core.tokenizers.tokenize_hf", return_value=[2]) as mock_tokenize:

        counts1 = tokenize_sentences(model_a, sentences, cache, dry_run=False)
        counts2 = tokenize_sentences(model_b, sentences, cache, dry_run=False)

        assert mock_load.call_count == 1
        assert mock_tokenize.call_count == 1
        assert counts1 == counts2 == [2]

def test_api_caching():
    cache = {}
    
    model_a = {"id": "claude-a", "type": "anthropic_api", "ref": "claude-sonnet"}
    model_b = {"id": "claude-b", "type": "anthropic_api", "ref": "claude-sonnet"}

    sentences = ["API test"]

    with patch("mothertoken.core.tokenizers.tokenize_anthropic_api", return_value=[4]) as mock_api:

        counts1 = tokenize_sentences(model_a, sentences, cache, dry_run=False)
        counts2 = tokenize_sentences(model_b, sentences, cache, dry_run=False)

        assert mock_api.call_count == 1
        assert counts1 == counts2 == [4]

def test_different_text_triggers_new_counts():
    cache = {}
    
    model = {"id": "model_a", "type": "tiktoken", "ref": "cl100k_base"}

    sentences1 = ["First text"]
    sentences2 = ["Second text"]

    with patch("mothertoken.core.tokenizers.load_tiktoken_tokenizer") as mock_load, \
         patch("mothertoken.core.tokenizers.tokenize_tiktoken") as mock_tokenize:
        
        mock_tokenize.side_effect = [[2], [3]]

        counts1 = tokenize_sentences(model, sentences1, cache, dry_run=False)
        counts2 = tokenize_sentences(model, sentences2, cache, dry_run=False)

        # Tokenizer is loaded once
        assert mock_load.call_count == 1
        
        # Tokenizing happens twice because text is different
        assert mock_tokenize.call_count == 2
        assert counts1 == [2]
        assert counts2 == [3]

def test_different_ref_triggers_new_counts():
    cache = {}
    
    model_1 = {"id": "gpt-4", "type": "tiktoken", "ref": "cl100k_base"}
    model_2 = {"id": "gpt-3", "type": "tiktoken", "ref": "p50k_base"}

    sentences = ["Exact same text"]

    with patch("mothertoken.core.tokenizers.load_tiktoken_tokenizer") as mock_load, \
         patch("mothertoken.core.tokenizers.tokenize_tiktoken") as mock_tokenize:

        mock_tokenize.side_effect = [[3], [4]]

        tokenize_sentences(model_1, sentences, cache, dry_run=False)
        tokenize_sentences(model_2, sentences, cache, dry_run=False)

        assert mock_load.call_count == 2
        assert mock_tokenize.call_count == 2
