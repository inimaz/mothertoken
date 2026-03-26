"""
mothertoken — core/tokenizers.py

Handles tokenization across different models and providers.
"""

import logging

log = logging.getLogger("mothertoken")

def load_tiktoken_tokenizer(ref: str):
    import tiktoken
    return tiktoken.get_encoding(ref)

def tokenize_tiktoken(tokenizer, sentences: list[str]) -> list[int]:
    return [len(tokenizer.encode(s)) for s in sentences]


def load_hf_tokenizer(ref: str):
    from transformers import AutoTokenizer
    log.info(f"Loading HuggingFace tokenizer: {ref}")
    return AutoTokenizer.from_pretrained(ref)

def tokenize_hf(tokenizer, sentences: list[str]) -> list[int]:
    return [len(tokenizer.encode(s)) for s in sentences]


def tokenize_anthropic_api(model_ref: str, sentences: list[str]) -> list[int]:
    """Count tokens via Anthropic count_tokens endpoint. Results are cached."""
    import os
    import anthropic
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is missing.")
        
    client = anthropic.Anthropic(api_key=api_key)
    counts = []
    for sentence in sentences:
        response = client.messages.count_tokens(
            model=model_ref,
            messages=[{"role": "user", "content": sentence}]
        )
        counts.append(response.input_tokens)
    return counts


def tokenize_google_api(model_ref: str, sentences: list[str]) -> list[int]:
    """Count tokens via Google Gemini count_tokens endpoint."""
    import os
    import google.generativeai as genai
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is missing.")
        
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(model_ref)
    counts = []
    for sentence in sentences:
        response = model.count_tokens(sentence)
        counts.append(response.total_tokens)
    return counts


def tokenize_sentences(model: dict, sentences: list[str], cache: dict, dry_run: bool) -> list[int]:
    """Dispatch tokenization based on model type, with tokenizer caching."""
    mtype = model["type"]
    ref = model["ref"]

    if dry_run:
        return [5] * len(sentences)  # dummy counts

    sentences_hash = hash(tuple(sentences))
    counts_cache_key = ("counts", mtype, ref, sentences_hash)
    
    if counts_cache_key in cache:
        log.info(f"Using cached token counts for tokenizer {ref}")
        return cache[counts_cache_key]

    tok_cache_key = ("tokenizer", mtype, ref)

    if mtype == "tiktoken":
        if tok_cache_key not in cache:
            cache[tok_cache_key] = load_tiktoken_tokenizer(ref)
        counts = tokenize_tiktoken(cache[tok_cache_key], sentences)

    elif mtype == "huggingface":
        if tok_cache_key not in cache:
            cache[tok_cache_key] = load_hf_tokenizer(ref)
        counts = tokenize_hf(cache[tok_cache_key], sentences)

    elif mtype == "anthropic_api":
        counts = tokenize_anthropic_api(ref, sentences)

    elif mtype == "google_api":
        counts = tokenize_google_api(ref, sentences)

    else:
        raise ValueError(f"Unknown tokenizer type: {mtype}")

    cache[counts_cache_key] = counts
    return counts
