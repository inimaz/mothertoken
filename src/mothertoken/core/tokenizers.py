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
    mid = model["id"]

    if dry_run:
        return [5] * len(sentences)  # dummy counts

    if model["type"] == "tiktoken":
        if mid not in cache:
            cache[mid] = load_tiktoken_tokenizer(model["ref"])
        return tokenize_tiktoken(cache[mid], sentences)

    elif model["type"] == "huggingface":
        if mid not in cache:
            cache[mid] = load_hf_tokenizer(model["ref"])
        return tokenize_hf(cache[mid], sentences)

    elif model["type"] == "anthropic_api":
        return tokenize_anthropic_api(model["ref"], sentences)

    elif model["type"] == "google_api":
        return tokenize_google_api(model["ref"], sentences)

    else:
        raise ValueError(f"Unknown tokenizer type: {model['type']}")
