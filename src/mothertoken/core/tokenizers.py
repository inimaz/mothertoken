"""
mothertoken — core/tokenizers.py

Handles tokenization across different models and providers.
"""

import json
import logging

from mothertoken.core.registry import ModelType

log = logging.getLogger("mothertoken")

FALLBACK_MAX_POSITION_EMBEDDINGS = 131_072


def load_tiktoken_tokenizer(ref: str):
    import tiktoken

    return tiktoken.get_encoding(ref)


def tokenize_tiktoken(tokenizer, sentences: list[str]) -> list[int]:
    return [len(tokenizer.encode(s)) for s in sentences]


def _tokenizer_config_value(config: dict, key: str):
    value = config.get(key)
    if isinstance(value, dict):
        return value.get("content")
    return value


def load_hf_tokenizer_file(ref: str):
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    tokenizer_file = hf_hub_download(ref, "tokenizer.json")
    tokenizer_config_file = hf_hub_download(ref, "tokenizer_config.json")
    tokenizer_config = json.loads(open(tokenizer_config_file, encoding="utf-8").read())

    kwargs = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        value = _tokenizer_config_value(tokenizer_config, key)
        if value is not None:
            kwargs[key] = value
    for key in ("chat_template", "model_max_length", "clean_up_tokenization_spaces"):
        if key in tokenizer_config:
            kwargs[key] = tokenizer_config[key]

    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, **kwargs)


def has_unknown_auto_config(ref: str) -> bool:
    from huggingface_hub import hf_hub_download
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    try:
        config_file = hf_hub_download(ref, "config.json")
        with open(config_file, encoding="utf-8") as f:
            model_config = json.load(f)
    except Exception:
        return False

    model_type = model_config.get("model_type")
    return bool(model_type and model_type not in CONFIG_MAPPING_NAMES)


def load_hf_tokenizer(ref: str):
    from transformers import AutoConfig, AutoTokenizer

    log.info(f"Loading HuggingFace tokenizer: {ref}")
    if has_unknown_auto_config(ref):
        return load_hf_tokenizer_file(ref)

    try:
        return AutoTokenizer.from_pretrained(ref)
    except AttributeError as exc:
        if "max_position_embeddings" not in str(exc):
            raise
        try:
            return load_hf_tokenizer_file(ref)
        except Exception:
            pass
        log.warning(
            "Retrying HuggingFace tokenizer load for %s with its model config patched to include "
            "max_position_embeddings.",
            ref,
        )
        config = AutoConfig.from_pretrained(ref)
        if not hasattr(config, "max_position_embeddings"):
            config.max_position_embeddings = FALLBACK_MAX_POSITION_EMBEDDINGS
        return AutoTokenizer.from_pretrained(
            ref,
            config=config,
        )


def encode_hf(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def tokenize_hf(tokenizer, sentences: list[str]) -> list[int]:
    return [len(encode_hf(tokenizer, sentence)) for sentence in sentences]


def make_counts_cache_key(model_type: str, ref: str, sentences: list[str]) -> tuple:
    return ("counts", model_type, ref, tuple(sentences))


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
        response = client.messages.count_tokens(model=model_ref, messages=[{"role": "user", "content": sentence}])
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

    counts_cache_key = make_counts_cache_key(mtype, ref, sentences)

    if counts_cache_key in cache:
        log.info(f"Using cached token counts for tokenizer {ref}")
        return cache[counts_cache_key]

    tok_cache_key = ("tokenizer", mtype, ref)

    if mtype == ModelType.TIKTOKEN:
        if tok_cache_key not in cache:
            cache[tok_cache_key] = load_tiktoken_tokenizer(ref)
        counts = tokenize_tiktoken(cache[tok_cache_key], sentences)

    elif mtype == ModelType.HUGGINGFACE:
        if tok_cache_key not in cache:
            cache[tok_cache_key] = load_hf_tokenizer(ref)
        counts = tokenize_hf(cache[tok_cache_key], sentences)

    elif mtype == ModelType.ANTHROPIC_API:
        counts = tokenize_anthropic_api(ref, sentences)

    elif mtype == ModelType.GOOGLE_API:
        counts = tokenize_google_api(ref, sentences)

    else:
        raise ValueError(f"Unknown tokenizer type: {mtype}")

    cache[counts_cache_key] = counts
    return counts
