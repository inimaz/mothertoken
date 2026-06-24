from unittest.mock import patch

import pytest

from mothertoken.core.tokenizers import (
    FALLBACK_MAX_POSITION_EMBEDDINGS,
    encode_hf,
    has_unknown_auto_config,
    load_hf_tokenizer,
    load_hf_tokenizer_file,
    tokenize_sentences,
)


class ChatTemplateTokenizer:
    chat_template = "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}"

    def __init__(self):
        self.applied_messages = []
        self.encoded = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.applied_messages.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        assert not tokenize
        return "<|user|>\nHello world<|assistant|>"

    def encode(self, text, *, add_special_tokens):
        self.encoded.append({"text": text, "add_special_tokens": add_special_tokens})
        return [101, 102, 103, 104]


def test_tokenizer_caching():
    cache = {}

    model_a = {"id": "model_a", "type": "tiktoken", "ref": "cl100k_base"}

    model_b = {
        "id": "model_b",
        "type": "tiktoken",
        "ref": "cl100k_base",  # Same tokenizer reference
    }

    sentences = ["Hello world"]

    with (
        patch("mothertoken.core.tokenizers.load_tiktoken_tokenizer") as mock_load,
        patch("mothertoken.core.tokenizers.tokenize_tiktoken", return_value=[2]) as mock_tokenize,
    ):
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

    with (
        patch("mothertoken.core.tokenizers.load_hf_tokenizer") as mock_load,
        patch("mothertoken.core.tokenizers.tokenize_hf", return_value=[2]) as mock_tokenize,
    ):
        counts1 = tokenize_sentences(model_a, sentences, cache, dry_run=False)
        counts2 = tokenize_sentences(model_b, sentences, cache, dry_run=False)

        assert mock_load.call_count == 1
        assert mock_tokenize.call_count == 1
        assert counts1 == counts2 == [2]


def test_huggingface_tokenize_sentences_uses_raw_text_even_with_chat_template():
    tokenizer = ChatTemplateTokenizer()

    counts = tokenize_sentences(
        {"id": "chat-model", "type": "huggingface", "ref": "chat/ref"},
        ["Hello world"],
        {("tokenizer", "huggingface", "chat/ref"): tokenizer},
        dry_run=False,
    )

    assert counts == [4]
    assert tokenizer.applied_messages == []
    assert tokenizer.encoded == [{"text": "Hello world", "add_special_tokens": False}]


def test_huggingface_raw_tokenizer_uses_plain_encoding_without_special_tokens():
    class RawTokenizer:
        def __init__(self):
            self.encoded = []

        def encode(self, text, *, add_special_tokens):
            self.encoded.append({"text": text, "add_special_tokens": add_special_tokens})
            return [1, 2]

    tokenizer = RawTokenizer()

    assert encode_hf(tokenizer, "Hello world") == [1, 2]
    assert tokenizer.encoded == [{"text": "Hello world", "add_special_tokens": False}]


def test_unknown_auto_config_loads_from_tokenizer_file_without_auto_tokenizer(tmp_path):
    tokenizer = object()
    config_json = tmp_path / "config.json"
    config_json.write_text('{"model_type":"future_model_type"}', encoding="utf-8")

    with patch("huggingface_hub.hf_hub_download", return_value=str(config_json)):
        assert has_unknown_auto_config("org/future-model")

    with (
        patch("mothertoken.core.tokenizers.load_hf_tokenizer_file", return_value=tokenizer) as mock_file_loader,
        patch("mothertoken.core.tokenizers.has_unknown_auto_config", return_value=True),
        patch("transformers.AutoTokenizer.from_pretrained") as mock_auto_tokenizer,
    ):
        assert load_hf_tokenizer("org/future-model") is tokenizer

    mock_file_loader.assert_called_once_with("org/future-model")
    mock_auto_tokenizer.assert_not_called()


def test_load_hf_tokenizer_file_uses_tokenizer_json_and_config(tmp_path):
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_config_json = tmp_path / "tokenizer_config.json"
    tokenizer_json.write_text("{}", encoding="utf-8")
    tokenizer_config_json.write_text(
        (
            '{"bos_token":{"content":"<bos>"},"eos_token":{"content":"<eos>"},'
            '"pad_token":{"content":"<pad>"},"unk_token":null,'
            '"model_max_length":131072,"chat_template":"{{ messages }}",'
            '"clean_up_tokenization_spaces":false}'
        ),
        encoding="utf-8",
    )

    def fake_hf_hub_download(ref, filename):
        return str(tokenizer_json if filename == "tokenizer.json" else tokenizer_config_json)

    with (
        patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download) as mock_download,
        patch("transformers.PreTrainedTokenizerFast", return_value=object()) as mock_fast,
    ):
        load_hf_tokenizer_file("deepseek-ai/DeepSeek-V3.2")

    assert [call.args for call in mock_download.call_args_list] == [
        ("deepseek-ai/DeepSeek-V3.2", "tokenizer.json"),
        ("deepseek-ai/DeepSeek-V3.2", "tokenizer_config.json"),
    ]
    assert mock_fast.call_args.kwargs == {
        "tokenizer_file": str(tokenizer_json),
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "chat_template": "{{ messages }}",
        "model_max_length": 131072,
        "clean_up_tokenization_spaces": False,
    }


def test_load_hf_tokenizer_retries_missing_max_position_embeddings():
    tokenizer = object()
    config = type("DeepseekV32Config", (), {})()

    with (
        patch("mothertoken.core.tokenizers.has_unknown_auto_config", return_value=False),
        patch("mothertoken.core.tokenizers.load_hf_tokenizer_file", side_effect=FileNotFoundError),
        patch("transformers.AutoConfig.from_pretrained", return_value=config) as mock_config_from_pretrained,
        patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained,
    ):
        mock_from_pretrained.side_effect = [
            AttributeError("'PreTrainedConfig' object has no attribute 'max_position_embeddings'"),
            tokenizer,
        ]

        assert load_hf_tokenizer("org/model-with-missing-position-config") is tokenizer

    assert mock_from_pretrained.call_count == 2
    assert mock_from_pretrained.call_args_list[0].args == ("org/model-with-missing-position-config",)
    mock_config_from_pretrained.assert_called_once_with("org/model-with-missing-position-config")
    retry_kwargs = mock_from_pretrained.call_args_list[1].kwargs
    assert mock_from_pretrained.call_args_list[1].args == ("org/model-with-missing-position-config",)
    assert retry_kwargs["config"] is config
    assert config.max_position_embeddings == FALLBACK_MAX_POSITION_EMBEDDINGS


def test_load_hf_tokenizer_reraises_unrelated_attribute_error():
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.side_effect = AttributeError("unrelated")

        with pytest.raises(AttributeError, match="unrelated"):
            load_hf_tokenizer("broken/model")

    assert mock_from_pretrained.call_count == 1


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

    with (
        patch("mothertoken.core.tokenizers.load_tiktoken_tokenizer") as mock_load,
        patch("mothertoken.core.tokenizers.tokenize_tiktoken") as mock_tokenize,
    ):
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

    with (
        patch("mothertoken.core.tokenizers.load_tiktoken_tokenizer") as mock_load,
        patch("mothertoken.core.tokenizers.tokenize_tiktoken") as mock_tokenize,
    ):
        mock_tokenize.side_effect = [[3], [4]]

        tokenize_sentences(model_1, sentences, cache, dry_run=False)
        tokenize_sentences(model_2, sentences, cache, dry_run=False)

        assert mock_load.call_count == 2
        assert mock_tokenize.call_count == 2
