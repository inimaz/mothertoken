from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "sync_tiktoken_tokenizers.py"
    spec = importlib.util.spec_from_file_location("sync_tiktoken_tokenizers", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sync_config_updates_existing_openai_tiktoken_entry(monkeypatch):
    module = _load_script_module()
    monkeypatch.setattr(module.tiktoken, "list_encoding_names", lambda: ["o200k_base"])
    monkeypatch.setattr(module.tiktoken_model, "MODEL_TO_ENCODING", {"gpt-5": "o200k_base"})
    monkeypatch.setattr(
        module.tiktoken_model,
        "MODEL_PREFIX_TO_ENCODING",
        {
            "ft:gpt-5": "o200k_base",
            "gpt-4.5-": "o200k_base",
            "gpt-5-": "o200k_base",
        },
    )

    config = {
        "tokenizers": [
            {
                "id": "gpt-4o",
                "name": "OpenAI o200k_base",
                "provider": "openai",
                "type": "tiktoken",
                "ref": "o200k_base",
                "access": "local",
                "tokenizer_source": "tiktoken",
                "verification_method": "local_fixture_count",
                "used_by_examples": ["GPT-4o", "GPT-5-*", "ft:gpt-5*", "ChatGPT-4o"],
                "api_key_env": None,
            }
        ]
    }

    synced = module.sync_config(config)

    assert synced["tokenizers"][0]["id"] == "gpt-4o"
    assert synced["tokenizers"][0]["used_by_examples"] == [
        "GPT-4o",
        "GPT-4.1",
        "GPT-4.5",
        "GPT-5",
        "o1",
        "o3",
        "o4-mini",
    ]


def test_sync_config_adds_missing_tiktoken_encoding_and_preserves_other_entries(monkeypatch):
    module = _load_script_module()
    monkeypatch.setattr(module.tiktoken, "list_encoding_names", lambda: ["future_base"])
    monkeypatch.setattr(module.tiktoken_model, "MODEL_TO_ENCODING", {"gpt-future": "future_base"})
    monkeypatch.setattr(module.tiktoken_model, "MODEL_PREFIX_TO_ENCODING", {})

    config = {
        "tokenizers": [
            {
                "id": "qwen3",
                "name": "Qwen 3 tokenizer",
                "provider": "qwen",
                "type": "huggingface",
                "ref": "Qwen/Qwen3-0.6B",
                "used_by_examples": ["Qwen 3"],
            }
        ]
    }

    synced = module.sync_config(config)

    assert synced["tokenizers"][0]["id"] == "openai-future-base"
    assert synced["tokenizers"][0]["ref"] == "future_base"
    assert synced["tokenizers"][0]["used_by_examples"] == ["GPT-future"]
    assert synced["tokenizers"][1] == config["tokenizers"][0]
