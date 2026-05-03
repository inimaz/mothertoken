from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mothertoken.core.tokenizer_registry_service import HEADER, TokenizerRegistryService

UNUSED_PATH = Path("unused-tokenizers.yaml")


def test_create_adds_tokenizer_entry():
    service = TokenizerRegistryService(UNUSED_PATH)
    config = {"tokenizers": []}

    created = service.create(config, {"id": "gpt-4o", "type": "tiktoken", "ref": "o200k_base"})

    assert created == {"id": "gpt-4o", "type": "tiktoken", "ref": "o200k_base"}
    assert config["tokenizers"] == [created]


def test_create_rejects_duplicate_id():
    service = TokenizerRegistryService(UNUSED_PATH)
    config = {"tokenizers": [{"id": "gpt-4o"}]}

    with pytest.raises(ValueError, match="already exists"):
        service.create(config, {"id": "gpt-4o"})


def test_update_changes_existing_entry():
    service = TokenizerRegistryService(UNUSED_PATH)
    config = {"tokenizers": [{"id": "llama3", "name": "Old name"}]}

    updated = service.update(config, "llama3", {"name": "Llama 3 tokenizer"})

    assert updated["name"] == "Llama 3 tokenizer"
    assert config["tokenizers"][0]["name"] == "Llama 3 tokenizer"


def test_delete_removes_existing_entry():
    service = TokenizerRegistryService(UNUSED_PATH)
    config = {"tokenizers": [{"id": "gpt-4o"}, {"id": "llama3"}]}

    deleted = service.delete(config, "gpt-4o")

    assert deleted == {"id": "gpt-4o"}
    assert config["tokenizers"] == [{"id": "llama3"}]


def test_add_if_missing_checks_requested_unique_fields():
    service = TokenizerRegistryService(UNUSED_PATH)
    config = {"tokenizers": [{"id": "hf-bert-tokenizer", "class_name": "BertTokenizer"}]}

    created = service.add_if_missing(
        config,
        {"id": "hf-bert-tokenizer-fast", "class_name": "BertTokenizer"},
        unique_fields=("id", "class_name"),
    )

    assert created is None
    assert config["tokenizers"] == [{"id": "hf-bert-tokenizer", "class_name": "BertTokenizer"}]


def test_load_save_and_dump_round_trip(tmp_path):
    path = tmp_path / "tokenizers.yaml"
    service = TokenizerRegistryService(path)
    config = {"tokenizers": [{"id": "gpt-4o", "used_by_examples": ["GPT-4o"]}]}

    service.save(config)

    text = path.read_text(encoding="utf-8")
    assert text.startswith(HEADER)
    assert service.load() == config
    assert yaml.safe_load(text) == config
