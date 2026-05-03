"""CRUD helpers for the tokenizers.yaml registry."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

HEADER = """# mothertoken - Tokenizer registry for tokenization benchmarks
# This registry dictates which tokenizers to load and which provider APIs to query.
# `used_by_examples` is discoverability metadata, not a complete model database.

"""


class TokenizerRegistryService:
    """Load, query, mutate, and write a tokenizers.yaml config."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        with self.path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        config.setdefault("tokenizers", [])
        return config

    def save(self, config: dict[str, Any]) -> None:
        self.path.write_text(self.dump(config), encoding="utf-8")

    def list(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        return list(config.get("tokenizers") or [])

    def get(self, config: dict[str, Any], tokenizer_id: str) -> dict[str, Any] | None:
        for entry in self.list(config):
            if entry.get("id") == tokenizer_id:
                return entry
        return None

    def find_by(self, config: dict[str, Any], field: str, value: Any) -> dict[str, Any] | None:
        for entry in self.list(config):
            if entry.get(field) == value:
                return entry
        return None

    def create(self, config: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
        if not entry.get("id"):
            raise ValueError("Tokenizer entries must include an id")
        if self.get(config, entry["id"]) is not None:
            raise ValueError(f"Tokenizer entry {entry['id']!r} already exists")

        tokenizers = config.setdefault("tokenizers", [])
        created = copy.deepcopy(entry)
        tokenizers.append(created)
        return created

    def update(self, config: dict[str, Any], tokenizer_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        entry = self.get(config, tokenizer_id)
        if entry is None:
            raise KeyError(f"Tokenizer entry {tokenizer_id!r} was not found")
        if "id" in updates and updates["id"] != tokenizer_id and self.get(config, updates["id"]) is not None:
            raise ValueError(f"Tokenizer entry {updates['id']!r} already exists")

        entry.update(copy.deepcopy(updates))
        return entry

    def delete(self, config: dict[str, Any], tokenizer_id: str) -> dict[str, Any]:
        tokenizers = config.setdefault("tokenizers", [])
        for index, entry in enumerate(tokenizers):
            if entry.get("id") == tokenizer_id:
                return tokenizers.pop(index)
        raise KeyError(f"Tokenizer entry {tokenizer_id!r} was not found")

    def add_if_missing(
        self,
        config: dict[str, Any],
        entry: dict[str, Any],
        *,
        unique_fields: tuple[str, ...] = ("id",),
    ) -> dict[str, Any] | None:
        for field in unique_fields:
            value = entry.get(field)
            if value is not None and self.find_by(config, field, value) is not None:
                return None
        return self.create(config, entry)

    @classmethod
    def dump(cls, config: dict[str, Any]) -> str:
        class IndentDumper(yaml.SafeDumper):
            def increase_indent(self, flow: bool = False, indentless: bool = False):
                return super().increase_indent(flow, indentless=False)

        body = yaml.dump(config, Dumper=IndentDumper, sort_keys=False, allow_unicode=True, default_flow_style=False)
        return f"{HEADER}{body}"
