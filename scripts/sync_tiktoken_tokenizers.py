#!/usr/bin/env python3
"""Sync OpenAI tiktoken encodings and model mappings into tokenizers.yaml."""

from __future__ import annotations

import argparse
import copy
import difflib
from pathlib import Path
from typing import Any

import tiktoken
import tiktoken.model as tiktoken_model
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZERS_PATH = ROOT / "src" / "mothertoken" / "data" / "tokenizers.yaml"

HEADER = """# mothertoken - Tokenizer registry for tokenization benchmarks
# This registry dictates which tokenizers to load and which provider APIs to query.
# `used_by_examples` is discoverability metadata, not a complete model database.

"""

ENCODING_ID_OVERRIDES = {
    "o200k_base": "gpt-4o",
    "o200k_harmony": "gpt-oss",
    "cl100k_base": "gpt-4",
    "p50k_base": "codex",
    "p50k_edit": "codex-edit",
    "r50k_base": "gpt-3",
    "gpt2": "gpt2",
}

MAJOR_EXAMPLES_BY_ENCODING = {
    "o200k_base": ["GPT-4o", "GPT-4.1", "GPT-4.5", "GPT-5", "o1", "o3", "o4-mini"],
    "o200k_harmony": ["GPT-OSS"],
    "cl100k_base": ["GPT-4", "GPT-3.5 Turbo"],
    "p50k_base": ["Codex"],
    "p50k_edit": ["Codex edit"],
    "r50k_base": ["GPT-3"],
    "gpt2": ["GPT-2"],
}


def _entry_id_for_encoding(encoding: str) -> str:
    return ENCODING_ID_OVERRIDES.get(encoding, f"openai-{encoding.replace('_', '-')}")


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _infer_major_model_name(model: str) -> str | None:
    model = model.rstrip("-")
    if model.startswith(("ft:", "chatgpt-", "text-", "code-", "davinci", "babbage", "ada", "curie")):
        return None
    if model in {"gpt-35-turbo", "gpt-35-turbo-"}:
        return None
    if model == "gpt2":
        return "GPT-2"
    if model.startswith("gpt-oss"):
        return "GPT-OSS"
    if model.startswith("gpt-"):
        return "GPT-" + model.removeprefix("gpt-")
    if model.startswith("o") and any(char.isdigit() for char in model):
        return model
    return None


def _keep_existing_example(example: str) -> bool:
    generated_examples = {example for examples in MAJOR_EXAMPLES_BY_ENCODING.values() for example in examples}
    return example in generated_examples


def tiktoken_examples_by_encoding() -> dict[str, list[str]]:
    """Return major user-facing model examples grouped by tiktoken encoding."""
    examples: dict[str, list[str]] = {
        encoding: list(MAJOR_EXAMPLES_BY_ENCODING.get(encoding, [])) for encoding in tiktoken.list_encoding_names()
    }

    for model, encoding in sorted(tiktoken_model.MODEL_TO_ENCODING.items()):
        if encoding in MAJOR_EXAMPLES_BY_ENCODING:
            continue
        inferred = _infer_major_model_name(model)
        if inferred:
            examples.setdefault(encoding, []).append(inferred)

    for prefix, encoding in sorted(tiktoken_model.MODEL_PREFIX_TO_ENCODING.items()):
        if encoding in MAJOR_EXAMPLES_BY_ENCODING:
            continue
        inferred = _infer_major_model_name(prefix)
        if inferred:
            examples.setdefault(encoding, []).append(inferred)

    return {encoding: _dedupe(values) for encoding, values in examples.items()}


def _encoding_for_entry(entry: dict[str, Any], encodings: set[str]) -> str | None:
    ref = entry.get("ref")
    if entry.get("provider") == "openai" and entry.get("type") == "tiktoken" and ref in encodings:
        return ref

    for encoding, entry_id in ENCODING_ID_OVERRIDES.items():
        if entry.get("provider") == "openai" and entry.get("id") == entry_id and encoding in encodings:
            return encoding

    return None


def _build_entry(encoding: str, existing: dict[str, Any] | None, examples: list[str]) -> dict[str, Any]:
    existing_examples = [
        example for example in (existing or {}).get("used_by_examples", []) if _keep_existing_example(example)
    ]

    return {
        "id": (existing or {}).get("id") or _entry_id_for_encoding(encoding),
        "name": f"OpenAI {encoding}",
        "provider": "openai",
        "type": "tiktoken",
        "ref": encoding,
        "access": "local",
        "tokenizer_source": "tiktoken",
        "verification_method": "local_fixture_count",
        "used_by_examples": _dedupe([*existing_examples, *examples]),
        "api_key_env": None,
    }


def sync_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of config with tiktoken-backed OpenAI tokenizers synced."""
    synced = copy.deepcopy(config)
    tokenizers = synced.get("tokenizers") or []
    examples_by_encoding = tiktoken_examples_by_encoding()
    encodings = set(examples_by_encoding)

    existing_by_encoding: dict[str, dict[str, Any]] = {}
    for entry in tokenizers:
        encoding = _encoding_for_entry(entry, encodings)
        if encoding and encoding not in existing_by_encoding:
            existing_by_encoding[encoding] = entry

    replacement_by_encoding = {
        encoding: _build_entry(encoding, existing_by_encoding.get(encoding), examples)
        for encoding, examples in examples_by_encoding.items()
    }

    next_tokenizers = []
    emitted = set()
    for entry in tokenizers:
        encoding = _encoding_for_entry(entry, encodings)
        if encoding:
            if encoding not in emitted:
                next_tokenizers.append(replacement_by_encoding[encoding])
                emitted.add(encoding)
            continue
        next_tokenizers.append(entry)

    missing = [encoding for encoding in tiktoken.list_encoding_names() if encoding not in emitted]
    insert_at = _openai_insert_index(next_tokenizers)
    for offset, encoding in enumerate(missing):
        next_tokenizers.insert(insert_at + offset, replacement_by_encoding[encoding])

    synced["tokenizers"] = next_tokenizers
    return synced


def _openai_insert_index(tokenizers: list[dict[str, Any]]) -> int:
    last_openai_index = -1
    for index, entry in enumerate(tokenizers):
        if entry.get("provider") == "openai":
            last_openai_index = index
    return last_openai_index + 1 if last_openai_index >= 0 else 0


def dump_config(config: dict[str, Any]) -> str:
    class IndentDumper(yaml.SafeDumper):
        def increase_indent(self, flow: bool = False, indentless: bool = False):
            return super().increase_indent(flow, indentless=False)

    body = yaml.dump(config, Dumper=IndentDumper, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return f"{HEADER}{body}"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync tiktoken encodings into tokenizers.yaml")
    parser.add_argument("--path", type=Path, default=DEFAULT_TOKENIZERS_PATH, help="Path to tokenizers.yaml")
    parser.add_argument("--check", action="store_true", help="Exit non-zero if tokenizers.yaml is out of sync")
    parser.add_argument("--dry-run", action="store_true", help="Print the synced YAML instead of writing it")
    args = parser.parse_args()

    original_text = args.path.read_text(encoding="utf-8")
    original = load_config(args.path)
    synced_text = dump_config(sync_config(original))

    if args.check:
        if synced_text == original_text:
            print(f"{args.path} is already in sync with tiktoken.")
            return 0
        diff = difflib.unified_diff(
            original_text.splitlines(keepends=True),
            synced_text.splitlines(keepends=True),
            fromfile=str(args.path),
            tofile=f"{args.path} (synced)",
        )
        print("".join(diff), end="")
        return 1

    if args.dry_run:
        print(synced_text, end="")
        return 0

    if synced_text == original_text:
        print(f"{args.path} is already in sync with tiktoken.")
        return 0

    args.path.write_text(synced_text, encoding="utf-8")
    print(f"Synced {args.path} from tiktoken.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
