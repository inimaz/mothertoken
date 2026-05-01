"""Smoke test the installed mothertoken CLI from a built wheel."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_cli(bin_dir: Path, *args: str) -> str:
    cmd = [str(bin_dir / "mothertoken"), *args]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return completed.stdout


def assert_contains(output: str, expected: list[str]) -> None:
    missing = [value for value in expected if value not in output]
    if missing:
        raise AssertionError(f"Missing expected output {missing!r} in:\n{output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    args = parser.parse_args()

    rank_output = run_cli(args.bin_dir, "rank", "ar")
    assert_contains(
        rank_output,
        [
            "Model ranking for",
            "arb_Arab",
            "GPT-4o",
            "Cost vs English",
            "1.56x",
        ],
    )

    list_output = run_cli(args.bin_dir, "list", "--local-only")
    assert_contains(
        list_output,
        [
            "gpt-4o",
            "qwen2.5",
            "local",
            "tiktoken / o200k_base",
        ],
    )
    if "claude-sonnet" in list_output or "API" in list_output:
        raise AssertionError(f"--local-only should not show API-backed tokenizers:\n{list_output}")

    tokenize_output = run_cli(args.bin_dir, "tokenize", "Hola mundo", "--language", "es", "--model", "gpt-4o")
    assert_contains(
        tokenize_output,
        [
            'Tokenizing: "Hola mundo"',
            "Benchmark language:",
            "spa_Latn",
            "GPT-4o",
            "English Est.",
            "Vs English",
        ],
    )


if __name__ == "__main__":
    main()
