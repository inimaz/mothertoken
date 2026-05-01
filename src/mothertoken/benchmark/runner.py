"""
mothertoken — benchmark/run_benchmark.py

Computes tokenization efficiency metrics across languages and models
using the FLORES+ corpus (openlanguagedata/flores_plus, CC BY-SA 4.0).

Outputs: src/mothertoken/data/benchmark.json — versioned, never contains raw sentences.

Usage:
    # Full run (all languages, all models)
    python run_benchmark.py

    # Single language
    python run_benchmark.py --languages tha_Thai

    # Subset of models
    python run_benchmark.py --models gpt-4o,llama3,mistral

    # Dry run to verify setup
    python run_benchmark.py --dry-run

Requirements:
    pip install datasets tiktoken transformers anthropic google-generativeai huggingface_hub
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

# Add `src/` to the python path so the `mothertoken` module can be resolved natively
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mothertoken.core import tokenizers
from mothertoken.core.resources import load_tokenizers_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mothertoken")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# FLORES+ split to use for public benchmark.
# Keep "devtest" as held-out validation — do not run against it routinely.
FLORES_SPLIT = "dev"
FLORES_DATASET = "openlanguagedata/flores_plus"

# English baseline config in FLORES+
ENGLISH_CONFIG = "eng_Latn"

# Languages to benchmark by default.
# Subset chosen to cover diverse script families and known efficiency gaps.
DEFAULT_LANGUAGES = [
    "eng_Latn",  # English — baseline
    "fra_Latn",  # French
    "spa_Latn",  # Spanish
    "por_Latn",  # Portuguese
    "deu_Latn",  # German
    "arb_Arab",  # Arabic
    "cmn_Hans",  # Chinese (Simplified)
    "jpn_Jpan",  # Japanese
    "tha_Thai",  # Thai
    "hin_Deva",  # Hindi
    "kor_Hang",  # Korean
    "tur_Latn",  # Turkish
    "ukr_Cyrl",  # Ukrainian
    "vie_Latn",  # Vietnamese
    "swh_Latn",  # Swahili
]


# Tokenizer registry and Configuration
def load_config() -> dict:
    return load_tokenizers_config()


def _get_config() -> dict:
    """Lazy loader so importing this module doesn't crash if tokenizers.yaml is absent."""
    if not hasattr(_get_config, "_cache"):
        _get_config._cache = load_config()  # type: ignore[attr-defined]
    return _get_config._cache  # type: ignore[attr-defined]


def _get_models() -> list:
    return _get_config().get("tokenizers", [])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class LanguageMetrics:
    language: str
    model_id: str
    num_sentences: int
    total_chars: int
    total_tokens: int
    total_words: int
    chars_per_token: float  # higher = better
    fertility: float  # tokens per word — lower = better
    rtc: float  # relative tokenization cost vs English baseline

    def to_dict(self):
        return asdict(self)


def compute_metrics(sentences: list[str], token_counts: list[int], english_chars_per_token: float) -> dict:
    """Compute aggregate metrics from a list of sentences and their token counts."""
    total_chars = sum(len(s) for s in sentences)
    total_tokens = sum(token_counts)
    total_words = sum(len(s.split()) for s in sentences)

    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0.0
    fertility = total_tokens / total_words if total_words > 0 else 0.0
    rtc = english_chars_per_token / chars_per_token if chars_per_token > 0 else 0.0

    return {
        "num_sentences": len(sentences),
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "chars_per_token": round(chars_per_token, 3),
        "fertility": round(fertility, 3),
        "rtc": round(rtc, 3),
    }


# ---------------------------------------------------------------------------
# FLORES+ loader
# ---------------------------------------------------------------------------


def load_flores_sentences(language_config: str, split: str = FLORES_SPLIT) -> list[str]:
    """
    Load sentences for a language config from FLORES+.
    Returns only the sentence strings — raw sentences never written to disk.
    Requires HuggingFace login: huggingface_hub.login()
    """
    from datasets import load_dataset

    log.info(f"Loading FLORES+ {language_config} / {split}")
    ds = load_dataset(FLORES_DATASET, language_config, split=split)
    return [row["text"] for row in ds]


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    languages: list[str],
    model_ids: list[str],
    dry_run: bool = False,
) -> tuple[dict, dict]:
    results = {}
    tokenizer_cache = {}

    # Step 1: get English baseline chars_per_token for each model
    log.info("Computing English baseline for all models...")
    english_sentences = load_flores_sentences(ENGLISH_CONFIG) if not dry_run else ["Hello world."] * 10
    english_cpt = {}  # model_id -> chars_per_token

    active_models = [m for m in _get_models() if m["id"] in model_ids]
    successful_models = []
    failed_models_log = {}

    for model in active_models:
        mid = model["id"]
        try:
            token_counts = tokenizers.tokenize_sentences(model, english_sentences, tokenizer_cache, dry_run)
            total_chars = sum(len(s) for s in english_sentences)
            total_tokens = sum(token_counts)
            english_cpt[mid] = total_chars / total_tokens if total_tokens > 0 else 1.0
            log.info(f"English baseline for {mid}: {english_cpt[mid]:.3f} chars/token")
            successful_models.append(model)
        except Exception as e:
            log.warning(f"Skipping {mid} entirely due to tokenizer load error: {e}")
            failed_models_log[mid] = str(e)

    active_models = successful_models

    # Step 2: run each language
    for lang_config in languages:
        log.info(f"Processing language: {lang_config}")
        sentences = load_flores_sentences(lang_config) if not dry_run else ["สวัสดีครับ"] * 10
        results[lang_config] = {}

        for model in active_models:
            mid = model["id"]
            try:
                token_counts = tokenizers.tokenize_sentences(model, sentences, tokenizer_cache, dry_run)
                metrics = compute_metrics(sentences, token_counts, english_cpt[mid])
                results[lang_config][mid] = metrics
                log.info(f"  {mid}: {metrics['chars_per_token']:.2f} c/t, RTC {metrics['rtc']:.2f}x")
            except Exception as e:
                log.warning(f"  Skipping {mid} for {lang_config}: {e}")
                results[lang_config][mid] = {"error": str(e)}

    return results, failed_models_log


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_benchmark(results: dict, errors: dict, output_path: Path, model_ids: list[str]):
    """
    Save benchmark results as versioned JSON.
    CRITICAL: raw sentences are never included in the output.
    """
    output = {
        "version": datetime.now(UTC).strftime("%Y-%m-%d"),
        "flores_split": FLORES_SPLIT,
        "flores_dataset": FLORES_DATASET,
        "baseline_language": ENGLISH_CONFIG,
        "models": [m for m in _get_models() if m["id"] in model_ids],
        "tokenizers": [m for m in _get_models() if m["id"] in model_ids],
        "metrics": results,
        "errors": errors,
        "note": (
            "Raw FLORES+ sentences are never stored here. "
            "Only aggregated metrics are published. "
            "See benchmark/run_benchmark.py to reproduce."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"Saved benchmark to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="mothertoken benchmark runner")
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated FLORES+ language configs (e.g. tha_Thai,arb_Arab)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(m["id"] for m in _get_models()),
        help="Comma-separated model IDs to benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "benchmark.json",
        help="Output path for benchmark.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without loading FLORES or real tokenizers — for setup verification",
    )
    args = parser.parse_args()

    languages = args.languages.split(",")
    model_ids = args.models.split(",")

    log.info(f"Running benchmark: {len(languages)} languages × {len(model_ids)} models")
    if args.dry_run:
        log.info("DRY RUN — using dummy data")

    results, errors = run_benchmark(languages, model_ids, dry_run=args.dry_run)
    save_benchmark(results, errors, args.output, model_ids)
    log.info("Done.")


if __name__ == "__main__":
    main()
