"""
mothertoken — benchmark/run_benchmark.py

Computes tokenization efficiency metrics across languages and models
using the FLORES+ corpus (openlanguagedata/flores_plus, CC BY-SA 4.0).

Outputs benchmark JSON — versioned, never contains raw sentences.

Usage:
    # Full run (all languages, all models)
    python run_benchmark.py

    # Single language
    python run_benchmark.py --languages tha_Thai

    # Subset of models
    python run_benchmark.py --models gpt-4o,llama3,mistral

    # Curated aliases and direct Hugging Face refs can be mixed
    python run_benchmark.py --models gpt-4o,Qwen/Qwen3-0.6B

    # Dry run to verify setup
    python run_benchmark.py --dry-run

Requirements:
    pip install datasets tiktoken transformers anthropic google-generativeai huggingface_hub
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

# Add `src/` to the python path so the `mothertoken` module can be resolved natively
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mothertoken.core import tokenizers
from mothertoken.core.tokenizer_registry_service import TokenizerRegistryService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mothertoken")
err_console = Console(stderr=True)

app = typer.Typer(
    name="benchmark",
    help="Run the mothertoken benchmark.",
    invoke_without_command=True,
    no_args_is_help=False,
    rich_markup_mode="rich",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# FLORES+ split to use for public benchmark.
# Keep "devtest" as held-out validation — do not run against it routinely.
FLORES_SPLIT = "dev"
FLORES_DATASET = "openlanguagedata/flores_plus"
DEFAULT_BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "data" / "default_benchmark.json"

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
    registry = TokenizerRegistryService()
    path_info = registry.path_info()
    if not registry.exists():
        raise FileNotFoundError(
            f"tokenizers.yaml not found at {path_info['path']} (exists={path_info['exists']}). Reinstall mothertoken."
        )
    return registry.load()


def _get_config() -> dict:
    """Lazy loader so importing this module doesn't crash if tokenizers.yaml is absent."""
    if not hasattr(_get_config, "_cache"):
        _get_config._cache = load_config()  # type: ignore[attr-defined]
    return _get_config._cache  # type: ignore[attr-defined]


def _get_models() -> list:
    return TokenizerRegistryService().list(_get_config())


def _looks_like_hf_ref(model_id: str) -> bool:
    return "/" in model_id or Path(model_id).exists()


def _hf_ref_model(ref: str) -> dict:
    return {
        "id": ref,
        "name": ref,
        "provider": "huggingface",
        "type": "huggingface",
        "ref": ref,
        "access": "local",
        "tokenizer_source": "huggingface",
        "verification_method": "user_supplied_ref",
        "used_by_examples": [ref],
        "api_key_env": None,
    }


def resolve_benchmark_models(model_ids: list[str]) -> list[dict]:
    """Resolve configured tokenizer aliases and direct Hugging Face refs."""
    configured_models = _get_models()
    configured_by_id = {model["id"]: model for model in configured_models}
    selected_models = []
    missing_model_ids = []

    for model_id in model_ids:
        if model_id in configured_by_id:
            selected_models.append(configured_by_id[model_id])
        elif _looks_like_hf_ref(model_id):
            selected_models.append(_hf_ref_model(model_id))
        else:
            missing_model_ids.append(model_id)

    if missing_model_ids:
        available = ", ".join(configured_by_id)
        missing = ", ".join(missing_model_ids)
        raise ValueError(
            f"Unknown model/tokenizer id(s): {missing}. "
            f"Use configured aliases ({available}) or Hugging Face refs like Qwen/Qwen3-0.6B."
        )

    return selected_models


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

    active_models = resolve_benchmark_models(model_ids)
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
    selected_models = resolve_benchmark_models(model_ids)
    output = {
        "version": datetime.now(UTC).strftime("%Y-%m-%d"),
        "flores_split": FLORES_SPLIT,
        "flores_dataset": FLORES_DATASET,
        "baseline_language": ENGLISH_CONFIG,
        "models": selected_models,
        "tokenizers": selected_models,
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


@app.callback()
def run(
    languages: Annotated[
        str,
        typer.Option(
            "--languages",
            help="Comma-separated FLORES+ language configs (e.g. tha_Thai,arb_Arab)",
        ),
    ] = ",".join(DEFAULT_LANGUAGES),
    models: Annotated[
        str | None,
        typer.Option("--models", help="Comma-separated model IDs to benchmark"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output path for benchmark JSON. Required unless --dry-run is used."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Run without loading FLORES or real tokenizers - for setup verification"),
    ] = False,
):
    """Run the tokenizer benchmark and optionally write benchmark JSON."""
    language_values = [language.strip() for language in languages.split(",") if language.strip()]
    if models:
        model_ids = [model.strip() for model in models.split(",") if model.strip()]
    else:
        model_ids = [m["id"] for m in _get_models()]

    if output is None and not dry_run:
        err_console.print(
            "[bold red]Error:[/] Provide [bold]--output[/] to choose where benchmark results should be written.\n"
            f"Maintainers can update the packaged default with: "
            f"[cyan]mothertoken benchmark --output {DEFAULT_BENCHMARK_PATH}[/]"
        )
        raise typer.Exit(code=1)

    log.info(f"Running benchmark: {len(language_values)} languages x {len(model_ids)} models")
    if dry_run:
        log.info("DRY RUN - using dummy data")

    results, errors = run_benchmark(language_values, model_ids, dry_run=dry_run)
    if output is not None:
        save_benchmark(results, errors, output, model_ids)
        log.info("Done.")
    else:
        log.info("Dry run complete. No output file written.")


def main():
    app()


if __name__ == "__main__":
    main()
