"""
mothertoken — cli/app.py

Typer-based CLI for the mothertoken tool.

Commands:
    rank      — rank all models for a given language from benchmark data
    models    — list supported models and tokenizer access
    tokenize  — count tokens in exact user-provided text
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="mothertoken",
    help="Rank language efficiency and tokenize text across models.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_benchmark_or_exit():
    from mothertoken.cli.benchmark_loader import load_benchmark

    try:
        return load_benchmark()
    except FileNotFoundError as e:
        err_console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1) from e


def _fmt_rtc(rtc: float) -> str:
    return f"{rtc:.2f}x"


LANGUAGE_ALIASES = {
    "en": "eng_Latn",
    "english": "eng_Latn",
    "es": "spa_Latn",
    "spanish": "spa_Latn",
    "pt": "por_Latn",
    "portuguese": "por_Latn",
    "de": "deu_Latn",
    "german": "deu_Latn",
    "fr": "fra_Latn",
    "french": "fra_Latn",
    "ar": "arb_Arab",
    "arabic": "arb_Arab",
    "zh": "cmn_Hans",
    "chinese": "cmn_Hans",
    "ja": "jpn_Jpan",
    "japanese": "jpn_Jpan",
    "th": "tha_Thai",
    "thai": "tha_Thai",
    "hi": "hin_Deva",
    "hindi": "hin_Deva",
    "ko": "kor_Hang",
    "korean": "kor_Hang",
    "tr": "tur_Latn",
    "turkish": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ukrainian": "ukr_Cyrl",
    "vi": "vie_Latn",
    "vietnamese": "vie_Latn",
    "sw": "swh_Latn",
    "swahili": "swh_Latn",
}


def _resolve_language(language: str, available: list[str]) -> str:
    """Resolve friendly language aliases while still accepting raw FLORES codes."""
    if language in available:
        return language

    resolved = LANGUAGE_ALIASES.get(language.strip().lower())
    if resolved in available:
        return resolved

    examples = "ar, arabic, es, spanish, eng_Latn"
    err_console.print(
        f"[bold red]Error:[/] Language [yellow]{language}[/] not found in benchmark.\nTry one of: {examples}"
    )
    raise typer.Exit(code=1)


def _load_models_config() -> list[dict]:
    """Load models from data/models.yaml (walk-up strategy, same as benchmark_loader)."""
    import yaml

    # Walk up from __file__ to find the project root containing data/models.yaml
    candidate = Path(__file__).resolve()
    config_path = None
    for _ in range(8):
        candidate = candidate.parent
        p = candidate / "data" / "models.yaml"
        if p.exists():
            config_path = p
            break
    if config_path is None:
        raise FileNotFoundError("data/models.yaml not found. Ensure you are running from the project root.")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("models", [])


def _model_access(model: dict) -> str:
    if model["type"] in ("tiktoken", "huggingface"):
        return "local"
    return "API"


def _tokenizer_backend(model: dict) -> str:
    labels = {
        "tiktoken": "tiktoken",
        "huggingface": "Hugging Face",
        "anthropic_api": "Anthropic",
        "google_api": "Google",
    }
    return f"{labels.get(model['type'], model['type'])} / {model['ref']}"


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------


def _show_model_ranking(language: str) -> None:
    data = _load_benchmark_or_exit()

    from mothertoken.cli.benchmark_loader import get_language_metrics, get_languages, get_model_name

    available = get_languages(data)
    language = _resolve_language(language, available)

    metrics = get_language_metrics(data, language)
    if not metrics:
        err_console.print(f"[bold red]Error:[/] No metrics found for {language}.")
        raise typer.Exit(code=1)

    # Sort by chars_per_token descending (higher = more efficient)
    rows = []
    for model_id, m in metrics.items():
        if "error" in m:
            continue
        rows.append((model_id, m))
    rows.sort(key=lambda r: r[1]["chars_per_token"], reverse=True)

    console.print()
    console.print(f"[bold]📊 Model ranking for[/] [cyan]{language}[/]")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Model")
    table.add_column("Chars/Token", justify="right")
    table.add_column("Cost vs English", justify="right")

    for i, (model_id, m) in enumerate(rows, 1):
        name = get_model_name(data, model_id)
        rtc_val = m["rtc"]
        rtc_str = _fmt_rtc(rtc_val)
        rtc_text = Text(rtc_str)
        if rtc_val >= 3.0:
            rtc_text.stylize("bold red")
        elif rtc_val >= 2.0:
            rtc_text.stylize("yellow")
        elif rtc_val >= 1.5:
            rtc_text.stylize("dark_orange")

        rank_str = "🥇" if i == 1 else str(i)
        table.add_row(
            rank_str,
            name,
            f"{m['chars_per_token']:.3f}",
            rtc_text,
        )

    console.print(table)

    best_model = get_model_name(data, rows[0][0])
    console.print(f"[green]💡 [bold]{best_model}[/] is the most efficient tokenizer for [cyan]{language}[/].[/]")
    console.print()


@app.command()
def rank(
    language: Annotated[str, typer.Argument(help="Language alias or FLORES+ code, e.g. ar, arabic, arb_Arab")],
):
    """Rank models by tokenizer efficiency for one language."""
    _show_model_ranking(language)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@app.command()
def models(
    local_only: Annotated[bool, typer.Option("--local-only", help="Show only local tokenizers")] = False,
):
    """List supported models and tokenizer access."""
    models_cfg = _load_models_config()
    if local_only:
        models_cfg = [m for m in models_cfg if m["type"] in ("tiktoken", "huggingface")]

    if not models_cfg:
        err_console.print("[bold red]Error:[/] No models are configured.")
        raise typer.Exit(code=1)

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Access")
    table.add_column("Tokenizer")

    for model_cfg in models_cfg:
        table.add_row(
            model_cfg["id"],
            model_cfg["name"],
            _model_access(model_cfg),
            _tokenizer_backend(model_cfg),
        )

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


@app.command()
def tokenize(
    text: Annotated[str | None, typer.Argument(help="Text to tokenize exactly as provided")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Path to a text file to tokenize")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model ID, e.g. gpt-4o")] = None,
):
    """Count tokens for exact text using local tokenizers."""
    if text is None and file is None:
        err_console.print("[bold red]Error:[/] Provide text or [bold]--file[/].")
        raise typer.Exit(code=1)
    if text is not None and file is not None:
        err_console.print("[bold red]Error:[/] Provide text or [bold]--file[/], not both.")
        raise typer.Exit(code=1)
    if file is not None:
        if not file.exists():
            err_console.print(f"[bold red]Error:[/] File not found: {file}")
            raise typer.Exit(code=1)
        input_text = file.read_text(encoding="utf-8")
    else:
        input_text = text or ""

    all_models = _load_models_config()
    local_models = [m for m in all_models if m["type"] in ("tiktoken", "huggingface")]

    if model is not None:
        model_cfg = next((m for m in all_models if m["id"] == model), None)
        if model_cfg is None:
            available = ", ".join(m["id"] for m in local_models)
            err_console.print(f"[bold red]Error:[/] Model [yellow]{model}[/] not found.\nAvailable: {available}")
            raise typer.Exit(code=1)
        if model_cfg["type"] not in ("tiktoken", "huggingface"):
            err_console.print(
                f"[bold red]Error:[/] Model [yellow]{model}[/] uses an API-backed tokenizer. "
                "The [bold]tokenize[/] command only uses local tokenizers."
            )
            raise typer.Exit(code=1)
        candidate_models = [model_cfg]
    else:
        candidate_models = local_models

    if not candidate_models:
        err_console.print("[bold red]Error:[/] No local tokenizers are configured.")
        raise typer.Exit(code=1)

    from mothertoken.core.tokenizers import tokenize_sentences

    cache: dict = {}
    results: list[tuple[dict, int | None, float | None, str | None]] = []

    preview = input_text.strip()[:60].replace("\n", " ")
    if len(input_text) > 60:
        preview += "..."

    console.print()
    console.print(f'[bold]Tokenizing:[/] [cyan]"{preview}"[/] ({len(input_text)} chars)\n')

    with console.status("Running local tokenizers..."):
        for model_cfg in candidate_models:
            try:
                counts = tokenize_sentences(model_cfg, [input_text], cache, dry_run=False)
                token_count = counts[0]
                cpt = len(input_text) / token_count if token_count > 0 else 0.0
                results.append((model_cfg, token_count, cpt, None))
            except Exception as e:
                results.append((model_cfg, None, None, str(e)))

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Tokens", justify="right")
    table.add_column("Chars/Token", justify="right")

    sorted_results = sorted(results, key=lambda row: row[2] if row[2] is not None else -1, reverse=True)
    for model_cfg, token_count, cpt, error in sorted_results:
        if error is not None:
            table.add_row(model_cfg["name"], "[red]error[/]", "—")
            continue
        table.add_row(model_cfg["name"], str(token_count), f"{cpt:.3f}")

    console.print(table)

    errors = [(model_cfg, error) for model_cfg, _, _, error in results if error is not None]
    if errors:
        console.print("\n[dim]Errors:[/]")
        for model_cfg, error in errors:
            console.print(f"  [yellow]{model_cfg['name']}[/]: [dim]{error}[/]")

    if model is not None and errors:
        raise typer.Exit(code=1)

    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
