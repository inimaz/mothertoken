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

from mothertoken.core.registry import LOCAL_MODEL_TYPES, AccessMode, ModelType

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


def _fmt_token_estimate(value: float | None) -> str:
    if value is None:
        return "—"
    return str(round(value))


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
    """Load the model registry."""
    from mothertoken.core.resources import load_models_config

    cfg = load_models_config()
    return cfg.get("models", [])


def _model_access(model: dict) -> str:
    access = model.get("access")
    if access:
        return "API" if access == AccessMode.API else access
    return AccessMode.LOCAL.value if model["type"] in LOCAL_MODEL_TYPES else "API"


def _tokenizer_backend(model: dict) -> str:
    labels = {
        ModelType.TIKTOKEN.value: "tiktoken",
        ModelType.HUGGINGFACE.value: "Hugging Face",
        ModelType.ANTHROPIC_API.value: "Anthropic",
        ModelType.GOOGLE_API.value: "Google",
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
    if data.get("version"):
        console.print(f"[dim]Benchmark version:[/] {data['version']}")
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
        models_cfg = [m for m in models_cfg if m["type"] in LOCAL_MODEL_TYPES]

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
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language alias or FLORES+ code for benchmark English-equivalent estimates",
        ),
    ] = None,
    english_text: Annotated[
        str | None,
        typer.Option("--english-text", help="English translation to compare against this text"),
    ] = None,
    english_file: Annotated[
        Path | None,
        typer.Option("--english-file", help="Path to an English translation file to compare against this text"),
    ] = None,
    include_api: Annotated[
        bool,
        typer.Option(
            "--include-api",
            help="Include API-backed tokenizers. Requires the provider API key environment variables.",
        ),
    ] = False,
):
    """Count tokens for exact text. Defaults to local tokenizers; use --include-api for provider counters."""
    if text is None and file is None:
        err_console.print("[bold red]Error:[/] Provide text or [bold]--file[/].")
        raise typer.Exit(code=1)
    if text is not None and file is not None:
        err_console.print("[bold red]Error:[/] Provide text or [bold]--file[/], not both.")
        raise typer.Exit(code=1)
    if english_text is not None and english_file is not None:
        err_console.print("[bold red]Error:[/] Provide [bold]--english-text[/] or [bold]--english-file[/], not both.")
        raise typer.Exit(code=1)
    if file is not None:
        if not file.exists():
            err_console.print(f"[bold red]Error:[/] File not found: {file}")
            raise typer.Exit(code=1)
        input_text = file.read_text(encoding="utf-8")
    else:
        input_text = text or ""

    paired_english_text: str | None = None
    if english_file is not None:
        if not english_file.exists():
            err_console.print(f"[bold red]Error:[/] English file not found: {english_file}")
            raise typer.Exit(code=1)
        paired_english_text = english_file.read_text(encoding="utf-8")
    elif english_text is not None:
        paired_english_text = english_text

    all_models = _load_models_config()
    local_models = [m for m in all_models if m["type"] in LOCAL_MODEL_TYPES]
    api_models = [m for m in all_models if m["type"] not in LOCAL_MODEL_TYPES]

    if model is not None:
        model_cfg = next((m for m in all_models if m["id"] == model), None)
        if model_cfg is None:
            available_models = all_models if include_api else local_models
            available = ", ".join(m["id"] for m in available_models)
            if not include_api and api_models:
                available += "\nAPI-backed models require --include-api."
            err_console.print(f"[bold red]Error:[/] Model [yellow]{model}[/] not found.\nAvailable: {available}")
            raise typer.Exit(code=1)
        if model_cfg["type"] not in LOCAL_MODEL_TYPES and not include_api:
            err_console.print(
                f"[bold red]Error:[/] Model [yellow]{model}[/] uses an API-backed tokenizer. "
                "Rerun with [bold]--include-api[/] and the required provider API key."
            )
            raise typer.Exit(code=1)
        candidate_models = [model_cfg]
    else:
        candidate_models = all_models if include_api else local_models

    if not candidate_models:
        err_console.print("[bold red]Error:[/] No tokenizers are configured.")
        raise typer.Exit(code=1)

    benchmark_metrics: dict[str, dict] = {}
    resolved_language: str | None = None
    if language is not None:
        data = _load_benchmark_or_exit()
        from mothertoken.cli.benchmark_loader import get_language_metrics, get_languages

        resolved_language = _resolve_language(language, get_languages(data))
        benchmark_metrics = get_language_metrics(data, resolved_language)
        if not benchmark_metrics:
            err_console.print(f"[bold red]Error:[/] No benchmark metrics found for {resolved_language}.")
            raise typer.Exit(code=1)

    from mothertoken.core.tokenizers import tokenize_sentences

    cache: dict = {}
    results: list[tuple[dict, int | None, float | None, int | None, str | None]] = []

    preview = input_text.strip()[:60].replace("\n", " ")
    if len(input_text) > 60:
        preview += "..."

    console.print()
    console.print(f'[bold]Tokenizing:[/] [cyan]"{preview}"[/] ({len(input_text)} chars)\n')
    if resolved_language is not None:
        console.print(f"[dim]Benchmark language:[/] [cyan]{resolved_language}[/]")
    if paired_english_text is not None:
        console.print("[dim]English comparison:[/] provided by user")
    if include_api:
        console.print("[dim]API-backed tokenizers:[/] enabled")
    if resolved_language is not None or paired_english_text is not None:
        console.print()

    status_label = "Running local and API tokenizers..." if include_api else "Running local tokenizers..."
    with console.status(status_label):
        for model_cfg in candidate_models:
            try:
                sentences = [input_text]
                if paired_english_text is not None:
                    sentences.append(paired_english_text)
                counts = tokenize_sentences(model_cfg, sentences, cache, dry_run=False)
                token_count = counts[0]
                english_token_count = counts[1] if paired_english_text is not None else None
                cpt = len(input_text) / token_count if token_count > 0 else 0.0
                results.append((model_cfg, token_count, cpt, english_token_count, None))
            except Exception as e:
                results.append((model_cfg, None, None, None, str(e)))

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Model")
    if include_api:
        table.add_column("Access")
    table.add_column("Tokens", justify="right")
    table.add_column("Chars/Token", justify="right")
    if resolved_language is not None:
        table.add_column("English Est.", justify="right")
        table.add_column("Vs English", justify="right")
    if paired_english_text is not None:
        table.add_column("English Tokens", justify="right")
        table.add_column("Paired Ratio", justify="right")

    sorted_results = sorted(results, key=lambda row: row[2] if row[2] is not None else -1, reverse=True)
    for model_cfg, token_count, cpt, english_token_count, error in sorted_results:
        access = _model_access(model_cfg)
        if error is not None:
            row = [model_cfg["name"]]
            if include_api:
                row.append(access)
            row.extend(["[red]error[/]", "—"])
            if resolved_language is not None:
                row.extend(["—", "—"])
            if paired_english_text is not None:
                row.extend(["—", "—"])
            table.add_row(*row)
            continue
        row = [model_cfg["name"]]
        if include_api:
            row.append(access)
        row.extend([str(token_count), f"{cpt:.3f}"])
        if resolved_language is not None:
            rtc = benchmark_metrics.get(model_cfg["id"], {}).get("rtc")
            english_estimate = token_count / rtc if token_count is not None and rtc else None
            row.extend([_fmt_token_estimate(english_estimate), _fmt_rtc(rtc) if rtc else "—"])
        if paired_english_text is not None:
            paired_ratio = (
                token_count / english_token_count if token_count is not None and english_token_count else None
            )
            row.extend([str(english_token_count), _fmt_rtc(paired_ratio) if paired_ratio else "—"])
        table.add_row(*row)

    console.print(table)

    errors = [(model_cfg, error) for model_cfg, _, _, _, error in results if error is not None]
    if errors:
        console.print("\n[dim]Errors:[/]")
        for model_cfg, error in errors:
            console.print(f"  [yellow]{model_cfg['name']}[/]: [dim]{error}[/]")

    if resolved_language is not None:
        console.print("\n[dim]English Est. uses the benchmark vs-English multiplier for this language/model.[/]")
    if paired_english_text is not None:
        console.print("[dim]Paired Ratio compares your original text against the supplied English translation.[/]")
    if include_api:
        console.print("[dim]API-backed rows use provider token counters and require provider API keys.[/]")

    if model is not None and errors:
        raise typer.Exit(code=1)

    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
