"""
mothertoken — cli/app.py

Typer-based CLI for the mothertoken tool.

Commands:
    compare   — rank all models for a given language from benchmark data
    token     — count tokens in a text string for a specific model (local tokenizers only)
    analyze   — tokenize user text across models using local or full (API) mode
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="mothertoken",
    help="Benchmark and analyze tokenizer efficiency across languages and models.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


class Mode(str, Enum):
    local = "local"
    full = "full"


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


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command()
def compare(
    language: Annotated[str, typer.Option("--language", "-l", help="FLORES+ language code, e.g. arb_Arab")],
):
    """Rank all models by tokenizer efficiency for a given language."""
    data = _load_benchmark_or_exit()

    from mothertoken.cli.benchmark_loader import get_language_metrics, get_languages, get_model_name

    available = get_languages(data)
    if language not in available:
        err_console.print(
            f"[bold red]Error:[/] Language [yellow]{language}[/] not found in benchmark.\n"
            f"Available: {', '.join(sorted(available))}"
        )
        raise typer.Exit(code=1)

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
    table.add_column("Fertility", justify="right")
    table.add_column("RTC (Cost Multiplier)", justify="right")

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
            f"{m['fertility']:.3f}",
            rtc_text,
        )

    console.print(table)

    best_model = get_model_name(data, rows[0][0])
    console.print(f"[green]💡 [bold]{best_model}[/] is the most efficient tokenizer for [cyan]{language}[/].[/]")
    console.print()


# ---------------------------------------------------------------------------
# token
# ---------------------------------------------------------------------------


@app.command()
def token(
    text: Annotated[str, typer.Option("--text", "-t", help="Text to tokenize")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model ID, e.g. gpt-4o")],
):
    """Count tokens in a text string for a specific model (local tokenizers only)."""
    models_cfg = _load_models_config()
    model_cfg = next((m for m in models_cfg if m["id"] == model), None)

    if model_cfg is None:
        available = ", ".join(m["id"] for m in models_cfg)
        err_console.print(f"[bold red]Error:[/] Model [yellow]{model}[/] not found.\nAvailable: {available}")
        raise typer.Exit(code=1)

    mtype = model_cfg["type"]

    # API-only models are not supported in this command (no text to translate)
    if mtype in ("anthropic_api", "google_api"):
        console.print(
            f"[yellow]ℹ️  {model_cfg['name']} uses a closed API tokenizer.[/]\n"
            "  The [bold]token[/] command only supports local tokenizers (tiktoken, HuggingFace).\n"
            "  Use [bold]analyze --mode full[/] with an API key to include this model."
        )
        raise typer.Exit(code=0)

    from mothertoken.core.tokenizers import tokenize_sentences

    cache: dict = {}
    with console.status(f"Tokenizing with [bold]{model_cfg['name']}[/]…"):
        try:
            counts = tokenize_sentences(model_cfg, [text], cache, dry_run=False)
        except Exception as e:
            err_console.print(f"[bold red]Error:[/] {e}")
            raise typer.Exit(code=1) from e

    token_count = counts[0]
    char_count = len(text)
    cpt = char_count / token_count if token_count > 0 else 0.0

    preview = text if len(text) <= 40 else text[:37] + "…"
    console.print()
    console.print(
        f'[bold]🔤[/]  [cyan]"{preview}"[/]\n'
        f"     → [bold]{token_count}[/] token{'s' if token_count != 1 else ''}  "
        f"([dim]{char_count} chars, {cpt:.2f} chars/token[/])  "
        f"[dim][{model_cfg['name']} / {model_cfg['ref']}][/]"
    )
    console.print()


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    text: Annotated[str | None, typer.Option("--text", "-t", help="Text to analyze inline")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Path to a text file to analyze")] = None,
    languages: Annotated[
        str,
        typer.Option("--languages", "-l", help="Comma-separated FLORES+ language codes, e.g. eng_Latn,arb_Arab"),
    ] = "eng_Latn",
    models: Annotated[
        str | None,
        typer.Option("--models", "-m", help="Comma-separated model IDs (default: all local models)"),
    ] = None,
    mode: Annotated[
        Mode, typer.Option("--mode", help="local = public tokenizers only; full = include API models")
    ] = Mode.local,
):
    """Analyze your text across models and languages using real tokenizers."""
    # Resolve input text
    if text is None and file is None:
        err_console.print("[bold red]Error:[/] Provide either [bold]--text[/] or [bold]--file[/].")
        raise typer.Exit(code=1)
    if file is not None:
        if not file.exists():
            err_console.print(f"[bold red]Error:[/] File not found: {file}")
            raise typer.Exit(code=1)
        input_text = file.read_text(encoding="utf-8")
    else:
        input_text = text  # type: ignore[assignment]

    lang_list = [lang.strip() for lang in languages.split(",") if lang.strip()]

    # Load model configs and filter
    all_models = _load_models_config()

    if mode == Mode.local:
        candidate_models = [m for m in all_models if m["type"] in ("tiktoken", "huggingface")]
    else:
        candidate_models = all_models  # full mode includes API models

    if models is not None:
        requested = {m.strip() for m in models.split(",")}
        candidate_models = [m for m in candidate_models if m["id"] in requested]
        if not candidate_models:
            err_console.print(f"[bold red]Error:[/] None of the requested models match mode '{mode.value}'.")
            raise typer.Exit(code=1)

    from mothertoken.core.tokenizers import tokenize_sentences

    # For analyze, we tokenize the same input text regardless of language —
    # the language arg selects which benchmark rows to show for comparison context,
    # not which corpus to tokenize. We tokenize the user's actual text per model.

    cache: dict = {}
    results: dict[str, dict] = {}  # model_id -> {token_count, chars_per_token, ...}

    preview = input_text.strip()[:60].replace("\n", " ")
    if len(input_text) > 60:
        preview += "…"

    console.print()
    console.print(f'[bold]📊 Tokenizing:[/] [cyan]"{preview}"[/] ({len(input_text)} chars)\n')

    data = _load_benchmark_or_exit()

    with console.status("Running tokenizers…"):
        for model_cfg in candidate_models:
            mid = model_cfg["id"]
            try:
                counts = tokenize_sentences(model_cfg, [input_text], cache, dry_run=False)
                token_count = counts[0]
                cpt = len(input_text) / token_count if token_count > 0 else 0.0
                results[mid] = {"token_count": token_count, "chars_per_token": cpt}
            except Exception as e:
                results[mid] = {"error": str(e)}

    # Build table — one row per model, showing token count and cost estimate
    benchmark_metrics = data.get("metrics", {})

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Tokens", justify="right")
    table.add_column("Chars/Token", justify="right")
    table.add_column("Cost Multiplier (RTC)", justify="right")

    # Sort by chars_per_token desc (most efficient first), errors last
    sorted_results = sorted(
        results.items(),
        key=lambda kv: kv[1].get("chars_per_token", -1),
        reverse=True,
    )

    for mid, res in sorted_results:
        model_name = next((m["name"] for m in candidate_models if m["id"] == mid), mid)
        if "error" in res:
            table.add_row(model_name, "[red]error[/]", "—", "—")
            continue

        token_count = res["token_count"]
        cpt = res["chars_per_token"]

        # Derive RTC from benchmark if possible
        rtc_str = "—"
        rtc_val = 0.0
        for lang in lang_list:
            lang_metrics = benchmark_metrics.get(lang, {})
            if mid in lang_metrics and "rtc" in lang_metrics[mid]:
                rtc_val = lang_metrics[mid]["rtc"]
                rtc_str = _fmt_rtc(rtc_val)
                break

        table.add_row(model_name, str(token_count), f"{cpt:.3f}", rtc_str)

    console.print(table)

    # Summary insight like in ROADMAP.md
    if results:
        # Pick the most expensive (highest RTC) and least efficient model for the insight
        # Or just pick the first one if it's the most efficient.
        # Actually follow the roadmap example: compare first lang if provided.
        primary_lang = lang_list[0] if lang_list else "your language"

        # Find model with best CPT
        if sorted_results and "chars_per_token" in sorted_results[0][1]:
            best_mid, best_res = sorted_results[0]
            best_name = next((m["name"] for m in candidate_models if m["id"] == best_mid), best_mid)

            # Find a model with high RTC for comparison
            # This is illustrative, matching the roadmap examples.
            worst_mid = next((mid for mid, res in reversed(sorted_results) if "chars_per_token" in res), None)
            if worst_mid:
                lang_metrics = benchmark_metrics.get(lang_list[0], {}) if lang_list else {}
                rtc_val = lang_metrics.get(worst_mid, {}).get("rtc", 0.0)
                if rtc_val > 1.0:
                    worst_name = next((m["name"] for m in candidate_models if m["id"] == worst_mid), worst_mid)
                    console.print(
                        f"\n[bold]💡 {primary_lang.split('_')[0].capitalize()} needs {rtc_val:.1f}x "
                        f"more tokens than English on {worst_name}.[/]"
                    )
                    console.print(f"📐 That's {rtc_val:.1f}x less effective context window for those users.")
                    console.print(
                        f"💸 Whatever you pay per token, {primary_lang.split('_')[0].capitalize()} "
                        f"costs you {rtc_val:.1f}x more on {worst_name}."
                    )
                    console.print(f"   → Switching to {best_name} reduces that overhead.\n")

    # Print errors
    errored = [(mid, res["error"]) for mid, res in results.items() if "error" in res]
    if errored:
        console.print("[dim]Errors:[/]")
        for mid, err in errored:
            model_name = next((m["name"] for m in candidate_models if m["id"] == mid), mid)
            console.print(f"  [yellow]{model_name}[/]: [dim]{err}[/]")
        console.print()

    if mode == Mode.local:
        api_models = [m for m in all_models if m["type"] in ("anthropic_api", "google_api")]
        if api_models:
            names = ", ".join(m["name"] for m in api_models)
            console.print(f"[dim]ℹ️  API models not shown ({names}). Use [bold]--mode full[/] with API keys.[/]")
    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
