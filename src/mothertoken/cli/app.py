"""
mothertoken — cli/app.py

Typer-based CLI for the mothertoken tool.

Commands:
    rank      — rank all models for a given language from benchmark data
    list      — list supported tokenizers and counter sources
    compare   — compare user-selected tokenizer aliases or Hugging Face refs
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

from mothertoken.core.registry import LOCAL_MODEL_TYPES, ModelType

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


def _load_tokenizers_config() -> list[dict]:
    """Load the tokenizer registry."""
    from mothertoken.core.resources import load_tokenizers_config

    cfg = load_tokenizers_config()
    return cfg.get("tokenizers", [])


def _is_local_tokenizer(model: dict) -> bool:
    return model["type"] in LOCAL_MODEL_TYPES


def _public_tokenizers(tokenizers_cfg: list[dict]) -> list[dict]:
    """Tokenizers exposed through the public user CLI."""
    return [t for t in tokenizers_cfg if _is_local_tokenizer(t)]


def _looks_like_hf_ref(model: str) -> bool:
    """Return whether a model string should be treated as a direct HF repo/path ref."""
    return "/" in model or Path(model).exists()


def _hf_ref_tokenizer(ref: str) -> dict:
    return {
        "id": ref,
        "name": ref,
        "provider": "huggingface",
        "type": ModelType.HUGGINGFACE.value,
        "ref": ref,
        "access": "local",
        "tokenizer_source": "huggingface",
        "verification_method": "user_supplied_ref",
        "used_by_examples": [ref],
        "api_key_env": None,
    }


def _resolve_tokenizer_selection(model: str, local_tokenizers: list[dict]) -> dict:
    """Resolve a user model value as a curated alias or direct Hugging Face ref."""
    tokenizer_cfg = next((t for t in local_tokenizers if t["id"] == model), None)
    if tokenizer_cfg is not None:
        return tokenizer_cfg
    if _looks_like_hf_ref(model):
        return _hf_ref_tokenizer(model)

    available = ", ".join(t["id"] for t in local_tokenizers)
    err_console.print(
        f"[bold red]Error:[/] Tokenizer [yellow]{model}[/] not found.\n"
        f"Use a configured alias ({available}) or a Hugging Face ref like [cyan]Qwen/Qwen3-0.6B[/]."
    )
    raise typer.Exit(code=1)


def _read_input_text(
    text: str | None,
    file: Path | None,
    *,
    text_label: str = "text",
    file_label: str = "--file",
) -> str:
    if text is None and file is None:
        err_console.print(f"[bold red]Error:[/] Provide {text_label} or [bold]{file_label}[/].")
        raise typer.Exit(code=1)
    if text is not None and file is not None:
        err_console.print(f"[bold red]Error:[/] Provide {text_label} or [bold]{file_label}[/], not both.")
        raise typer.Exit(code=1)
    if file is not None:
        if not file.exists():
            err_console.print(f"[bold red]Error:[/] File not found: {file}")
            raise typer.Exit(code=1)
        return file.read_text(encoding="utf-8")
    return text or ""


def _tokenizer_backend(model: dict) -> str:
    labels = {
        ModelType.TIKTOKEN.value: "tiktoken",
        ModelType.HUGGINGFACE.value: "Hugging Face",
        ModelType.ANTHROPIC_API.value: "Anthropic count_tokens",
        ModelType.GOOGLE_API.value: "Google count_tokens",
    }
    return f"{labels.get(model['type'], model['type'])} / {model['ref']}"


def _used_by_examples(model: dict) -> str:
    examples = model.get("used_by_examples") or []
    return ", ".join(examples) if examples else "—"


def _tokenizer_table(tokenizers_cfg: list[dict]) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("ID", no_wrap=True)
    table.add_column("Name")
    table.add_column("Used by")
    table.add_column("Counter source", overflow="fold")

    for tokenizer_cfg in sorted(tokenizers_cfg, key=lambda t: t["id"]):
        table.add_row(
            tokenizer_cfg["id"],
            tokenizer_cfg["name"],
            _used_by_examples(tokenizer_cfg),
            _tokenizer_backend(tokenizer_cfg),
        )
    return table


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------


def _show_tokenizer_ranking(language: str) -> None:
    data = _load_benchmark_or_exit()

    from mothertoken.cli.benchmark_loader import get_language_metrics, get_languages, get_model_name

    public_tokenizer_ids = {t["id"] for t in _public_tokenizers(data.get("tokenizers") or data.get("models") or [])}

    available = get_languages(data)
    language = _resolve_language(language, available)

    metrics = get_language_metrics(data, language)
    if not metrics:
        err_console.print(f"[bold red]Error:[/] No metrics found for {language}.")
        raise typer.Exit(code=1)

    # Sort by chars_per_token descending (higher = more efficient)
    rows = []
    for model_id, m in metrics.items():
        if public_tokenizer_ids and model_id not in public_tokenizer_ids:
            continue
        if "error" in m:
            continue
        rows.append((model_id, m))
    rows.sort(key=lambda r: r[1]["chars_per_token"], reverse=True)

    console.print()
    console.print(f"[bold]📊 Tokenizer ranking for[/] [cyan]{language}[/]")
    if data.get("version"):
        console.print(f"[dim]Benchmark version:[/] {data['version']}")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Tokenizer")
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

    best_tokenizer = get_model_name(data, rows[0][0])
    console.print(f"[green]💡 [bold]{best_tokenizer}[/] is the most efficient tokenizer for [cyan]{language}[/].[/]")
    console.print()


@app.command()
def rank(
    language: Annotated[str, typer.Argument(help="Language alias or FLORES+ code, e.g. ar, arabic, arb_Arab")],
):
    """Rank tokenizers by efficiency for one language."""
    _show_tokenizer_ranking(language)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command("list")
def list_tokenizers():
    """List supported tokenizers and counter sources."""
    tokenizers_cfg = _public_tokenizers(_load_tokenizers_config())

    if not tokenizers_cfg:
        err_console.print("[bold red]Error:[/] No tokenizers are configured.")
        raise typer.Exit(code=1)

    console.print()
    console.print("[bold]Local counters[/]")
    console.print(_tokenizer_table(tokenizers_cfg))
    console.print()


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


@app.command()
def tokenize(
    text: Annotated[str | None, typer.Argument(help="Text to tokenize exactly as provided")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Path to a text file to tokenize")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Tokenizer ID, e.g. gpt-4o")] = None,
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
):
    """Count tokens for exact text using local counters."""
    input_text = _read_input_text(text, file)
    if english_text is not None and english_file is not None:
        err_console.print("[bold red]Error:[/] Provide [bold]--english-text[/] or [bold]--english-file[/], not both.")
        raise typer.Exit(code=1)

    paired_english_text: str | None = None
    if english_file is not None:
        if not english_file.exists():
            err_console.print(f"[bold red]Error:[/] English file not found: {english_file}")
            raise typer.Exit(code=1)
        paired_english_text = english_file.read_text(encoding="utf-8")
    elif english_text is not None:
        paired_english_text = english_text

    local_tokenizers = _public_tokenizers(_load_tokenizers_config())

    if model is not None:
        candidate_tokenizers = [_resolve_tokenizer_selection(model, local_tokenizers)]
    else:
        candidate_tokenizers = local_tokenizers

    if not candidate_tokenizers:
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
    if resolved_language is not None or paired_english_text is not None:
        console.print()

    with console.status("Running local counters..."):
        for tokenizer_cfg in candidate_tokenizers:
            try:
                sentences = [input_text]
                if paired_english_text is not None:
                    sentences.append(paired_english_text)
                counts = tokenize_sentences(tokenizer_cfg, sentences, cache, dry_run=False)
                token_count = counts[0]
                english_token_count = counts[1] if paired_english_text is not None else None
                cpt = len(input_text) / token_count if token_count > 0 else 0.0
                results.append((tokenizer_cfg, token_count, cpt, english_token_count, None))
            except Exception as e:
                results.append((tokenizer_cfg, None, None, None, str(e)))

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Tokenizer")
    table.add_column("Tokens", justify="right")
    table.add_column("Chars/Token", justify="right")
    if resolved_language is not None:
        table.add_column("English Est.", justify="right")
        table.add_column("Vs English", justify="right")
    if paired_english_text is not None:
        table.add_column("English Tokens", justify="right")
        table.add_column("Paired Ratio", justify="right")

    sorted_results = sorted(results, key=lambda row: row[2] if row[2] is not None else -1, reverse=True)
    for tokenizer_cfg, token_count, cpt, english_token_count, error in sorted_results:
        if error is not None:
            row = [tokenizer_cfg["name"]]
            row.extend(["[red]error[/]", "—"])
            if resolved_language is not None:
                row.extend(["—", "—"])
            if paired_english_text is not None:
                row.extend(["—", "—"])
            table.add_row(*row)
            continue
        row = [tokenizer_cfg["name"]]
        row.extend([str(token_count), f"{cpt:.3f}"])
        if resolved_language is not None:
            rtc = benchmark_metrics.get(tokenizer_cfg["id"], {}).get("rtc")
            english_estimate = token_count / rtc if token_count is not None and rtc else None
            row.extend([_fmt_token_estimate(english_estimate), _fmt_rtc(rtc) if rtc else "—"])
        if paired_english_text is not None:
            paired_ratio = (
                token_count / english_token_count if token_count is not None and english_token_count else None
            )
            row.extend([str(english_token_count), _fmt_rtc(paired_ratio) if paired_ratio else "—"])
        table.add_row(*row)

    console.print(table)

    errors = [(tokenizer_cfg, error) for tokenizer_cfg, _, _, _, error in results if error is not None]
    if errors:
        console.print("\n[dim]Errors:[/]")
        for tokenizer_cfg, error in errors:
            console.print(f"  [yellow]{tokenizer_cfg['name']}[/]: [dim]{error}[/]")

    if resolved_language is not None:
        console.print("\n[dim]English Est. uses the benchmark vs-English multiplier for this language/tokenizer.[/]")
    if paired_english_text is not None:
        console.print("[dim]Paired Ratio compares your original text against the supplied English translation.[/]")

    if model is not None and errors:
        raise typer.Exit(code=1)

    console.print()


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command()
def compare(
    text: Annotated[str | None, typer.Argument(help="Text to compare exactly as provided")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Path to a text file to compare")] = None,
    models: Annotated[
        list[str] | None,
        typer.Option("--model", "-m", help="Tokenizer alias or Hugging Face ref. Repeat to compare multiple."),
    ] = None,
):
    """Compare selected tokenizer aliases or Hugging Face refs on exact text."""
    input_text = _read_input_text(text, file)
    model_values = models or []
    if not model_values:
        err_console.print(
            "[bold red]Error:[/] Provide at least one [bold]--model[/] alias or Hugging Face ref to compare."
        )
        raise typer.Exit(code=1)

    local_tokenizers = _public_tokenizers(_load_tokenizers_config())
    candidate_tokenizers = [_resolve_tokenizer_selection(model, local_tokenizers) for model in model_values]

    from mothertoken.core.tokenizers import tokenize_sentences

    cache: dict = {}
    results: list[tuple[dict, int | None, float | None, str | None]] = []

    preview = input_text.strip()[:60].replace("\n", " ")
    if len(input_text) > 60:
        preview += "..."

    console.print()
    console.print(f'[bold]Comparing:[/] [cyan]"{preview}"[/] ({len(input_text)} chars)\n')

    with console.status("Running counters..."):
        for tokenizer_cfg in candidate_tokenizers:
            try:
                token_count = tokenize_sentences(tokenizer_cfg, [input_text], cache, dry_run=False)[0]
                cpt = len(input_text) / token_count if token_count > 0 else 0.0
                results.append((tokenizer_cfg, token_count, cpt, None))
            except Exception as e:
                results.append((tokenizer_cfg, None, None, str(e)))

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Tokenizer")
    table.add_column("Ref", overflow="fold")
    table.add_column("Tokens", justify="right")
    table.add_column("Chars/Token", justify="right")

    sorted_results = sorted(results, key=lambda row: row[2] if row[2] is not None else -1, reverse=True)
    for tokenizer_cfg, token_count, cpt, error in sorted_results:
        if error is not None:
            table.add_row(tokenizer_cfg["name"], tokenizer_cfg["ref"], "[red]error[/]", "—")
            continue
        table.add_row(tokenizer_cfg["name"], tokenizer_cfg["ref"], str(token_count), f"{cpt:.3f}")

    console.print(table)

    errors = [(tokenizer_cfg, error) for tokenizer_cfg, _, _, error in results if error is not None]
    if errors:
        console.print("\n[dim]Errors:[/]")
        for tokenizer_cfg, error in errors:
            console.print(f"  [yellow]{tokenizer_cfg['name']}[/]: [dim]{error}[/]")
        raise typer.Exit(code=1)

    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
