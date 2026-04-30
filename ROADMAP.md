# MotherToken — Project Roadmap

> *Every model has a native tongue. The question is whether yours matches.*

A benchmarking tool and article exploring how tokenizer design creates silent efficiency, quality, and carbon inequities — for any language that doesn't match a model's training data dominance.

---

## The Core Insight

Tokenizer efficiency is not a billing quirk — it is a proxy for how well a model understands your language. The efficiency gap and the quality gap are the same gap:

- Underrepresented language in training data
- → Tokenizer fragments it inefficiently
- → Model reasons over pieces, not concepts
- → Output and translation quality degrades
- → Users get less context, worse answers, and higher CO₂ per useful output

**This is bidirectional.** A model trained primarily on Chinese (e.g. Qwen) will tokenize Chinese efficiently — but English becomes the fragmented language. GPT-4o is last for Chinese users; Qwen is last for English users. There is no universally best model — only the best model *for your language*.

The ranking flips depending on whose internet the model grew up on.

**Pricing data is out of scope — but the cost argument is not.** RTC is a multiplier the user applies to their own situation. If Arabic needs 2.8x more tokens than English on GPT-4o, then whatever you pay per token, Arabic costs you 2.8x more. That's true regardless of the current price per token, whether you use the API or self-host, and whether prices change next month. The multiplier is the durable fact. The dollar amount is the user's own math to do.

---

## Minimum Publishable Release — v0.1

**Goal:** Publish the smallest version that is genuinely useful: a data-backed CLI for ranking tokenizer efficiency by language and checking exact text with local tokenizers.

### Product promise

> Mothertoken helps developers see which model tokenizers are efficient for which languages. It ships a precomputed FLORES+ benchmark and a CLI for ranking models or checking your own text against local tokenizers.

Avoid promising provider billing parity, fairness diagnostics, CI regression checks, custom datasets, CSV/Markdown reports, or provider drift validation until those features exist.

### Required before publishing

- [ ] **Package benchmark data correctly**
  - Ensure `data/benchmark.json` and `data/models.yaml` are included in the built wheel.
  - Add the needed `pyproject.toml` package-data / force-include config.
  - Verify with `uv build` and install the built wheel into a clean temporary environment.

- [x] **Ship one clear CLI workflow**
  - `mothertoken rank ar`
  - `mothertoken models`
  - `mothertoken models --local-only`
  - `mothertoken tokenize "Hola mundo" --language es --model gpt-4o`
  - `mothertoken tokenize --file prompt.txt --language ar`
  - Implemented in `src/mothertoken/cli/app.py` and covered by CLI tests.

- [x] **Make API-backed models non-confusing**
  - `tokenize` defaults to local tokenizers.
  - `tokenize --include-api` includes API-backed provider counters when keys are configured.
  - Selecting an API-backed model without `--include-api` gives an explicit opt-in message.

- [ ] **Tighten README around what exists**
  - Describe the package as a precomputed multilingual tokenizer-efficiency benchmark plus local tokenizer CLI.
  - Document only the supported commands.
  - Keep custom corpora, reports, CI, and provider parity as roadmap items.

- [ ] **Add publish metadata**
  - Add or confirm `LICENSE` file.
  - Add PyPI-ready project metadata as needed: authors, project URLs, classifiers, keywords.
  - Replace clone-first installation docs with `pip install mothertoken` once published.

- [ ] **Add installed-package smoke test**
  - Build the wheel.
  - Install into a clean environment.
  - Run:

```bash
mothertoken rank ar
mothertoken models --local-only
mothertoken tokenize "Hola mundo" --language es --model gpt-4o
```

### Explicitly defer

- `mothertoken compare`
- `mothertoken report`
- `mothertoken benchmark` as an integrated Typer subcommand
- User-provided corpus benchmarking
- CSV/Markdown export
- CI budget/regression checks
- Provider parity and tokenizer drift validation
- Persistent cache for paid provider token counts
- Direct SentencePiece backend

---

## Phase 0 — Research & Validation ✅ COMPLETE

**Goal:** Confirm novelty, define scope, establish credibility baseline.

- [x] Map existing work (Petrov et al. 2023, NeurIPS 2023, `tokenization-fairness` repo)
- [x] Identify the gap: research exists, practitioner-facing tool does not
- [x] Confirm FLORES-200 as the benchmark corpus (1,012 sentences × 200 languages, CC BY-SA 4.0)
- [x] Define key metrics:
  - **Fertility score** — tokens per word
  - **RTC (Relative Tokenization Cost)** — fertility vs English on the same model (v1 baseline; v2 may parametrize)
  - **Chars/token** — compression efficiency
  - ~~**Fair cost delta**~~ — replaced by cost multiplier framing (RTC × user's own token price)
- [x] Confirm project name: `mothertoken` — clean on PyPI, no conflict on GitHub
- [x] Pricing data maintenance is out of scope — RTC is the durable multiplier, users apply it to their own costs
- [x] `tokencost` by AgentOps evaluated and rejected — maintaining live prices is not mothertoken's responsibility

### Key findings

- Petrov et al. is the canonical paper — seminal, NeurIPS 2023, already uses FLORES-200
- Follow-on work exists for Thai, Hungarian, Portuguese, French, Ukrainian — language-specific studies but no unified practitioner tool
- No existing PyPI package covers the full language × model efficiency benchmark
- The bidirectional insight (ranking flips per language) is not prominently surfaced in existing tools
- The carbon angle (tokenizer inefficiency → more compute → more CO₂) is not covered anywhere

---

## Phase 1 — Benchmark Dataset

**Goal:** Produce a static, versioned benchmark table (language × model → tokenization metrics only, no pricing).

### Tokenizer coverage strategy

| Model | Tokenizer access | Approach |
|---|---|---|
| GPT-4o | `o200k_base` via tiktoken | Local, free, exact |
| GPT-4 | `cl100k_base` via tiktoken | Local, free, exact |
| LLaMA 3 | Public on Hugging Face | Local, free, exact |
| Mistral, Qwen, Gemma | Public on Hugging Face | Local, free, exact |
| Claude | Closed — count_tokens API | API call, cached |
| Gemini | Closed — count_tokens API | API call, cached |

### What the benchmark measures

Only tokenization efficiency — no pricing:

- **Fertility** — tokens per word for each language × model pair
- **Chars/token** — compression efficiency
- **RTC** — relative to English on the same model
- **Context efficiency** — how much of a typical context window a given language consumes vs English

### HuggingFace access requirements

- FLORES+ (`openlanguagedata/flores_plus`) — gated, requires accepting terms, standard read token sufficient
- LLaMA 3 — gated, requires Meta approval (hours to days)
- Gemma 2 — gated, requires Google approval
- Mistral, Qwen — public, no approval needed
- HF token must have "Read access to gated repos" permission enabled

### Outputs

- `benchmark.json` — versioned snapshot of tokenization metrics across models and languages, no pricing data
- Scripts to regenerate when new models or tokenizers are released

### Status

- [x] `data/benchmark.json` exists as a versioned aggregate snapshot.
- [x] `data/models.yaml` exists as the model registry.
- [x] `mothertoken-benchmark` entry point exists for regeneration.
- [x] Benchmark docs exist in `docs/benchmarking.md`.
- [x] Core metrics implemented: fertility, chars/token, RTC.
- [x] Local tokenizer backends implemented: `tiktoken`, Hugging Face.
- [x] API tokenizer backends implemented: Anthropic, Google.
- [ ] API token count cache is persistent across runs. Current cache is in-memory only.
- [ ] Context efficiency is stored directly in `benchmark.json`. Current tools derive cost/context implications from RTC.
- [ ] Benchmark data is confirmed to ship inside the built wheel.

---

## Phase 2 — The CLI Tool (`mothertoken`)

**Goal:** Let developers and researchers analyze tokenization efficiency for their own text.

### Two user personas

- **Developer** — wants to inspect their own prompts and understand tokenization impact on context window
- **Researcher** — wants to reproduce or extend the benchmark across new models or languages

### Subcommand structure

```
mothertoken rank         # implemented — rank models for one benchmark language
mothertoken models       # implemented — list supported models and tokenizer access
mothertoken tokenize     # implemented — count exact text with local tokenizers
mothertoken-benchmark    # implemented — researcher benchmark runner

mothertoken analyze      # deferred — broader prompt/corpus analysis workflow
mothertoken benchmark    # deferred — integrated Typer subcommand wrapper
mothertoken compare      # deferred — compare two languages head to head on a model
mothertoken report       # deferred — shareable report generation
```

### Two modes

```
# Local mode — public tokenizers only, instant, no API keys needed
$ mothertoken analyze --file prompt.txt --languages pt,ar,th --mode local

# Full mode — includes closed models via API
$ mothertoken analyze --file prompt.txt --languages pt,ar,th --mode full
```

### Example output

```
📊 Results for prompt.txt (2,400 characters)

Model          Language     Tokens   Chars/token   vs English
──────────────────────────────────────────────────────────────
gpt-4o         English        580       4.1          1.0x
gpt-4o         Portuguese     740       3.2          1.3x
gpt-4o         Arabic        1630       1.5          2.8x
claude-sonnet  English        560       4.3          1.0x
claude-sonnet  Arabic        1480       1.6          2.7x
qwen-2.5       Arabic         740       3.2          1.3x  ← most efficient for Arabic

💡 Arabic needs 2.8x more tokens than English on GPT-4o.
📐 That's 2.8x less effective context window for your Arabic users.
💸 Whatever you pay per token, Arabic costs you 2.8x more on GPT-4o.
   → Switching to Qwen 2.5 reduces that to 1.3x.
```

### Additional commands

```
# Compare languages head to head
$ mothertoken compare --languages en,ar --model gpt-4o

# Run the full FLORES benchmark (researcher mode)
$ mothertoken benchmark --setup        # downloads FLORES-200
$ mothertoken benchmark --language ar  # run for one language
$ mothertoken benchmark --all          # full run

# Generate a shareable report
$ mothertoken report --corpus flores200 --output report.html

# Check a single term
$ mothertoken token --text "ChatGPT" --model gpt-4o
```

### Status

- [x] `rank` implemented with language aliases and precomputed benchmark data.
- [x] `models` implemented with `--local-only`.
- [x] `tokenize` implemented for exact text and files.
- [x] `tokenize --language` implemented for benchmark English-equivalent estimates.
- [x] `tokenize --english-text` and `--english-file` implemented for paired translation comparison.
- [x] API-backed models are opt-in via `tokenize --include-api` instead of appearing broken by default.
- [x] Separate `mothertoken-benchmark` command exists for researcher regeneration.
- [ ] Integrated `mothertoken benchmark` subcommand exists.
- [ ] `compare`, `report`, and legacy `token` commands exist.
- [x] Provider-backed tokenization opt-in exists in everyday CLI via `--include-api`.
- [ ] Provider parity / local-vs-API drift validation exists.
- [ ] CI-friendly threshold command exists.
- [ ] Machine-readable CLI output modes exist.

### API key behaviour

```
$ mothertoken analyze --file prompt.txt --language ar

✓ GPT-4o        (public tokenizer)
✓ LLaMA 3       (public tokenizer)
✓ Mistral       (public tokenizer)
✓ Qwen 2.5      (public tokenizer)
⚠ Claude Sonnet (no API key — skipped)
⚠ Gemini Pro    (no API key — skipped)

→ Run `mothertoken models setup` to add API keys for full results
```

Useful immediately without keys. Never fails silently.

### Technical stack

- Python CLI with `typer`
- `tiktoken` for GPT models
- `transformers` for open models
- Anthropic + Google SDKs for closed model token counting
- Results cached locally to avoid redundant API calls
- Packaged on PyPI

Current caveat: tokenizer results are cached only for the current process. Persistent API caching remains a v2 feature.

---

## Phase 3 — The Web Tool

**Goal:** Zero-friction, shareable page. Land on it → instantly know how efficiently your language is tokenized on each model.

### The locale-detection page

Browser locale detected automatically → instant ranking of models for your language. The ranking is different for every language — including English, which fragments on Chinese-optimized models like Qwen.

### The context efficiency + cost multiplier calculator (no backend, no pricing data)

Two complementary outputs from the same RTC number — both derived from precomputed efficiency ratios, no pricing tables needed:

```
My language:          [Thai          ▾]
My context window:    [128,000 tokens]

Model           Usable context   vs English   Cost multiplier
──────────────────────────────────────────────────────────────
Qwen 2.5        98,000 tokens     -5%         1.3x your English cost
GPT-4o          45,000 tokens     -65%        2.8x your English cost
Claude Sonnet   46,000 tokens     -64%        2.7x your English cost
Mistral         41,000 tokens     -68%        3.1x your English cost
```

"You lose 65% of your context window and pay 2.8x more — just by speaking Thai on GPT-4o." The multiplier is the user's own math to apply to their own bill. mothertoken never touches a pricing table.

### Tech stack

- **Astro** — static site generator, zero JS by default
- Benchmark data served as static JSON, baked in at build time
- Interactive calculator as an Astro island (minimal JS, hydrated on demand)
- No pricing data anywhere in the stack
- Deployable on Cloudflare Pages or Vercel — free tier sufficient
- Domain: `mothertoken.dev` — buy when Phase 3 begins, not before

### Status

- [x] Astro web app exists under `web/`.
- [x] Home page exists with browser-locale detection.
- [x] Benchmark explorer page exists.
- [x] CLI documentation page exists.
- [x] Methodology page exists.
- [x] Web data is generated from root `data/benchmark.json`.
- [x] Cost multiplier calculator exists.
- [ ] Context-window calculator exists as a first-class control.
- [ ] Production domain is configured.
- [ ] Deployment target is configured and documented.

---

## Phase 4 — The Article

**Goal:** Publish a developer-facing piece that establishes the project and drives adoption.

### Thesis

> Every model has a native tongue — the language its tokenizer was optimized for. If yours matches, you get efficient responses and full use of your context window. If it doesn't, your messages fragment into pieces, your context shrinks, the model understands you less well, and you pay a multiplier on every token. This is not a billing quirk. It is an architectural fact. Here is how to measure it.

### Angle

Not an academic fairness paper — a practical, data-driven piece for engineers making model selection decisions. Pricing data is out of scope, but the cost argument is made through RTC multipliers — durable, maintenance-free, and universally applicable regardless of current prices or whether you self-host.

### Outline

1. What is a tokenizer and why does it matter
2. The compression ratio as a proxy for training data representation
3. The vicious cycle: underrepresentation → fragmentation → worse quality → worse translations
4. The twist: it's bidirectional — English fragments on Qwen, Chinese fragments on GPT-4o
5. Benchmark results across 15 languages × 6 models — the ranking flips
6. The context window consequence — you're not just paying more, you're getting less
7. The carbon dimension — more tokens = more compute = more CO₂ (CodeCarbon angle)
8. What you can do: model selection, the tool
9. What needs to change at the infrastructure level

### Publication targets (in order of preference)

- Personal blog (inimaz.com) — establish canonical version
- dev.to / Towards Data Science — reach developer audience
- Conference talk — EuroPython, PyCon, or an AI fairness track

---

## The Carbon Angle (CodeCarbon tie-in)

This is the differentiating dimension not covered by existing work:

- Tokenizer inefficiency → more tokens → more inference compute → more CO₂
- Non-English speakers have a larger carbon footprint **per useful unit of output**
- The "AI for everyone" narrative has a hidden environmental justice dimension
- Quantify using CodeCarbon methodology — measure energy per token, multiply by RTC ratio

This connects mothertoken to existing open source credibility and makes it genuinely novel.

---

## Benchmark Integrity

A known risk in NLP benchmarking is **benchmark contamination** — test sentences leaking into future training data, making models score artificially well on that specific set.

### Why the risk is lower for tokenization benchmarks

Tokenizer efficiency is determined at training time and frozen into the vocabulary. A model cannot "overfit" its tokenizer to FLORES by seeing FLORES sentences at inference time — the tokenizer either fragments a word efficiently or it doesn't, regardless of prior exposure. This is fundamentally different from comprehension or translation benchmarks where contamination directly inflates scores.

Running FLORES sentences through a token counting API does not feed model training pipelines.

### The real risk: plain text crawling

The concern FLORES maintainers themselves flagged is sentences ending up as plain text in future web crawls, inflating apparent multilingual ability on that specific sentence set for future model versions.

### Mitigations

**Never expose raw sentences publicly** — the web page, CLI output, and any public artifacts only ever show aggregated metrics (fertility, chars/token, RTC). Raw FLORES sentences never leave the researcher's local machine.

**Use FLORES splits deliberately** — use `dev` for the public benchmark, keep `devtest` as a private held-out validation set to check for drift across model versions.

**Rotate corpora over time** — v1 uses FLORES-200, future versions can introduce additional corpora.

**Document the methodology openly** — publish the benchmark scripts so results are reproducible, without publishing the sentences themselves.

---

## Open Questions

- [ ] Whether to frame primarily as an **efficiency tool** (developer audience) or **fairness tool** (researcher/journalist audience) — probably both, with different entry points
- [ ] Carbon methodology: estimate from token count alone, or instrument actual inference?
- [ ] v2 consideration: parametrized baseline for researchers (compare against any language, not just English)
- [ ] Domain: buy `mothertoken.dev` at the start of Phase 3

---

## Success Metrics

- CLI tool (`mothertoken`) published on PyPI with real usage
- Article published and referenced by others
- Web page shareable enough to spread organically ("I just found out I lose 65% of my context window and pay 2.8x more speaking Thai on GPT-4o")
- Conference talk accepted
- Potential: cited by a paper or picked up by AI fairness researchers
