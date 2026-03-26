# MotherToken — Project Roadmap

> *Every model has a native tongue. The question is whether yours matches.*

A benchmarking tool and article exploring how tokenizer design creates silent cost, quality, and carbon inequities — for any language that doesn't match a model's training data dominance.

---

## The Core Insight

Tokenizer efficiency is not just a billing quirk — it is a proxy for how well a model understands your language. The cost gap and the quality gap are the same gap:

- Underrepresented language in training data
- → Tokenizer fragments it inefficiently
- → Model reasons over pieces, not concepts
- → Output and translation quality degrades
- → Users pay more, get less context, and emit more CO₂

**This is bidirectional.** A model trained primarily on Chinese (e.g. Qwen) will tokenize Chinese efficiently — but English becomes the relatively expensive language. GPT-4o is last for Chinese users; Qwen is last for English users. There is no universally best model — only the best model *for your language*.

The ranking flips depending on whose internet the model grew up on.

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
  - **Fair cost delta** — overpayment vs theoretical efficient baseline
- [x] Confirm project name: `mothertoken` — clean on PyPI, no conflict on GitHub
- [x] Pricing data source confirmed: LiteLLM `model_prices_and_context_window.json` (MIT, actively maintained, pin commit hash for reproducibility)
- [x] `tokencost` by AgentOps evaluated and rejected — bot-maintained pricing but thin human oversight, and uses tiktoken under the hood which only tokenizes OpenAI models accurately

### Key findings

- Petrov et al. is the canonical paper — seminal, NeurIPS 2023, already uses FLORES-200
- Follow-on work exists for Thai, Hungarian, Portuguese, French, Ukrainian — language-specific studies but no unified practitioner tool
- No existing PyPI package covers the full language × model efficiency benchmark
- The bidirectional insight (ranking flips per language) is not prominently surfaced in existing tools
- The carbon angle (tokenizer inefficiency → more compute → more CO₂) is not covered anywhere

---

## Phase 1 — Benchmark Dataset

**Goal:** Produce a static, versioned benchmark table (language × model → metrics).

### Tokenizer coverage strategy

| Model | Tokenizer access | Approach |
|---|---|---|
| GPT-4o | `cl100k_base` via tiktoken | Local, free, exact |
| LLaMA 3 | Public on Hugging Face | Local, free, exact |
| Mistral, Qwen, Gemma | Public on Hugging Face | Local, free, exact |
| Claude | Closed — count_tokens API | API call, cached |
| Gemini | Closed — count_tokens API | API call, cached |

### Pricing strategy

- Use **LiteLLM's `model_prices_and_context_window.json`** (MIT licensed, actively maintained)
- Pin a specific commit hash at benchmark generation time — pricing changes don't silently alter results
- Update the pinned version deliberately with each new benchmark release

### Outputs

- `benchmark.json` — versioned snapshot of all metrics across models and languages
- Scripts to regenerate when models or pricing changes (e.g., via a scheduled GitHub Actions workflow that pulls the latest pricing from LiteLLM and opens a PR)

---

## Phase 2 — The CLI Tool (`mothertoken`)

**Goal:** Let developers analyze their own text or prompts against the benchmark.

### Two modes

```
# Local mode — public tokenizers only, instant, no API keys needed
$ mothertoken analyze --file prompt.txt --languages pt,ar,th --mode local

# Full mode — includes closed models via API
# (Note: Requires user to provide API keys locally, e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
$ mothertoken analyze --file prompt.txt --languages pt,ar,th --mode full
```

### Example output

```
📊 Results for prompt.txt (2,400 characters)

Model          Language   Tokens   Cost/1M chars   vs English
──────────────────────────────────────────────────────────────
gpt-4o         English      580      $0.037          1.0x
gpt-4o         Portuguese   740      $0.047          1.27x
gpt-4o         Arabic      1630      $0.104          2.8x
claude-sonnet  English      560      $0.021          1.0x
claude-sonnet  Arabic      1480      $0.056          2.6x
qwen-2.5       Arabic       740      $0.028          1.3x  ← best for Arabic

💡 For Arabic users, Qwen 2.5 is 2.2x more efficient than GPT-4o.
💸 At 2M chars/day, switching models saves ~$55/day (~$20k/year).
```

### Additional commands

```
# Compare models for a specific language
$ mothertoken compare --language ar

# Generate a shareable HTML report
$ mothertoken report --corpus flores200 --output report.html

# Check a single term
$ mothertoken token --text "ChatGPT" --model gpt-4o
```

### Technical stack (proposed)

- Python CLI with `typer` or `click`
- `tiktoken` for GPT models
- `transformers` for open models
- Anthropic + Google SDKs for closed model token counting
- Results cached locally to avoid redundant API calls
- Packaged on PyPI

---

## Phase 3 — The Web Tool

**Goal:** Zero-friction, shareable page. Land on it → instantly know if your language is expensive on the model you're using.

### The locale-detection page

Browser locale detected automatically → instant ranking of models for your language. The ranking is different for every language — including English, which fares poorly on Chinese-optimized models like Qwen.

### The cost calculator (no backend needed)

Instead of translating arbitrary user text, use precomputed efficiency ratios applied to user-input numbers:

```
My language:        [Thai       ▾]
Messages per day:   [1000       ]
Avg message length: [200 chars  ]

Model          Monthly cost    Fair cost*    You overpay
──────────────────────────────────────────────────────────
Qwen 2.5       $12             $9            +$3   (+18%)
GPT-4o         $31             $11           +$20  (+180%)
Claude Sonnet  $28             $10           +$18  (+160%)

* what you'd pay if your language were tokenized as efficiently
  as the model's dominant training language
```

"Fair cost" makes the inequality concrete and personal — the dollar overpayment is more visceral than a ratio.

### Tech stack

- **Astro** — static site generator, zero JS by default, perfect for a content-heavy benchmark page with minimal interactivity
- Benchmark data served as a static JSON — no backend needed for the locale page
- Interactive cost calculator as an Astro island (minimal JS, hydrated on demand)
- Pricing data sourced from LiteLLM's `model_prices_and_context_window.json` (MIT, pinned at build time)
- Deployable on Cloudflare Pages or Vercel — free tier sufficient

---

## Phase 4 — The Article

**Goal:** Publish a developer-facing piece that establishes the project and drives adoption.

### Thesis

> Every model has a native tongue — the language its tokenizer was optimized for. If yours matches, you get efficient, affordable, high-quality responses. If it doesn't, you pay more, get less context, and receive subtly worse answers. This is not a bug. Here is how to measure it, and how to pick the right model for your language.

### Angle

Not an academic fairness paper — a practical, data-driven piece for engineers and CTOs making model selection decisions.

### Outline

1. What is a tokenizer and why does it matter
2. The compression ratio as a proxy for training data representation
3. The vicious cycle: underrepresentation → fragmentation → worse quality → worse translations
4. The twist: it's bidirectional — English is expensive on Qwen, Chinese is expensive on GPT-4o
5. Benchmark results across 10 languages × 6 models — the ranking flips
6. The carbon dimension — more tokens = more compute = more CO₂ (CodeCarbon angle)
7. What you can do: model selection, the tool, prompt language strategies
8. What needs to change at the infrastructure level

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
- Quantify this using CodeCarbon methodology — measure energy per token, multiply by inefficiency ratio

This connects the project to existing open source credibility and makes it genuinely novel.

---

## Benchmark Integrity

A known risk in NLP benchmarking is **benchmark contamination** — test sentences leaking into future training data, making models score artificially well on that specific set.

### Why the risk is lower for tokenization benchmarks

Tokenizer efficiency is determined at training time and frozen into the vocabulary. A model cannot "overfit" its tokenizer to FLORES by seeing FLORES sentences at inference time — the tokenizer either fragments a word efficiently or it doesn't, regardless of prior exposure. This is fundamentally different from comprehension or translation benchmarks where contamination directly inflates scores.

Running FLORES sentences through a token counting API does not feed model training pipelines.

### The real risk: plain text crawling

The concern FLORES maintainers themselves flagged is sentences ending up as plain text in future web crawls, inflating apparent multilingual ability on that specific sentence set for future model versions.

### Mitigations

**Never expose raw sentences publicly** — the web page, CLI output, and any public artifacts only ever show aggregated metrics (fertility scores, chars/token, RTC values). Raw FLORES sentences never leave the researcher's local machine. This is consistent with FLORES maintainers' own request not to re-host plain text.

**Use FLORES splits deliberately** — FLORES has dev and test splits. Use dev for the public benchmark, keep test as a private held-out validation set to occasionally check for drift across model versions.

**Rotate corpora over time** — v1 uses FLORES-200, future versions can introduce additional corpora so the benchmark stays fresh and harder to game.

**Document the methodology openly** — publish the benchmark scripts so results are reproducible, without publishing the sentences themselves.

---

## Open Questions

- [x] Domain: `mothertoken.dev` (better aligns with developer tools and the article focus)
- [ ] Whether to frame primarily as a **cost tool** (developer audience) or **fairness tool** (researcher/journalist audience) — probably both, with different entry points
- [ ] Carbon methodology: estimate from token count alone, or instrument actual inference?
- [ ] v2 consideration: parametrized baseline for researchers (compare against any language, not just English)

---

## Success Metrics

- CLI tool (`mothertoken`) published on PyPI with measurable usage (e.g., 500+ downloads/month)
- Article published and reaches a wide audience (e.g., front page of HN, top of Dev.to)
- Web page shareable enough to spread organically ("I just found out GPT-4 is terrible for my language" — or "I just found out Qwen is terrible for English")
- Conference talk accepted (e.g., EuroPython, PyCon)
- Potential: cited by a paper or picked up by AI fairness researchers