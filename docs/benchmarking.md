# Benchmarking

This document is for maintainers and researchers who want to regenerate or extend the benchmark dataset.

Most users do not need this workflow. The regular CLI reads the precomputed `data/benchmark.json` file.

## What The Benchmark Does

The benchmark computes aggregate tokenizer-efficiency metrics across languages and models using FLORES+.

Output is written to `data/benchmark.json`.

The output contains only aggregate metrics. Raw FLORES+ sentences are never written to disk or included in public artifacts.

## Setup

FLORES+ is hosted on Hugging Face and may require login.

```bash
uv run python -c "import huggingface_hub; huggingface_hub.login()"
```

Some model tokenizers are gated on Hugging Face. Make sure your token has access to any gated repos you want to benchmark.

## Run The Benchmark

Verify setup without loading FLORES or real tokenizers:

```bash
uv run mothertoken-benchmark --dry-run
```

Run the default benchmark:

```bash
uv run mothertoken-benchmark
```

Run a smaller benchmark:

```bash
uv run mothertoken-benchmark --languages arb_Arab,spa_Latn --models gpt-4o,qwen2.5
```

Run public local tokenizers only:

```bash
uv run mothertoken-benchmark --models gpt-4o,gpt-4,llama3,mistral,qwen2.5,gemma2
```

## API-Backed Tokenizers

Closed tokenizer APIs require provider keys:

```bash
export ANTHROPIC_API_KEY=your_key
export GOOGLE_API_KEY=your_key
```

If a key is missing, the benchmark runner records the error and skips that model.

## Add A Model

Add an entry to `data/models.yaml`:

```yaml
  - id: my-model
    name: "My Model"
    type: huggingface       # tiktoken | huggingface | anthropic_api | google_api
    ref: org/model-name     # tokenizer ref
    api_key_env: null       # env var name, or null if no key needed
```

Then run a focused benchmark before regenerating everything:

```bash
uv run mothertoken-benchmark --languages eng_Latn,arb_Arab --models my-model
```

## Output Contract

`data/benchmark.json` should remain a versioned aggregate dataset with:

- benchmark metadata
- model metadata
- per-language, per-model aggregate metrics
- tokenizer errors

It should not contain raw corpus text.
