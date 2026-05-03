# Benchmarking

This document is for maintainers and researchers who want to regenerate or extend the benchmark dataset.

Most users do not need this workflow. The regular CLI reads the precomputed `src/mothertoken/data/benchmark.json` file.

## What The Benchmark Does

The benchmark computes aggregate tokenizer-efficiency metrics across languages and models using FLORES+.

Output is written to `src/mothertoken/data/benchmark.json`.

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

## Add A Tokenizer

Add a local tokenizer entry to `src/mothertoken/data/tokenizers.yaml`:

```yaml
  - id: my-tokenizer
    name: "My tokenizer"
    provider: example
    type: huggingface       # tiktoken | huggingface
    ref: org/model-name     # tokenizer ref
    access: local
    tokenizer_source: huggingface
    verification_method: local_fixture_count
    used_by_examples:
      - My Model
    api_key_env: null
```

Then run a focused benchmark before regenerating everything:

```bash
uv run mothertoken-benchmark --languages eng_Latn,arb_Arab --models my-tokenizer
```

## Output Contract

`src/mothertoken/data/benchmark.json` should remain a versioned aggregate dataset with:

- benchmark metadata
- tokenizer metadata
- per-language, per-tokenizer aggregate metrics
- tokenizer errors

It should not contain raw corpus text.
