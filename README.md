# mothertoken — benchmark

Generates `data/benchmark.json` — the tokenization efficiency dataset
that powers the mothertoken web page and CLI tool.

## Setup

```bash
# Install dependencies
uv sync

# Log into HuggingFace (required for FLORES+)
python -c "import huggingface_hub; huggingface_hub.login()"
```

## Running

```bash
# Verify setup without loading real data
uv run src/mothertoken/benchmark/runner.py --dry-run

# Run for a single language (fast, good for testing)
uv run src/mothertoken/benchmark/runner.py --languages tha_Thai --models gpt-4o,llama3

# Full benchmark — all languages, public tokenizers only (no API keys needed)
uv run src/mothertoken/benchmark/runner.py --models gpt-4o,gpt-4,llama3,mistral,qwen2.5,gemma2

# Full benchmark — include closed models (requires API keys)
export ANTHROPIC_API_KEY=your_key
export GOOGLE_API_KEY=your_key
uv run src/mothertoken/benchmark/runner.py
```

## Output

Results are saved to `data/benchmark.json`. The file contains only
aggregated metrics — raw FLORES+ sentences are never written to disk
or included in any public artifact.

## Reproducibility

The benchmark is versioned by date. To reproduce a specific run,
use the same FLORES+ dataset version and model tokenizer versions
listed in the output JSON.

## Adding a new model

Add an entry to `data/models.yaml`:

```yaml
  - id: my-model
    name: "My Model"
    type: huggingface       # tiktoken | huggingface | anthropic_api | google_api
    ref: org/model-name     # tokenizer ref
    api_key_env: null       # env var name, or null if no key needed
```

## Benchmark integrity

- Always use the `dev` split for the public benchmark
- Keep the `devtest` split as a private held-out validation set
- Never commit raw FLORES+ sentences to any public repository
- Pin `huggingface_hub` and tokenizer versions for reproducible runs
