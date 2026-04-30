# mothertoken

> *Every model has a native tongue. The question is whether yours matches.*

Benchmarking tool and dataset exploring how tokenizer design creates silent efficiency, quality, and carbon inequities for non-English languages.

## Installation

```bash
# Clone the repo
git clone https://github.com/inimaz/mothertoken
cd mothertoken

# Install dependencies and the package in editable mode
uv sync
uv pip install -e .
```

## CLI Usage

The `mothertoken` command is available after installation.

### 📊 Rank models for a language
Rank all models for a specific language using the precomputed benchmark data.
```bash
mothertoken rank spanish

# Raw FLORES+ codes still work
mothertoken rank spa_Latn
```

### 🔤 Tokenize exact text
Count tokens for exact text using local tokenizers.
```bash
mothertoken tokenize "ChatGPT"

# Check one model
mothertoken tokenize "ChatGPT" --model gpt-4o

# Tokenize a file
mothertoken tokenize --file prompt.txt
```

---

## 🛠️ Regenerating the Benchmark

If you want to contribute new models or regenerate the efficiency metrics, you can run the benchmark generator.

### Setup
```bash
# Log into HuggingFace (required for FLORES+ datasets)
uv run python -c "import huggingface_hub; huggingface_hub.login()"
```

### Running the generator
```bash
# Verify setup (fast, no real data)
uv run mothertoken-benchmark --dry-run

# Run full benchmark (public tokenizers)
uv run mothertoken-benchmark --models gpt-4o,gpt-4,llama3,mistral,qwen2.5,gemma2

# Run full benchmark (including closed models via API)
uv run mothertoken-benchmark
```
*Note: `mothertoken-benchmark` is an alias for the internal `runner.py` script.*

## Output
Results are saved to `data/benchmark.json`. The file contains aggregated metrics; raw FLORES+ sentences are never written to disk or included in public artifacts.

## Adding a New Model
Add an entry to `data/models.yaml`:
```yaml
  - id: my-model
    name: "My Model"
    type: huggingface       # tiktoken | huggingface | anthropic_api | google_api
    ref: org/model-name     # tokenizer ref
    api_key_env: null       # env var name, or null if no key needed
```

## License
MIT
