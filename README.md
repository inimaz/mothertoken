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

### 🧭 List models
See which model IDs can be used with `--model`.
```bash
mothertoken models

# Only local tokenizers
mothertoken models --local-only
```

### 🔤 Tokenize exact text
Count tokens for exact text using local tokenizers. Add `--language` to estimate the English-equivalent token count from the benchmark multiplier.
```bash
mothertoken tokenize "Hola Mundo" --language es

# Check one model
mothertoken tokenize "Hello" --model gpt-4o

# Estimate the English-equivalent count for a known language
mothertoken tokenize "مرحبا بالعالم" --language ar --model gpt-4o

# Compare against your own English translation
mothertoken tokenize "مرحبا بالعالم" --language ar --english-text "Hello world"

# Tokenize a file
mothertoken tokenize --file prompt.txt

# Compare translated files
mothertoken tokenize --file prompt.ar.txt --language ar --english-file prompt.en.txt
```

---

## Researcher Workflow

Benchmark regeneration and model-extension docs live in [`docs/benchmarking.md`](docs/benchmarking.md).

## License
MIT
