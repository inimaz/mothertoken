<p align="center">
  <img src="web/public/favicon.svg" alt="mothertoken logo" width="96" height="96">
</p>

# mothertoken

> *Every model has a native tongue. The question is whether yours matches.*

Benchmarking tool and dataset exploring how tokenizer design creates silent efficiency, quality, and carbon inequities for non-English languages.

> [!NOTE]
> **Alpha release:** tokenizer coverage is still expanding. The current tokenizer list is useful for comparison, but it is not complete yet.

## Installation

```bash
# Published package
pip install mothertoken
```

For local development:

```bash
git clone https://github.com/inimaz/mothertoken
cd mothertoken
uv sync
uv pip install -e .
```

## CLI Usage

The `mothertoken` command is available after installation.

### 📊 Rank tokenizers for a language
Rank supported tokenizers for a specific language using the precomputed benchmark data.
```bash
mothertoken rank spanish

# Raw FLORES+ codes still work
mothertoken rank spa_Latn
```

### 🧭 List tokenizers
See which tokenizer IDs can be used and which familiar models use them.
```bash
mothertoken list

# Only local tokenizers
mothertoken list --local-only
```

### 🔤 Tokenize exact text
Count tokens for exact text using local tokenizers by default. Add `--language` to estimate the English-equivalent token count from the benchmark multiplier.
```bash
mothertoken tokenize "Hola Mundo" --language es

# Check one model
mothertoken tokenize "Hello" --model gpt-4o

# Include API-backed provider token counters when API keys are configured
mothertoken tokenize "Hello" --include-api
mothertoken tokenize "Hello" --model claude-sonnet --include-api

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
