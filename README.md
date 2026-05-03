<p align="center">
  <img src="web/public/favicon.svg" alt="mothertoken logo" width="96" height="96">
</p>

# mothertoken

> *Every model has a native tongue. The question is whether yours matches.*

Toolkit for comparing tokenizer efficiency across languages, model families, and user-supplied Hugging Face refs.

> [!NOTE]
> **Alpha release:** the bundled benchmark is curated and representative, not exhaustive. Use direct Hugging Face refs when you want to compare tokenizers outside the starter set.

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
```

### 🔤 Tokenize exact text
Count tokens for exact text using local tokenizers by default. Add `--language` to estimate the English-equivalent token count from the benchmark multiplier.
```bash
mothertoken tokenize "Hola Mundo" --language es

# Check one model
mothertoken tokenize "Hello" --model gpt-4o

# Check a Hugging Face model/tokenizer ref directly
mothertoken tokenize "Hello" --model Qwen/Qwen3-0.6B

# Estimate the English-equivalent count for a known language
mothertoken tokenize "مرحبا بالعالم" --language ar --model gpt-4o

# Compare against your own English translation
mothertoken tokenize "مرحبا بالعالم" --language ar --english-text "Hello world"

# Tokenize a file
mothertoken tokenize --file prompt.txt

# Compare translated files
mothertoken tokenize --file prompt.ar.txt --language ar --english-file prompt.en.txt
```

### ⚖️ Compare selected tokenizers
Compare aliases from `mothertoken list` with direct Hugging Face refs. This is the main workflow when you care about a specific set of models.
```bash
mothertoken compare "Travesura realizada" \
  --model gpt-4o \
  --model Qwen/Qwen3-0.6B \
  --model mistralai/Mistral-7B-v0.1

mothertoken compare --file prompt.txt \
  --model mistralai/Mistral-7B-v0.1 \
  --model deepseek-ai/DeepSeek-V4-Pro
```

---

## Researcher Workflow

Benchmark regeneration and model-extension docs live in [`docs/benchmarking.md`](docs/benchmarking.md).

You can also benchmark a direct Hugging Face ref without adding it to `tokenizers.yaml`:

```bash
uv run mothertoken-benchmark --languages eng_Latn,arb_Arab --models Qwen/Qwen3-0.6B
```

## License
MIT
