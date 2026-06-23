# audio-rag

A local, privacy-first **RAG (Retrieval-Augmented Generation) pipeline** over audio files.

Drop audio files in `input/`, ask questions in natural language, and get answers with precise audio timestamps — all running 100 % on your machine, no API keys required.

---

## Architecture

```
input/ audio files
    └─► Transcriber (Whisper large-v3-turbo via faster-whisper)
            └─► Chunker (overlapping text chunks with timestamp mapping)
                    └─► Embedder (sentence-transformers → ChromaDB)
                            └─► Retriever (semantic search)
                                    └─► Generator (Ollama gemma4:e2b)
                                                └─► Answer + References
```

---

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.10 | |
| [Ollama](https://ollama.ai) | Running locally at `http://localhost:11434` |
| `gemma4:e2b` model pulled in Ollama | `ollama pull gemma4:e2b` |
| CUDA (optional) | Falls back to CPU automatically |

---

## Installation

```bash
# 1. Clone / enter the project
cd audio-rag

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the Ollama model (Ollama must be running)
ollama pull gemma4:e2b
```

---

## Usage

### 1. Ingest audio files

Copy your audio files (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.mp4`) into the `input/` folder, then run:

```bash
python main.py ingest
```

Transcriptions are cached in `data/transcriptions/` — re-running skips already-transcribed files.

### 2. Query

```bash
python main.py query "Which audio files discuss Italian politics?"
```

Save the answer to a JSON file:

```bash
python main.py query "Who is the minister of economy?" --output output/answer.json
```

### 3. Ingest + Query in one step

```bash
python main.py query "main topic" --input-dir /path/to/audio/
```

### 4. LLM-as-a-Judge evaluation

```bash
python main.py judge \
  --output output/answer.json \
  --ground-truth "expected reference text"
```

### Global options

| Option | Description |
|---|---|
| `--input-dir PATH` | Override the default `input/` directory |
| `--verbose` / `-v` | Enable DEBUG-level logging |

---

## Example output

```
Query: "Which audio files discuss Italian politics?"

Answer:
The analysed audio files discuss Italian politics mainly in two files:
- intervista_mario.mp3 (00:02:15 – 00:05:40): the budget law and the positions
  of the main parties are analysed...
- podcast_news.wav (00:10:00 – 00:13:22): the composition of the current
  government and the latest ministerial statements are discussed...

Reference:
[1] intervista_mario.mp3 — chunk 3, 5 (02:15–05:40)
[2] podcast_news.wav — chunk 12 (10:00–13:22)
```

---

## Configuration

All parameters live in [src/audio_rag/config.py](src/audio_rag/config.py):

| Parameter | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model variant |
| `WHISPER_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `OLLAMA_MODEL` | `gemma4:e2b` | Ollama model for generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `TOP_K` | `5` | Retrieved chunks per query |

---

## Project structure

```
audio-rag/
├── input/                        # Audio files (gitignored)
├── output/                       # Saved query results (gitignored)
├── data/
│   ├── transcriptions/           # Cached JSON transcriptions (gitignored)
│   ├── chroma_db/                # Persistent vector store (gitignored)
│   └── synthetic_dataset/        # Generated test dataset
│       ├── audio/                #   25 WAV files (gitignored)
│       └── manifest.json         #   Dataset metadata
├── src/
│   └── audio_rag/
│       ├── config.py             # Centralised configuration
│       ├── transcriber.py        # Whisper transcription + cache
│       ├── chunker.py            # Text chunking with timestamp mapping
│       ├── embedder.py           # ChromaDB management
│       ├── retriever.py          # Semantic search
│       ├── generator.py          # Ollama answer generation
│       ├── pipeline.py           # End-to-end orchestration
│       ├── judge.py              # LLM-as-a-Judge evaluation
│       └── dataset_generator/    # Synthetic dataset module
│           ├── tts_engine.py     #   Kokoro TTS wrapper
│           ├── script_builder.py #   25 hardcoded scripts
│           ├── dataset_builder.py#   Orchestration + manifest
│           └── manifest.py       #   Serialization + ground-truth queries
├── scripts/
│   └── generate_dataset.py       # Standalone CLI for dataset generation
├── tests/                        # pytest test suite
├── main.py                       # CLI entry point (typer)
├── requirements.txt
└── pyproject.toml
```

---

## Synthetic test dataset

A dedicated module generates 25 synthetic audio files (Italian + English, male and female voices)
using [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) locally, with no external APIs.
The audio files cover varied topics with known ground truth, ideal for evaluating
RAG retrieval quality in a reproducible way.

### Prerequisites

```bash
pip install kokoro-onnx soundfile
# or, using the optional dependency group in pyproject.toml:
pip install -e ".[dataset]"
```

### Dataset generation

```bash
# Full dataset (25 files)
python scripts/generate_dataset.py

# Or via the main CLI
python main.py generate-dataset

# Regenerate and overwrite existing files
python main.py generate-dataset --overwrite

# Italian only
python main.py generate-dataset --lang it

# A specific subset (via standalone script)
python scripts/generate_dataset.py --ids it_f_001_politica it_m_002_tecnologia

# Show the existing manifest
python scripts/generate_dataset.py --show-manifest

# Export ground-truth queries to JSON
python scripts/generate_dataset.py --export-queries data/ground_truth_queries.json
```

### Generated structure

```
data/synthetic_dataset/
├── audio/                  # 25 .wav files (24 kHz, mono, float32)
└── manifest.json           # Metadata: language, gender, voice, duration, keywords
```

### Dataset distribution

| Language | Gender | N. | Kokoro voice | Topics |
|---|---|---|---|---|
| Italian | Female | 6 | `if_sara` | politics, cooking, Renaissance art, tourism, economics, sports |
| Italian | Male | 7 | `im_nicola` | history, technology, environment, football, classical music, cinema, science |
| English | Female | 6 | `af_heart` / `af_bella` | US politics, travel, health, climate change, literature, education |
| English | Male | 6 | `am_adam` / `am_michael` | technology, sports, history, economics, space, philosophy |

Three shared topic pairs (same `topic` across different audio files) enable multi-document
retrieval testing:

| Topic | File audio |
|---|---|
| `technology` | `it_male_002_tecnologia.wav`, `en_male_001_technology.wav` |
| `economics` | `it_female_005_economia.wav`, `en_male_004_economics.wav` |
| `sport` | `it_female_006_sport.wav`, `it_male_004_calcio.wav` |

### Using as input for the RAG pipeline

```bash
# Copy synthetic audio files into the input folder
cp data/synthetic_dataset/audio/*.wav input/

# Index
python main.py ingest

# Test queries (with known ground truth)
python main.py query "Which audio files discuss Italian politics?"
python main.py query "Which audio files discuss technology?"
```

### Evaluation with LLM-as-a-Judge

```bash
# 1. Run a query and save the output
python main.py query "Which audio files discuss Italian politics?" \
  --output output/politics_result.json

# 2. Evaluate against the ground truth from the manifest
python main.py judge \
  --output output/politics_result.json \
  --ground-truth "it_female_001_politica.wav, it_male_004_calcio.wav"
```

---

## Evaluation

### LLM-as-a-Judge (`--eval`)

Add `--eval` to any `query` call to automatically score the answer immediately after it is generated — no saved output file or reference answer required.
The judge uses the same Ollama model to assess whether the answer is grounded in the retrieved context and whether it addresses the question.

```bash
python main.py query "Which audio files discuss Italian politics?" --eval
```

**Example output**

```
Answer:
  The analysed audio files discuss Italian politics mainly in two files...

Evaluation:
  Score:        0.87
  Faithfulness: True
  Relevance:    True
  Explanation:  The answer is directly supported by the retrieved context and
                accurately addresses the question without introducing unsupported claims.

Result saved to: eval_results/eval_20240615_143022.json
```

**Metrics**

| Metric | Type | Description |
|---|---|---|
| `score` | float 0–1 | Overall quality (0 = wrong/hallucinated, 1 = accurate and complete) |
| `faithfulness` | bool | `True` if every claim is grounded in the retrieved context |
| `relevance` | bool | `True` if the answer directly addresses the question |
| `explanation` | str | One or two sentences justifying the score |

Results are saved automatically to `eval_results/eval_<timestamp>.json`.

---

### Retrieval benchmark (`evaluate_retrieval.py`)

Measures retrieval accuracy against a labeled CSV dataset, without invoking the LLM generator.
Useful for tuning `TOP_K`, the embedding model, or chunk size independently of answer quality.

#### Input CSV format

The CSV must have exactly these two columns:

| `question` | `expected_retrieval` |
|---|---|
| Which audio files discuss Italian politics? | `it_female_001_politica.wav::chunk::3` |
| Who talks about technology? | `it_male_002_tecnologia.wav::chunk::1` |

The chunk ID format is `<filename>::chunk::<index>` (e.g. `it_female_001_politica.wav::chunk::3`).

#### Usage

```bash
# Basic run (top-5, saves to eval_results/)
python evaluate_retrieval.py --csv data/eval_dataset.csv

# Custom top-k and output directory
python evaluate_retrieval.py --csv data/eval_dataset.csv --top-k 10 --output-dir reports/
```

**Arguments**

| Argument | Description | Default |
|---|---|---|
| `--csv PATH` | Path to the input CSV file *(required)* | — |
| `--top-k N` | Number of top results to consider for Hit Rate | `5` |
| `--output-dir DIR` | Directory for saved reports | `eval_results/` |

#### Metrics

| Metric | Description |
|---|---|
| **Hit Rate @k** | Fraction of queries where the expected chunk appeared in the top-k results (Recall@k) |
| **MRR** | Mean Reciprocal Rank — average of 1/rank across all queries |
| **Exact Match @1** | Fraction of queries where the top-1 result exactly matched the expected chunk |

**Example output**

```
Retrieval Benchmark Results
============================================================
Total questions : 10
Hit Rate @5     : 0.80  (8/10)
MRR             : 0.72
Exact Match @1  : 0.60  (6/10)

Per-question breakdown:
  Q1   | expected: it_female_001_politica.wav::chunk::3  | retrieved: it_female_001_politica.wav::chunk::3  | hit: True  | rank: 1
  Q2   | expected: it_male_002_tecnologia.wav::chunk::1  | retrieved: en_male_001_technology.wav::chunk::2  | hit: False | rank: -
  ...

Report saved to: eval_results/retrieval_benchmark_20240615_143500.csv
Summary saved to: eval_results/retrieval_benchmark_20240615_143500_summary.json
```

Two files are written per run:
- **`retrieval_benchmark_<timestamp>.csv`** — per-question detail (question, expected, retrieved_top1, hit, rank)
- **`retrieval_benchmark_<timestamp>_summary.json`** — aggregate metrics (hit_rate, mrr, exact_match)

---

## Running tests

```bash
pytest                        # run all tests
pytest --cov=src/audio_rag    # with coverage report
pytest tests/test_chunker.py  # single module
```

---

## Notes

- **No external API keys** — everything runs locally via Ollama and faster-whisper.
- **Transcription cache** — files already transcribed are not reprocessed on subsequent `ingest` runs.
- **Multilingual** — the embedding model (`paraphrase-multilingual-mpnet-base-v2`) handles Italian and other languages natively.
- **GPU acceleration** — if CUDA is available, Whisper automatically uses it (`float16`); otherwise falls back to CPU (`int8`).
