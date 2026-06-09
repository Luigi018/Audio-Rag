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
python main.py query "Dove si parla della politica italiana?"
```

Save the answer to a JSON file:

```bash
python main.py query "Chi è il ministro dell'economia?" --output output/answer.json
```

### 3. Ingest + Query in one step

```bash
python main.py query "tema principale" --input-dir /path/to/audio/
```

### 4. LLM-as-a-Judge evaluation

```bash
python main.py judge \
  --output output/answer.json \
  --ground-truth "testo di riferimento atteso"
```

### Global options

| Option | Description |
|---|---|
| `--input-dir PATH` | Override the default `input/` directory |
| `--verbose` / `-v` | Enable DEBUG-level logging |

---

## Example output

```
Query: "Dove si parla della politica italiana?"

Risposta:
Negli audio analizzati si discute di politica italiana principalmente in due file:
- intervista_mario.mp3 (00:02:15 – 00:05:40): viene analizzata la legge di bilancio
  e le posizioni dei principali partiti...
- podcast_news.wav (00:10:00 – 00:13:22): si commenta la composizione del governo
  attuale e le ultime dichiarazioni ministeriali...

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
│   └── chroma_db/                # Persistent vector store (gitignored)
├── src/
│   └── audio_rag/
│       ├── config.py             # Centralised configuration
│       ├── transcriber.py        # Whisper transcription + cache
│       ├── chunker.py            # Text chunking with timestamp mapping
│       ├── embedder.py           # ChromaDB management
│       ├── retriever.py          # Semantic search
│       ├── generator.py          # Ollama answer generation
│       ├── pipeline.py           # End-to-end orchestration
│       └── judge.py              # LLM-as-a-Judge evaluation
├── tests/                        # pytest test suite
├── main.py                       # CLI entry point (typer)
├── requirements.txt
└── pyproject.toml
```

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
