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

## Dataset sintetico di test

Un modulo dedicato genera 25 file audio sintetici (italiano + inglese, voci maschili e femminili)
usando [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) in locale, senza API esterne.
Gli audio coprono argomenti tematici variegati e hanno ground truth noti, ideali per valutare
la qualità del retrieval RAG in modo riproducibile.

### Prerequisiti

```bash
pip install kokoro-onnx soundfile
# oppure, con il gruppo opzionale di pyproject.toml:
pip install -e ".[dataset]"
```

### Generazione del dataset

```bash
# Dataset completo (25 file)
python scripts/generate_dataset.py

# Oppure via CLI principale
python main.py generate-dataset

# Rigenera sovrascrivendo i file esistenti
python main.py generate-dataset --overwrite

# Solo italiano
python main.py generate-dataset --lang it

# Solo un sottoinsieme (via script standalone)
python scripts/generate_dataset.py --ids it_f_001_politica it_m_002_tecnologia

# Mostra il manifest esistente
python scripts/generate_dataset.py --show-manifest

# Esporta le ground-truth queries in JSON
python scripts/generate_dataset.py --export-queries data/ground_truth_queries.json
```

### Struttura generata

```
data/synthetic_dataset/
├── audio/                  # 25 file .wav (24 kHz, mono, float32)
└── manifest.json           # Metadata: lingua, genere, voce, durata, keyword
```

### Distribuzione del dataset

| Lingua | Genere | N. | Voce Kokoro | Argomenti |
|---|---|---|---|---|
| Italiano | Femminile | 6 | `if_sara` | politica, cucina, arte rinascimentale, turismo, economia, sport |
| Italiano | Maschile | 7 | `im_nicola` | storia, tecnologia, ambiente, calcio, musica classica, cinema, scienza |
| Inglese | Femminile | 6 | `af_heart` / `af_bella` | US politics, travel, health, climate change, literature, education |
| Inglese | Maschile | 6 | `am_adam` / `am_michael` | technology, sports, history, economics, space, philosophy |

Tre coppie di topic condivisi (stesso `topic` su audio diversi) consentono di testare
il retrieval multi-documento:

| Topic | File audio |
|---|---|
| `technology` | `it_male_002_tecnologia.wav`, `en_male_001_technology.wav` |
| `economics` | `it_female_005_economia.wav`, `en_male_004_economics.wav` |
| `sport` | `it_female_006_sport.wav`, `it_male_004_calcio.wav` |

### Utilizzo come input per la pipeline RAG

```bash
# Copia gli audio sintetici nella cartella di input
cp data/synthetic_dataset/audio/*.wav input/

# Indicizza
python main.py ingest

# Query di test (con ground truth noti)
python main.py query "Dove si parla della politica italiana?"
python main.py query "Which audio files discuss technology?"
```

### Valutazione con LLM-as-a-Judge

```bash
# 1. Esegui la query e salva l'output
python main.py query "Dove si parla della politica italiana?" \
  --output output/risultato_politica.json

# 2. Valuta con ground truth dal manifest
python main.py judge \
  --output output/risultato_politica.json \
  --ground-truth "it_female_001_politica.wav, it_male_004_calcio.wav"
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
