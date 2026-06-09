"""Centralized configuration for the audio-rag pipeline."""
from __future__ import annotations

import logging
from pathlib import Path


class Config:
    """All tuneable parameters for the audio-rag pipeline."""

    # Directories
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    INPUT_DIR: Path = BASE_DIR / "input"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    TRANSCRIPTIONS_DIR: Path = BASE_DIR / "data" / "transcriptions"
    CHROMA_DB_PATH: Path = BASE_DIR / "data" / "chroma_db"

    # Whisper
    WHISPER_MODEL: str = "large-v3-turbo"
    WHISPER_DEVICE: str = "auto"  # "cuda", "cpu", or "auto"

    # Embedding
    EMBEDDING_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"
    CHROMA_COLLECTION_NAME: str = "audio_rag"

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Ollama / LLM
    OLLAMA_MODEL: str = "gemma4:e2b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Retrieval
    TOP_K: int = 5

    # Logging
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Supported audio extensions
    AUDIO_EXTENSIONS: tuple[str, ...] = (
        ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4"
    )
