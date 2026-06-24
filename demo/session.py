"""Centralized st.session_state management for the audio-rag demo.

All session keys are defined and initialized here; app.py and ui_components.py
must never create new session keys directly.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from src.audio_rag.pipeline import AudioRAGPipeline

UPLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "audio_rag_uploads"


# ── Config ────────────────────────────────────────────────────────────────────

def _build_config():
    """Return a Config instance with env-var overrides applied.

    The demo uses its own ChromaDB directory so it never reads from
    (or pollutes) the main pipeline database.
    """
    from src.audio_rag.config import Config
    cfg = Config()
    cfg.OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", cfg.OLLAMA_BASE_URL)
    cfg.OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", cfg.OLLAMA_MODEL)
    cfg.WHISPER_MODEL = os.environ.get("WHISPER_MODEL", cfg.WHISPER_MODEL)
    # Isolated vector store — prevents the demo from seeing audio indexed by the CLI
    cfg.CHROMA_DB_PATH = cfg.BASE_DIR / "data" / "demo_chroma_db"
    return cfg


# ── Initialization ────────────────────────────────────────────────────────────

def init() -> None:
    """Initialize all session_state keys exactly once per browser session."""
    defaults = {
        "pipeline": None,
        "indexed_files": [],   # list[dict] — files indexed in this session
        "chat_history": [],    # list[dict] — {role, content, sources}
        "kb_ready": False,     # True when ChromaDB has at least one chunk
        "_config": None,       # Config instance (built lazily)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Pipeline access ───────────────────────────────────────────────────────────

def get_config():
    if st.session_state._config is None:
        st.session_state._config = _build_config()
    return st.session_state._config


def get_pipeline() -> "AudioRAGPipeline":
    """Return the pipeline singleton, creating it lazily on first call."""
    from src.audio_rag.pipeline import AudioRAGPipeline

    if st.session_state.pipeline is None:
        st.session_state.pipeline = AudioRAGPipeline(get_config())
        refresh_kb_status()
    return st.session_state.pipeline


def refresh_kb_status() -> None:
    """Update kb_ready by querying the ChromaDB collection count."""
    try:
        pipeline = st.session_state.pipeline
        if pipeline is None:
            return
        count = pipeline._embedder.get_raw_collection().count()
        st.session_state.kb_ready = count > 0
    except Exception:
        st.session_state.kb_ready = False


def get_kb_chunk_count() -> int:
    """Return the number of chunks currently in ChromaDB (0 if unavailable)."""
    try:
        pipeline = st.session_state.pipeline
        if pipeline is None:
            return 0
        return pipeline._embedder.get_raw_collection().count()
    except Exception:
        return 0


# ── File ingestion ────────────────────────────────────────────────────────────

def get_upload_dir() -> Path:
    UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_TMP_DIR


def ingest_uploaded_file(uploaded_file) -> dict:
    """Save, transcribe, chunk and index one Streamlit UploadedFile.

    Returns a metadata dict suitable for appending to indexed_files.
    Raises on transcription / indexing failure.
    """
    pipeline = get_pipeline()
    upload_dir = get_upload_dir()

    tmp_path = upload_dir / uploaded_file.name
    tmp_path.write_bytes(uploaded_file.getbuffer())

    transcription = pipeline._transcriber.transcribe_file(tmp_path)
    chunks = pipeline._chunker.chunk(transcription)
    pipeline._embedder.index_chunks(chunks)

    st.session_state.kb_ready = True

    return {
        "name": uploaded_file.name,
        "indexed_at": datetime.now().strftime("%H:%M:%S"),
        "text": transcription.full_text,
        "language": transcription.language,
        "chunks": len(chunks),
    }


def already_indexed(filename: str) -> bool:
    return any(f["name"] == filename for f in st.session_state.indexed_files)
