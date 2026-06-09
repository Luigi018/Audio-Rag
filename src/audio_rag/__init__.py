"""audio-rag: RAG pipeline over local audio files."""
from .config import Config
from .pipeline import AudioRAGPipeline

__all__ = ["AudioRAGPipeline", "Config"]
