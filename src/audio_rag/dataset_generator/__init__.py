"""Synthetic audio dataset generation for audio-rag testing."""
from __future__ import annotations

from src.audio_rag.dataset_generator.dataset_builder import DatasetBuilder
from src.audio_rag.dataset_generator.manifest import (
    DatasetManifest,
    GroundTruthQuery,
    ManifestEntry,
    ManifestManager,
)
from src.audio_rag.dataset_generator.script_builder import AudioScript, ScriptBuilder
from src.audio_rag.dataset_generator.tts_engine import AudioMeta, TTSEngine

__all__ = [
    "AudioMeta",
    "AudioScript",
    "DatasetBuilder",
    "DatasetManifest",
    "GroundTruthQuery",
    "ManifestEntry",
    "ManifestManager",
    "ScriptBuilder",
    "TTSEngine",
]
