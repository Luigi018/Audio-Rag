"""Shared pytest fixtures for audio-rag tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.audio_rag.chunker import Chunk
from src.audio_rag.config import Config
from src.audio_rag.generator import GeneratedAnswer, Reference
from src.audio_rag.retriever import RetrievedChunk
from src.audio_rag.transcriber import Segment, TranscriptionResult


@pytest.fixture()
def test_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.INPUT_DIR = tmp_path / "input"
    cfg.OUTPUT_DIR = tmp_path / "output"
    cfg.TRANSCRIPTIONS_DIR = tmp_path / "transcriptions"
    cfg.CHROMA_DB_PATH = tmp_path / "chroma_db"
    cfg.INPUT_DIR.mkdir()
    cfg.OUTPUT_DIR.mkdir()
    cfg.TRANSCRIPTIONS_DIR.mkdir()
    return cfg


@pytest.fixture()
def sample_segments() -> list[Segment]:
    return [
        Segment(start=0.0, end=2.0, text="Hello this is the first segment"),
        Segment(start=2.0, end=5.0, text="and this is the second segment"),
        Segment(start=5.0, end=8.0, text="finally the third and last segment"),
    ]


@pytest.fixture()
def sample_transcription(tmp_path: Path, sample_segments: list[Segment]) -> TranscriptionResult:
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.touch()
    full_text = " ".join(s.text for s in sample_segments)
    return TranscriptionResult(
        file_path=audio_file,
        full_text=full_text,
        segments=sample_segments,
        language="it",
        duration=8.0,
    )


@pytest.fixture()
def sample_chunks(tmp_path: Path) -> list[Chunk]:
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.touch()
    return [
        Chunk(text="first chunk of text", source_file=audio_file, chunk_index=0, start_time=0.0, end_time=2.0),
        Chunk(text="second chunk of text", source_file=audio_file, chunk_index=1, start_time=2.0, end_time=5.0),
    ]


@pytest.fixture()
def sample_retrieved_chunks(tmp_path: Path) -> list[RetrievedChunk]:
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.touch()
    return [
        RetrievedChunk(
            text="first chunk of text",
            source_file=audio_file,
            chunk_index=0,
            start_time=0.0,
            end_time=2.0,
            similarity_score=0.9,
        ),
        RetrievedChunk(
            text="second chunk of text",
            source_file=audio_file,
            chunk_index=1,
            start_time=2.0,
            end_time=5.0,
            similarity_score=0.7,
        ),
    ]


@pytest.fixture()
def sample_answer() -> GeneratedAnswer:
    return GeneratedAnswer(
        summary="Generated test answer.",
        references=[
            Reference(
                file_name="test_audio.mp3",
                chunk_indices=[0, 1],
                start_time=0.0,
                end_time=5.0,
            )
        ],
        raw_context="raw test context",
    )
