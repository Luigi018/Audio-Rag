"""Integration tests for pipeline.py — all sub-modules mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.audio_rag.chunker import Chunk
from src.audio_rag.config import Config
from src.audio_rag.generator import GeneratedAnswer, Reference
from src.audio_rag.pipeline import AudioRAGPipeline
from src.audio_rag.retriever import RetrievedChunk
from src.audio_rag.transcriber import TranscriptionResult


@pytest.fixture()
def mock_pipeline(test_config: Config, tmp_path: Path) -> AudioRAGPipeline:
    """Pipeline with all sub-components mocked."""
    pipeline = AudioRAGPipeline(test_config)

    # Mock transcriber
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()
    mock_transcription = TranscriptionResult(
        file_path=audio_file, full_text="testo di test", segments=[], language="it", duration=5.0
    )
    pipeline._transcriber = MagicMock()
    pipeline._transcriber.transcribe_all.return_value = [mock_transcription]

    # Mock chunker
    mock_chunk = Chunk(text="chunk testo", source_file=audio_file, chunk_index=0, start_time=0.0, end_time=5.0)
    pipeline._chunker = MagicMock()
    pipeline._chunker.chunk.return_value = [mock_chunk]

    # Mock embedder
    pipeline._embedder = MagicMock()

    # Mock retriever
    mock_retrieved = RetrievedChunk(
        text="chunk testo", source_file=audio_file, chunk_index=0, start_time=0.0, end_time=5.0, similarity_score=0.9
    )
    pipeline._retriever = MagicMock()
    pipeline._retriever.search.return_value = [mock_retrieved]
    pipeline._retriever.group_by_source.return_value = {"test.mp3": [mock_retrieved]}

    # Mock generator
    mock_answer = GeneratedAnswer(
        summary="Risposta test",
        references=[Reference("test.mp3", [0], 0.0, 5.0)],
        raw_context="raw",
    )
    pipeline._generator = MagicMock()
    pipeline._generator.generate_answer.return_value = mock_answer

    return pipeline


class TestPipelineIngest:
    def test_ingest_calls_transcribe_and_chunk_and_embed(
        self, mock_pipeline: AudioRAGPipeline, test_config: Config
    ) -> None:
        n = mock_pipeline.ingest()
        mock_pipeline._transcriber.transcribe_all.assert_called_once()
        mock_pipeline._chunker.chunk.assert_called_once()
        mock_pipeline._embedder.index_chunks.assert_called_once()
        assert n == 1

    def test_ingest_with_custom_input_dir(
        self, mock_pipeline: AudioRAGPipeline, tmp_path: Path
    ) -> None:
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        mock_pipeline.ingest(custom_dir)
        mock_pipeline._transcriber.transcribe_all.assert_called_once_with(custom_dir)

    def test_ingest_no_files_returns_zero(
        self, mock_pipeline: AudioRAGPipeline
    ) -> None:
        mock_pipeline._transcriber.transcribe_all.return_value = []
        n = mock_pipeline.ingest()
        assert n == 0
        mock_pipeline._embedder.index_chunks.assert_not_called()

    def test_ingest_multiple_transcriptions(
        self, mock_pipeline: AudioRAGPipeline, tmp_path: Path
    ) -> None:
        a = tmp_path / "a.mp3"
        b = tmp_path / "b.mp3"
        a.touch()
        b.touch()
        tr_a = TranscriptionResult(file_path=a, full_text="text a", segments=[], language="it", duration=1.0)
        tr_b = TranscriptionResult(file_path=b, full_text="text b", segments=[], language="it", duration=1.0)
        mock_pipeline._transcriber.transcribe_all.return_value = [tr_a, tr_b]
        chunk_a = Chunk(text="a", source_file=a, chunk_index=0)
        chunk_b = Chunk(text="b", source_file=b, chunk_index=0)
        mock_pipeline._chunker.chunk.side_effect = [[chunk_a], [chunk_b]]
        n = mock_pipeline.ingest()
        assert n == 2


class TestPipelineQuery:
    def test_query_returns_generated_answer(
        self, mock_pipeline: AudioRAGPipeline
    ) -> None:
        answer = mock_pipeline.query("domanda di test")
        assert answer.summary == "Risposta test"
        assert len(answer.references) == 1

    def test_query_calls_retriever_and_generator(
        self, mock_pipeline: AudioRAGPipeline
    ) -> None:
        mock_pipeline.query("chi parla di politica?")
        mock_pipeline._retriever.search.assert_called_once()
        mock_pipeline._generator.generate_answer.assert_called_once()

    def test_query_no_chunks_returns_empty_answer(
        self, mock_pipeline: AudioRAGPipeline
    ) -> None:
        mock_pipeline._retriever.search.return_value = []
        answer = mock_pipeline.query("domanda senza risultati")
        assert "Non ho trovato" in answer.summary
        mock_pipeline._generator.generate_answer.assert_not_called()


class TestPipelineIngestAndQuery:
    def test_ingest_and_query_end_to_end(
        self, mock_pipeline: AudioRAGPipeline, tmp_path: Path
    ) -> None:
        answer = mock_pipeline.ingest_and_query(tmp_path, "domanda")
        mock_pipeline._transcriber.transcribe_all.assert_called_once()
        mock_pipeline._generator.generate_answer.assert_called_once()
        assert answer.summary == "Risposta test"
