"""Tests for generator.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.audio_rag.config import Config
from src.audio_rag.generator import GeneratedAnswer, Generator, Reference, _fmt_time
from src.audio_rag.retriever import RetrievedChunk


@pytest.fixture()
def generator(test_config: Config) -> Generator:
    return Generator(test_config)


@pytest.fixture()
def two_file_chunks(tmp_path: Path) -> list[RetrievedChunk]:
    a = tmp_path / "interview.mp3"
    b = tmp_path / "podcast.wav"
    a.touch()
    b.touch()
    return [
        RetrievedChunk(text="Italian politics", source_file=a, chunk_index=0, start_time=135.0, end_time=340.0, similarity_score=0.9),
        RetrievedChunk(text="current government", source_file=b, chunk_index=2, start_time=600.0, end_time=802.0, similarity_score=0.8),
    ]


def _mock_ollama_response(text: str) -> dict:
    return {"message": {"content": text}}


class TestFmtTime:
    def test_none_returns_question_mark(self) -> None:
        assert _fmt_time(None) == "?"

    def test_seconds_only(self) -> None:
        assert _fmt_time(45.0) == "00:45"

    def test_minutes_and_seconds(self) -> None:
        assert _fmt_time(125.0) == "02:05"

    def test_hours_minutes_seconds(self) -> None:
        assert _fmt_time(3661.0) == "01:01:01"


class TestReference:
    def test_format_with_timestamps(self) -> None:
        ref = Reference("audio.mp3", [0, 1], 135.0, 340.0)
        formatted = ref.format()
        assert "audio.mp3" in formatted
        assert "02:15" in formatted  # 135s
        assert "05:40" in formatted  # 340s

    def test_format_without_timestamps(self) -> None:
        ref = Reference("audio.mp3", [0])
        formatted = ref.format()
        assert "?" in formatted


class TestGeneratedAnswer:
    def test_format_references_numbered(self, sample_answer: GeneratedAnswer) -> None:
        formatted = sample_answer.format_references()
        assert "[1]" in formatted
        assert "test_audio.mp3" in formatted

    def test_format_references_empty(self) -> None:
        answer = GeneratedAnswer(summary="no refs")
        assert answer.format_references() == ""


class TestGenerator:
    def test_generate_answer_calls_ollama(
        self, generator: Generator, sample_retrieved_chunks: list[RetrievedChunk]
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_ollama_response("Test answer.")

        with patch("src.audio_rag.generator.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            answer = generator.generate_answer("Who is speaking?", sample_retrieved_chunks)

        assert answer.summary == "Test answer."
        mock_client.chat.assert_called_once()

    def test_generate_answer_prompt_contains_query(
        self, generator: Generator, sample_retrieved_chunks: list[RetrievedChunk]
    ) -> None:
        captured_prompt = {}
        mock_client = MagicMock()

        def capture_chat(model, messages):
            captured_prompt["content"] = messages[0]["content"]
            return _mock_ollama_response("ok")

        mock_client.chat.side_effect = capture_chat

        with patch("src.audio_rag.generator.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            generator.generate_answer("Italian politics?", sample_retrieved_chunks)

        assert "Italian politics?" in captured_prompt["content"]

    def test_generate_answer_references_built(
        self, generator: Generator, two_file_chunks: list[RetrievedChunk]
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_ollama_response("Answer.")

        with patch("src.audio_rag.generator.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            answer = generator.generate_answer("question", two_file_chunks)

        assert len(answer.references) == 2
        file_names = {r.file_name for r in answer.references}
        assert "interview.mp3" in file_names
        assert "podcast.wav" in file_names

    def test_generate_answer_missing_ollama_raises(
        self, generator: Generator, sample_retrieved_chunks: list[RetrievedChunk]
    ) -> None:
        with patch("src.audio_rag.generator.ollama", None):
            with pytest.raises(ImportError, match="ollama"):
                generator.generate_answer("query", sample_retrieved_chunks)

    def test_generate_answer_context_in_raw(
        self, generator: Generator, sample_retrieved_chunks: list[RetrievedChunk]
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_ollama_response("ok")

        with patch("src.audio_rag.generator.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            answer = generator.generate_answer("q", sample_retrieved_chunks)

        assert len(answer.raw_context) > 0
