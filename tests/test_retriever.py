"""Tests for retriever.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.audio_rag.config import Config
from src.audio_rag.embedder import Embedder
from src.audio_rag.retriever import RetrievedChunk, Retriever


def _make_retriever(tmp_path: Path, query_results: dict) -> Retriever:
    """Build a Retriever with a fully mocked Embedder."""
    cfg = Config()
    mock_embed_fn = MagicMock()
    mock_embed_fn.side_effect = lambda texts: [[0.1] * 10 for _ in texts]

    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.query.return_value = query_results

    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.get_raw_collection.return_value = mock_collection
    mock_embedder.get_embed_fn.return_value = mock_embed_fn

    return Retriever(embedder=mock_embedder, config=cfg)


def _make_query_results(source_files: list[str], texts: list[str], distances: list[float]) -> dict:
    return {
        "documents": [texts],
        "metadatas": [
            [
                {"source_file": sf, "chunk_index": i, "start_time": float(i), "end_time": float(i + 1)}
                for i, sf in enumerate(source_files)
            ]
        ],
        "distances": [distances],
    }


class TestRetrieverSearch:
    def test_search_returns_retrieved_chunks(self, tmp_path: Path) -> None:
        results = _make_query_results(
            [str(tmp_path / "a.mp3"), str(tmp_path / "b.mp3")],
            ["chunk a text", "chunk b text"],
            [0.1, 0.3],
        )
        retriever = _make_retriever(tmp_path, results)
        chunks = retriever.search("query about something")
        assert len(chunks) == 2
        assert all(isinstance(c, RetrievedChunk) for c in chunks)

    def test_search_sorted_by_score_descending(self, tmp_path: Path) -> None:
        results = _make_query_results(
            [str(tmp_path / "a.mp3"), str(tmp_path / "b.mp3")],
            ["low score", "high score"],
            [0.4, 0.05],  # lower distance = higher similarity
        )
        retriever = _make_retriever(tmp_path, results)
        chunks = retriever.search("query")
        assert chunks[0].similarity_score > chunks[1].similarity_score

    def test_similarity_score_computed_from_distance(self, tmp_path: Path) -> None:
        results = _make_query_results(
            [str(tmp_path / "a.mp3")],
            ["text"],
            [0.3],
        )
        retriever = _make_retriever(tmp_path, results)
        chunks = retriever.search("query")
        assert abs(chunks[0].similarity_score - 0.7) < 1e-6

    def test_timestamps_none_when_negative(self, tmp_path: Path) -> None:
        results = {
            "documents": [["text"]],
            "metadatas": [[{"source_file": str(tmp_path / "a.mp3"), "chunk_index": 0, "start_time": -1.0, "end_time": -1.0}]],
            "distances": [[0.2]],
        }
        retriever = _make_retriever(tmp_path, results)
        chunks = retriever.search("q")
        assert chunks[0].start_time is None
        assert chunks[0].end_time is None

    def test_empty_results(self, tmp_path: Path) -> None:
        results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        retriever = _make_retriever(tmp_path, results)
        chunks = retriever.search("query")
        assert chunks == []


class TestGroupBySource:
    def test_group_by_source_single_file(self, sample_retrieved_chunks: list[RetrievedChunk]) -> None:
        retriever = Retriever(config=Config())
        grouped = retriever.group_by_source(sample_retrieved_chunks)
        assert len(grouped) == 1
        assert "test_audio.mp3" in grouped

    def test_group_by_source_multiple_files(self, tmp_path: Path) -> None:
        cfg = Config()
        retriever = Retriever(config=cfg)
        file_a = tmp_path / "a.mp3"
        file_b = tmp_path / "b.mp3"
        file_a.touch()
        file_b.touch()
        chunks = [
            RetrievedChunk(text="a1", source_file=file_a, chunk_index=0, similarity_score=0.9),
            RetrievedChunk(text="b1", source_file=file_b, chunk_index=0, similarity_score=0.8),
            RetrievedChunk(text="a2", source_file=file_a, chunk_index=1, similarity_score=0.7),
        ]
        grouped = retriever.group_by_source(chunks)
        assert len(grouped) == 2
        assert len(grouped["a.mp3"]) == 2
        assert len(grouped["b.mp3"]) == 1

    def test_group_by_source_empty_input(self) -> None:
        retriever = Retriever(config=Config())
        assert retriever.group_by_source([]) == {}
