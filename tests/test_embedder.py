"""Tests for embedder.py — uses ChromaDB in-memory via monkeypatching."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from src.audio_rag.chunker import Chunk
from src.audio_rag.config import Config
from src.audio_rag.embedder import Embedder, _make_doc_id


def _make_fake_embed_fn():
    """Build a ChromaDB-compatible embedding function stub at runtime.

    We subclass chromadb.EmbeddingFunction so that all interface checks
    (signature, name(), is_legacy, etc.) are satisfied regardless of the
    installed ChromaDB version.
    """
    chromadb = pytest.importorskip("chromadb")

    class _FakeEmbedFn(chromadb.EmbeddingFunction):
        def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:  # noqa: A002
            return [[0.1] * 10 for _ in input]

    return _FakeEmbedFn()


@pytest.fixture()
def chroma_client():
    """Return a truly ephemeral (non-persistent) in-memory ChromaDB client."""
    chromadb = pytest.importorskip("chromadb")
    # EphemeralClient() is the fully in-memory client in chromadb >=0.4.22;
    # Client() became an alias for PersistentClient in newer releases.
    try:
        return chromadb.EphemeralClient()
    except AttributeError:
        return chromadb.Client()


@pytest.fixture()
def embedder_with_mock(test_config: Config, chroma_client) -> Embedder:
    """Embedder wired to an in-memory ChromaDB client with a stub embed fn.

    A UUID collection name is used so every test starts with an empty collection
    even when the underlying ChromaDB client caches state across fixture calls.
    """
    import uuid
    embed_fn = _make_fake_embed_fn()
    # Unique name per test avoids cross-test contamination in shared clients.
    test_config.CHROMA_COLLECTION_NAME = f"test_{uuid.uuid4().hex}"
    e = Embedder(test_config)
    e._client = chroma_client
    e._embed_fn = embed_fn
    e._collection = chroma_client.get_or_create_collection(
        name=test_config.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    return e


@pytest.fixture()
def sample_chunks(tmp_path: Path) -> list[Chunk]:
    audio = tmp_path / "audio.mp3"
    audio.touch()
    return [
        Chunk(text="chunk zero text", source_file=audio, chunk_index=0, start_time=0.0, end_time=2.0),
        Chunk(text="chunk one text", source_file=audio, chunk_index=1, start_time=2.0, end_time=4.0),
    ]


class TestDocId:
    def test_make_doc_id_format(self, tmp_path: Path) -> None:
        path = tmp_path / "file.mp3"
        doc_id = _make_doc_id(path, 3)
        assert doc_id == "file.mp3::chunk::3"


class TestEmbedder:
    def test_collection_starts_empty(self, embedder_with_mock: Embedder) -> None:
        assert not embedder_with_mock.collection_exists()

    def test_index_chunks_adds_documents(
        self, embedder_with_mock: Embedder, sample_chunks: list[Chunk]
    ) -> None:
        embedder_with_mock.index_chunks(sample_chunks)
        assert embedder_with_mock.collection_exists()
        assert embedder_with_mock._collection.count() == 2

    def test_index_chunks_upsert_no_duplicates(
        self, embedder_with_mock: Embedder, sample_chunks: list[Chunk]
    ) -> None:
        embedder_with_mock.index_chunks(sample_chunks)
        embedder_with_mock.index_chunks(sample_chunks)  # second upsert
        assert embedder_with_mock._collection.count() == 2

    def test_index_empty_list_is_noop(self, embedder_with_mock: Embedder) -> None:
        embedder_with_mock.index_chunks([])
        assert embedder_with_mock._collection.count() == 0

    def test_clear_collection_resets_to_empty(
        self, embedder_with_mock: Embedder, sample_chunks: list[Chunk]
    ) -> None:
        embedder_with_mock.index_chunks(sample_chunks)
        assert embedder_with_mock._collection.count() == 2
        embedder_with_mock.clear_collection()
        assert embedder_with_mock._collection.count() == 0

    def test_metadata_stored_correctly(
        self, embedder_with_mock: Embedder, sample_chunks: list[Chunk]
    ) -> None:
        embedder_with_mock.index_chunks(sample_chunks)
        results = embedder_with_mock._collection.get(include=["metadatas"])
        metas = {m["chunk_index"]: m for m in results["metadatas"]}
        assert metas[0]["start_time"] == 0.0
        assert metas[1]["end_time"] == 4.0

    def test_null_timestamps_stored_as_minus_one(
        self, embedder_with_mock: Embedder, tmp_path: Path
    ) -> None:
        audio = tmp_path / "no_ts.mp3"
        audio.touch()
        chunk = Chunk(text="no timestamp", source_file=audio, chunk_index=0)
        embedder_with_mock.index_chunks([chunk])
        doc_id = _make_doc_id(audio, 0)
        results = embedder_with_mock._collection.get(ids=[doc_id], include=["metadatas"])
        assert results["metadatas"][0]["start_time"] == -1.0
        assert results["metadatas"][0]["end_time"] == -1.0
