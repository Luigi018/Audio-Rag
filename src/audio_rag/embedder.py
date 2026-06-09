"""Embedding and ChromaDB management."""
from __future__ import annotations

import logging
from pathlib import Path

from .chunker import Chunk
from .config import Config

logger = logging.getLogger(__name__)


def _make_doc_id(source_file: Path, chunk_index: int) -> str:
    return f"{source_file.name}::chunk::{chunk_index}"


class Embedder:
    """Manages sentence-transformer embeddings and a persistent ChromaDB collection."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        self._client = None
        self._collection = None
        self._embed_fn = None

    def _init(self) -> None:
        if self._collection is not None:
            return
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            ) from exc

        try:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            ) from exc

        self._config.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._config.CHROMA_DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=self._config.EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=self._config.CHROMA_COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d docs)",
            self._config.CHROMA_COLLECTION_NAME,
            self._collection.count(),
        )

    def collection_exists(self) -> bool:
        """Return True if the ChromaDB collection contains at least one document."""
        self._init()
        return self._collection.count() > 0  # type: ignore[union-attr]

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert chunks into ChromaDB (idempotent — no duplicates).

        Args:
            chunks: List of Chunk objects to embed and store.
        """
        self._init()
        if not chunks:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            doc_id = _make_doc_id(chunk.source_file, chunk.chunk_index)
            ids.append(doc_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source_file": str(chunk.source_file),
                    "chunk_index": chunk.chunk_index,
                    "start_time": chunk.start_time if chunk.start_time is not None else -1.0,
                    "end_time": chunk.end_time if chunk.end_time is not None else -1.0,
                }
            )

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)  # type: ignore[union-attr]
        logger.info("Indexed %d chunk(s) into ChromaDB", len(chunks))

    def clear_collection(self) -> None:
        """Delete and recreate the ChromaDB collection."""
        self._init()
        self._client.delete_collection(self._config.CHROMA_COLLECTION_NAME)  # type: ignore[union-attr]
        self._collection = self._client.get_or_create_collection(  # type: ignore[union-attr]
            name=self._config.CHROMA_COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' cleared.", self._config.CHROMA_COLLECTION_NAME)

    def get_raw_collection(self):  # type: ignore[return]
        """Expose the underlying ChromaDB collection (for Retriever)."""
        self._init()
        return self._collection

    def get_embed_fn(self):
        """Expose the embedding function (for Retriever)."""
        self._init()
        return self._embed_fn
