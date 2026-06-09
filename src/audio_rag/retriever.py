"""Semantic retrieval over the ChromaDB vector store."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .chunker import Chunk
from .config import Config
from .embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk(Chunk):
    similarity_score: float = 0.0


class Retriever:
    """Searches the ChromaDB collection and groups results by source file."""

    def __init__(self, embedder: Embedder | None = None, config: Config | None = None) -> None:
        self._config = config or Config()
        self._embedder = embedder or Embedder(self._config)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed query and retrieve the most similar chunks.

        Args:
            query: Natural-language question.
            top_k: Number of results to return. Defaults to Config.TOP_K.

        Returns:
            List of RetrievedChunk sorted by descending similarity score.
        """
        k = top_k or self._config.TOP_K
        collection = self._embedder.get_raw_collection()
        embed_fn = self._embedder.get_embed_fn()

        query_embedding = embed_fn([query])
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB cosine distance → similarity: 1 - distance
            score = max(0.0, 1.0 - float(dist))
            start = meta.get("start_time", -1.0)
            end = meta.get("end_time", -1.0)
            retrieved.append(
                RetrievedChunk(
                    text=doc,
                    source_file=Path(meta["source_file"]),
                    chunk_index=int(meta["chunk_index"]),
                    start_time=start if start >= 0 else None,
                    end_time=end if end >= 0 else None,
                    similarity_score=score,
                )
            )

        retrieved.sort(key=lambda c: c.similarity_score, reverse=True)
        logger.info(
            "Query returned %d chunk(s) (top score: %.3f)",
            len(retrieved),
            retrieved[0].similarity_score if retrieved else 0.0,
        )
        return retrieved

    def group_by_source(
        self, chunks: list[RetrievedChunk]
    ) -> dict[str, list[RetrievedChunk]]:
        """Group retrieved chunks by their source file name.

        Args:
            chunks: List of RetrievedChunk objects.

        Returns:
            Dict mapping file name to list of RetrievedChunk.
        """
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            key = chunk.source_file.name
            grouped.setdefault(key, []).append(chunk)
        return grouped
