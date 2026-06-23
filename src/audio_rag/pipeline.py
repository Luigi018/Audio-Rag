"""End-to-end orchestration of the audio-rag pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

from .chunker import Chunker
from .config import Config
from .embedder import Embedder
from .generator import GeneratedAnswer, Generator
from .retriever import Retriever
from .transcriber import Transcriber

logger = logging.getLogger(__name__)


class AudioRAGPipeline:
    """Orchestrates transcription, indexing, retrieval and generation."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        self._transcriber = Transcriber(self._config)
        self._chunker = Chunker(self._config)
        self._embedder = Embedder(self._config)
        self._retriever = Retriever(self._embedder, self._config)
        self._generator = Generator(self._config)

    def ingest(self, input_dir: Path | None = None) -> int:
        """Transcribe, chunk and index all audio files in input_dir.

        Args:
            input_dir: Directory containing audio files. Defaults to Config.INPUT_DIR.

        Returns:
            Total number of chunks indexed.
        """
        search_dir = input_dir or self._config.INPUT_DIR
        logger.info("Starting ingestion from '%s'", search_dir)

        transcriptions = self._transcriber.transcribe_all(search_dir)
        if not transcriptions:
            logger.warning("No audio files found in '%s'", search_dir)
            return 0

        all_chunks = []
        for transcription in transcriptions:
            chunks = self._chunker.chunk(transcription)
            all_chunks.extend(chunks)

        self._embedder.index_chunks(all_chunks)
        logger.info(
            "Ingestion complete: %d file(s), %d chunk(s) indexed",
            len(transcriptions),
            len(all_chunks),
        )
        return len(all_chunks)

    def query(self, question: str) -> GeneratedAnswer:
        """Retrieve relevant chunks and generate an answer.

        Args:
            question: Natural-language question from the user.

        Returns:
            GeneratedAnswer with narrative summary and references.
        """
        logger.info("Processing query: %s", question[:120])
        chunks = self._retriever.search(question, top_k=self._config.TOP_K)
        if not chunks:
            logger.warning("No chunks retrieved for query.")
            return GeneratedAnswer(
                summary="No relevant information found in the indexed audio files.",
                references=[],
                raw_context="",
            )

        grouped = self._retriever.group_by_source(chunks)
        return self._generator.generate_answer(question, chunks, grouped)

    def ingest_and_query(
        self, input_dir: Path | None = None, question: str = ""
    ) -> GeneratedAnswer:
        """Full pipeline: ingest audio files then answer a question.

        Args:
            input_dir: Directory with audio files.
            question: Natural-language question.

        Returns:
            GeneratedAnswer.
        """
        self.ingest(input_dir)
        return self.query(question)
