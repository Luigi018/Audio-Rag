"""Answer generation via Ollama (gemma4:e2b)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore[assignment]

from .config import Config
from .retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass
class Reference:
    file_name: str
    chunk_indices: list[int] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None

    def format(self) -> str:
        chunks_str = ", ".join(str(i) for i in self.chunk_indices)
        time_str = f"{_fmt_time(self.start_time)}–{_fmt_time(self.end_time)}"
        return f"{self.file_name} — chunk {chunks_str} ({time_str})"


@dataclass
class GeneratedAnswer:
    summary: str
    references: list[Reference] = field(default_factory=list)
    raw_context: str = ""

    def format_references(self) -> str:
        lines = []
        for i, ref in enumerate(self.references, 1):
            lines.append(f"[{i}] {ref.format()}")
        return "\n".join(lines)


_PROMPT_TEMPLATE = """\
Sei un assistente che risponde a domande basandosi su trascrizioni di file audio.

Di seguito sono riportati i frammenti audio più rilevanti, raggruppati per file sorgente:

{context}

---

Domanda dell'utente: {query}

Istruzioni:
- Rispondi in italiano in modo chiaro e strutturato.
- Cita esplicitamente i file audio e i momenti temporali rilevanti.
- Non inventare informazioni non presenti nei frammenti forniti.
- Se i frammenti non contengono informazioni sufficienti, dillo esplicitamente.

Risposta:
"""


class Generator:
    """Generates a narrative answer from retrieved chunks using Ollama."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()

    def _build_context(
        self, grouped: dict[str, list[RetrievedChunk]]
    ) -> tuple[str, list[Reference]]:
        """Build prompt context string and Reference list from grouped chunks."""
        context_parts: list[str] = []
        references: list[Reference] = []

        for file_name, chunks in grouped.items():
            chunks_sorted = sorted(chunks, key=lambda c: c.chunk_index)
            file_section = [f"### {file_name}"]
            chunk_indices: list[int] = []
            min_start: float | None = None
            max_end: float | None = None

            for chunk in chunks_sorted:
                ts = f"[{_fmt_time(chunk.start_time)} – {_fmt_time(chunk.end_time)}]"
                file_section.append(f"{ts} {chunk.text}")
                chunk_indices.append(chunk.chunk_index)
                if chunk.start_time is not None:
                    min_start = (
                        chunk.start_time
                        if min_start is None
                        else min(min_start, chunk.start_time)
                    )
                if chunk.end_time is not None:
                    max_end = (
                        chunk.end_time
                        if max_end is None
                        else max(max_end, chunk.end_time)
                    )

            context_parts.append("\n".join(file_section))
            references.append(
                Reference(
                    file_name=file_name,
                    chunk_indices=chunk_indices,
                    start_time=min_start,
                    end_time=max_end,
                )
            )

        return "\n\n".join(context_parts), references

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        grouped: dict[str, list[RetrievedChunk]] | None = None,
    ) -> GeneratedAnswer:
        """Generate a narrative answer with source references.

        Args:
            query: User question in natural language.
            retrieved_chunks: Chunks from retrieval step.
            grouped: Pre-grouped chunks (optional; computed if not provided).

        Returns:
            GeneratedAnswer with summary, references and raw context.
        """
        if not grouped:
            from .retriever import Retriever
            r = Retriever(config=self._config)
            grouped = r.group_by_source(retrieved_chunks)

        context, references = self._build_context(grouped)
        prompt = _PROMPT_TEMPLATE.format(context=context, query=query)

        if ollama is None:
            raise ImportError(
                "ollama Python SDK is required. Install with: pip install ollama"
            )

        logger.info(
            "Calling Ollama model '%s' for query: %s", self._config.OLLAMA_MODEL, query[:80]
        )
        client = ollama.Client(host=self._config.OLLAMA_BASE_URL)
        response = client.chat(
            model=self._config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response["message"]["content"].strip()
        logger.debug("Ollama response: %s", summary[:200])

        return GeneratedAnswer(
            summary=summary,
            references=references,
            raw_context=context,
        )
