"""Text chunking with timestamp interpolation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .transcriber import Segment, TranscriptionResult

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    source_file: Path
    chunk_index: int
    start_time: float | None = None
    end_time: float | None = None


class Chunker:
    """Splits transcriptions into overlapping text chunks with timestamp mapping."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()

    def _interpolate_timestamps(
        self, char_start: int, char_end: int, segments: list[Segment], full_text: str
    ) -> tuple[float | None, float | None]:
        """Map character offsets back to audio timestamps via segment boundaries."""
        if not segments:
            return None, None

        # Build cumulative character positions for each segment
        positions: list[tuple[int, int, Segment]] = []
        offset = 0
        for seg in segments:
            seg_len = len(seg.text)
            positions.append((offset, offset + seg_len, seg))
            offset += seg_len + 1  # +1 for the space joining segments

        def _find_time(char_pos: int, use_end: bool) -> float | None:
            for seg_start, seg_end, seg in positions:
                if seg_start <= char_pos <= seg_end:
                    return seg.end if use_end else seg.start
            # Clamp to boundaries
            if char_pos <= positions[0][0]:
                return positions[0][2].start
            return positions[-1][2].end

        return _find_time(char_start, False), _find_time(char_end, True)

    def chunk(self, transcription: TranscriptionResult) -> list[Chunk]:
        """Split a transcription into chunks, preserving overlap and timestamps.

        Args:
            transcription: Source TranscriptionResult to chunk.

        Returns:
            List of Chunk objects. Returns a single chunk if text is short.
        """
        text = transcription.full_text.strip()
        if not text:
            logger.warning("Empty transcription for '%s'", transcription.file_path.name)
            return []

        size = self._config.CHUNK_SIZE
        overlap = self._config.CHUNK_OVERLAP

        if len(text) <= size:
            t_start = transcription.segments[0].start if transcription.segments else None
            t_end = transcription.segments[-1].end if transcription.segments else None
            return [
                Chunk(
                    text=text,
                    source_file=transcription.file_path,
                    chunk_index=0,
                    start_time=t_start,
                    end_time=t_end,
                )
            ]

        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk_text = text[start:end]
            t_start, t_end = self._interpolate_timestamps(
                start, end, transcription.segments, text
            )
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_file=transcription.file_path,
                    chunk_index=idx,
                    start_time=t_start,
                    end_time=t_end,
                )
            )
            idx += 1
            if end == len(text):
                break
            start = end - overlap

        logger.debug(
            "Chunked '%s' into %d chunk(s)", transcription.file_path.name, len(chunks)
        )
        return chunks
