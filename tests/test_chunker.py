"""Tests for chunker.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.audio_rag.chunker import Chunk, Chunker
from src.audio_rag.config import Config
from src.audio_rag.transcriber import Segment, TranscriptionResult


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    c = Config()
    c.CHUNK_SIZE = 50
    c.CHUNK_OVERLAP = 10
    c.TRANSCRIPTIONS_DIR = tmp_path / "t"
    c.TRANSCRIPTIONS_DIR.mkdir()
    return c


def _make_transcription(tmp_path: Path, text: str, segments: list[Segment] | None = None) -> TranscriptionResult:
    audio = tmp_path / "audio.mp3"
    audio.touch()
    segs = segments or [Segment(0.0, float(len(text)), text)]
    return TranscriptionResult(
        file_path=audio, full_text=text, segments=segs, language="it", duration=float(len(text))
    )


class TestChunkerShortText:
    def test_short_text_single_chunk(self, cfg: Config, tmp_path: Path) -> None:
        t = _make_transcription(tmp_path, "Testo breve.")
        chunker = Chunker(cfg)
        chunks = chunker.chunk(t)
        assert len(chunks) == 1
        assert chunks[0].text == "Testo breve."
        assert chunks[0].chunk_index == 0

    def test_short_text_timestamps_preserved(self, tmp_path: Path, cfg: Config) -> None:
        segs = [Segment(0.0, 1.0, "Breve"), Segment(1.0, 3.0, "testo")]
        t = _make_transcription(tmp_path, "Breve testo", segs)
        chunker = Chunker(cfg)
        chunks = chunker.chunk(t)
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 3.0

    def test_text_exactly_at_size_limit_single_chunk(self, cfg: Config, tmp_path: Path) -> None:
        text = "A" * cfg.CHUNK_SIZE
        t = _make_transcription(tmp_path, text)
        chunks = Chunker(cfg).chunk(t)
        assert len(chunks) == 1


class TestChunkerLongText:
    def test_long_text_multiple_chunks(self, cfg: Config, tmp_path: Path) -> None:
        text = "A" * (cfg.CHUNK_SIZE * 3)
        t = _make_transcription(tmp_path, text)
        chunks = Chunker(cfg).chunk(t)
        assert len(chunks) > 1

    def test_chunk_size_respected(self, cfg: Config, tmp_path: Path) -> None:
        text = "B" * 200
        t = _make_transcription(tmp_path, text)
        chunks = Chunker(cfg).chunk(t)
        for chunk in chunks:
            assert len(chunk.text) <= cfg.CHUNK_SIZE

    def test_overlap_present(self, cfg: Config, tmp_path: Path) -> None:
        text = "X" * 120
        t = _make_transcription(tmp_path, text)
        chunks = Chunker(cfg).chunk(t)
        assert len(chunks) >= 2
        # Last char of chunk[0] should appear at start of chunk[1]
        end_of_first = chunks[0].text[-cfg.CHUNK_OVERLAP:]
        start_of_second = chunks[1].text[: cfg.CHUNK_OVERLAP]
        assert end_of_first == start_of_second

    def test_chunk_indices_sequential(self, cfg: Config, tmp_path: Path) -> None:
        text = "C" * 300
        t = _make_transcription(tmp_path, text)
        chunks = Chunker(cfg).chunk(t)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_source_file_propagated(self, cfg: Config, tmp_path: Path) -> None:
        text = "D" * 200
        t = _make_transcription(tmp_path, text)
        chunks = Chunker(cfg).chunk(t)
        for chunk in chunks:
            assert chunk.source_file == t.file_path

    def test_timestamp_mapping_with_segments(self, cfg: Config, tmp_path: Path) -> None:
        segs = [
            Segment(0.0, 10.0, "A" * 40),
            Segment(10.0, 20.0, "B" * 40),
            Segment(20.0, 30.0, "C" * 40),
        ]
        full_text = " ".join(s.text for s in segs)
        audio = tmp_path / "seg_audio.mp3"
        audio.touch()
        t = TranscriptionResult(
            file_path=audio, full_text=full_text, segments=segs, language="it", duration=30.0
        )
        chunks = Chunker(cfg).chunk(t)
        for chunk in chunks:
            assert chunk.start_time is not None
            assert chunk.end_time is not None


class TestChunkerEdgeCases:
    def test_empty_text_returns_empty_list(self, cfg: Config, tmp_path: Path) -> None:
        t = _make_transcription(tmp_path, "")
        chunks = Chunker(cfg).chunk(t)
        assert chunks == []

    def test_whitespace_only_text(self, cfg: Config, tmp_path: Path) -> None:
        t = _make_transcription(tmp_path, "   ")
        chunks = Chunker(cfg).chunk(t)
        assert chunks == []

    def test_no_segments_still_works(self, cfg: Config, tmp_path: Path) -> None:
        audio = tmp_path / "noseg.mp3"
        audio.touch()
        t = TranscriptionResult(
            file_path=audio, full_text="hello world", segments=[], language="en", duration=5.0
        )
        chunks = Chunker(cfg).chunk(t)
        assert len(chunks) == 1
        assert chunks[0].start_time is None
        assert chunks[0].end_time is None
