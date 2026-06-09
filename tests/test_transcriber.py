"""Tests for transcriber.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.audio_rag.config import Config
from src.audio_rag.transcriber import Segment, TranscriptionResult, Transcriber


class TestTranscriptionResult:
    def test_to_dict_round_trip(self, tmp_path: Path) -> None:
        audio = tmp_path / "audio.mp3"
        audio.touch()
        original = TranscriptionResult(
            file_path=audio,
            full_text="hello world",
            segments=[Segment(0.0, 1.0, "hello"), Segment(1.0, 2.0, "world")],
            language="en",
            duration=2.0,
        )
        restored = TranscriptionResult.from_dict(original.to_dict())
        assert restored.full_text == original.full_text
        assert restored.language == original.language
        assert len(restored.segments) == 2
        assert restored.segments[0].start == 0.0

    def test_from_dict_missing_optional_fields(self, tmp_path: Path) -> None:
        audio = tmp_path / "audio.mp3"
        audio.touch()
        data = {"file_path": str(audio), "full_text": "text", "segments": []}
        result = TranscriptionResult.from_dict(data)
        assert result.language == ""
        assert result.duration == 0.0


class TestTranscriber:
    def _make_transcriber(self, config: Config) -> Transcriber:
        t = Transcriber(config)
        t._model = MagicMock()
        return t

    def _make_mock_model(self, segments: list[tuple]) -> MagicMock:
        mock_model = MagicMock()
        mock_info = MagicMock()
        mock_info.language = "it"
        mock_info.duration = 5.0
        mock_segments = [
            MagicMock(start=s[0], end=s[1], text=s[2]) for s in segments
        ]
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
        return mock_model

    def test_transcribe_file_not_found(self, test_config: Config) -> None:
        t = Transcriber(test_config)
        with pytest.raises(FileNotFoundError):
            t.transcribe_file(Path("/nonexistent/audio.mp3"))

    def test_transcribe_file_unsupported_extension(self, test_config: Config, tmp_path: Path) -> None:
        bad_file = tmp_path / "audio.xyz"
        bad_file.touch()
        t = Transcriber(test_config)
        with pytest.raises(ValueError, match="Unsupported audio format"):
            t.transcribe_file(bad_file)

    def test_transcribe_file_uses_cache(self, test_config: Config, tmp_path: Path) -> None:
        audio = tmp_path / "cached.mp3"
        audio.touch()
        result = TranscriptionResult(
            file_path=audio, full_text="cached text", segments=[], language="en", duration=1.0
        )
        cache_path = test_config.TRANSCRIPTIONS_DIR / "cached.json"
        cache_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        t = Transcriber(test_config)
        loaded = t.transcribe_file(audio)
        assert loaded.full_text == "cached text"
        assert t._model is None  # model was never loaded

    def test_transcribe_file_happy_path(self, test_config: Config, tmp_path: Path) -> None:
        audio = tmp_path / "fresh.mp3"
        audio.touch()
        t = Transcriber(test_config)
        t._model = self._make_mock_model([(0.0, 2.0, "ciao"), (2.0, 4.0, "mondo")])

        result = t.transcribe_file(audio)
        assert "ciao" in result.full_text
        assert "mondo" in result.full_text
        assert result.language == "it"
        assert result.duration == 5.0
        # Cache should have been written
        cache = test_config.TRANSCRIPTIONS_DIR / "fresh.json"
        assert cache.exists()

    def test_transcribe_file_writes_valid_cache(self, test_config: Config, tmp_path: Path) -> None:
        audio = tmp_path / "cacheme.mp3"
        audio.touch()
        t = Transcriber(test_config)
        t._model = self._make_mock_model([(0.0, 1.0, "testo")])
        t.transcribe_file(audio)
        cache = test_config.TRANSCRIPTIONS_DIR / "cacheme.json"
        data = json.loads(cache.read_text(encoding="utf-8"))
        assert data["full_text"] == "testo"

    def test_transcribe_all_skips_unsupported(self, test_config: Config, tmp_path: Path) -> None:
        (tmp_path / "audio.mp3").touch()
        (tmp_path / "document.pdf").touch()
        test_config.INPUT_DIR = tmp_path
        t = Transcriber(test_config)
        t._model = self._make_mock_model([(0.0, 1.0, "hello")])
        results = t.transcribe_all()
        assert len(results) == 1

    def test_transcribe_all_empty_dir(self, test_config: Config) -> None:
        t = Transcriber(test_config)
        results = t.transcribe_all()
        assert results == []

    def test_transcribe_all_nonexistent_dir(self, test_config: Config) -> None:
        t = Transcriber(test_config)
        results = t.transcribe_all(Path("/no/such/dir"))
        assert results == []

    def test_corrupted_cache_falls_back_to_transcription(
        self, test_config: Config, tmp_path: Path
    ) -> None:
        audio = tmp_path / "corrupt.mp3"
        audio.touch()
        cache = test_config.TRANSCRIPTIONS_DIR / "corrupt.json"
        cache.write_text("INVALID JSON{{{{", encoding="utf-8")
        t = Transcriber(test_config)
        t._model = self._make_mock_model([(0.0, 1.0, "fallback")])
        result = t.transcribe_file(audio)
        assert result.full_text == "fallback"

    def test_all_supported_extensions(self, test_config: Config, tmp_path: Path) -> None:
        for ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4"):
            audio = tmp_path / f"file{ext}"
            audio.touch()
            t = Transcriber(test_config)
            t._model = self._make_mock_model([(0.0, 1.0, "testo")])
            result = t.transcribe_file(audio)
            assert result.full_text == "testo"
            # clear cache between iterations
            (test_config.TRANSCRIPTIONS_DIR / f"file.json").unlink(missing_ok=True)
