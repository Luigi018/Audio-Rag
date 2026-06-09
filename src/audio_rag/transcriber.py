"""Audio transcription with faster-whisper."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    file_path: Path
    full_text: str
    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "file_path": str(self.file_path),
            "full_text": self.full_text,
            "segments": [asdict(s) for s in self.segments],
            "language": self.language,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptionResult":
        return cls(
            file_path=Path(data["file_path"]),
            full_text=data["full_text"],
            segments=[Segment(**s) for s in data.get("segments", [])],
            language=data.get("language", ""),
            duration=data.get("duration", 0.0),
        )


class Transcriber:
    """Transcribes audio files using faster-whisper."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        self._model = None
        self._cache_dir = self._config.TRANSCRIPTIONS_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required. Install it with: pip install faster-whisper"
            ) from exc

        device = self._config.WHISPER_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(
            "Loading Whisper model '%s' on %s (compute=%s)",
            self._config.WHISPER_MODEL,
            device,
            compute_type,
        )
        self._model = WhisperModel(
            self._config.WHISPER_MODEL, device=device, compute_type=compute_type
        )

    def _cache_path(self, audio_path: Path) -> Path:
        return self._cache_dir / (audio_path.stem + ".json")

    def _load_cached(self, audio_path: Path) -> TranscriptionResult | None:
        cache_file = self._cache_path(audio_path)
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            logger.debug("Cache hit for '%s'", audio_path.name)
            return TranscriptionResult.from_dict(data)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Corrupted cache for '%s': %s", audio_path.name, exc)
            return None

    def _save_cache(self, result: TranscriptionResult) -> None:
        cache_file = self._cache_path(result.file_path)
        cache_file.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("Cached transcription for '%s'", result.file_path.name)

    def transcribe_file(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe a single audio file, using cache when available.

        Args:
            audio_path: Path to the audio file.

        Returns:
            TranscriptionResult with full text, segments, language and duration.

        Raises:
            FileNotFoundError: If audio_path does not exist.
            ValueError: If the file extension is not supported.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.lower() not in self._config.AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{audio_path.suffix}'. "
                f"Supported: {self._config.AUDIO_EXTENSIONS}"
            )

        cached = self._load_cached(audio_path)
        if cached is not None:
            return cached

        self._load_model()
        logger.info("Transcribing '%s' ...", audio_path.name)

        segments_iter, info = self._model.transcribe(
            str(audio_path), beam_size=5, language=None
        )
        segments: list[Segment] = []
        texts: list[str] = []
        for seg in segments_iter:
            segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip()))
            texts.append(seg.text.strip())

        result = TranscriptionResult(
            file_path=audio_path,
            full_text=" ".join(texts),
            segments=segments,
            language=info.language,
            duration=info.duration,
        )
        self._save_cache(result)
        logger.info(
            "Transcribed '%s' in %s (%.1fs)",
            audio_path.name,
            result.language,
            result.duration,
        )
        return result

    def transcribe_all(self, input_dir: Path | None = None) -> list[TranscriptionResult]:
        """Transcribe all supported audio files in input_dir.

        Args:
            input_dir: Directory to scan. Defaults to Config.INPUT_DIR.

        Returns:
            List of TranscriptionResult, one per file.
        """
        search_dir = input_dir or self._config.INPUT_DIR
        if not search_dir.exists():
            logger.warning("Input directory '%s' does not exist.", search_dir)
            return []

        audio_files = [
            p
            for p in sorted(search_dir.iterdir())
            if p.is_file() and p.suffix.lower() in self._config.AUDIO_EXTENSIONS
        ]
        logger.info("Found %d audio file(s) in '%s'", len(audio_files), search_dir)

        results: list[TranscriptionResult] = []
        for audio_file in audio_files:
            try:
                results.append(self.transcribe_file(audio_file))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to transcribe '%s': %s", audio_file.name, exc)
        return results
