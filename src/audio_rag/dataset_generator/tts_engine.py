"""Kokoro TTS engine wrapper for synthetic audio generation."""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.audio_rag.config import Config

logger = logging.getLogger(__name__)

_KOKORO_RELEASES = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
_KOKORO_MODEL_URL = f"{_KOKORO_RELEASES}/kokoro-v1.0.int8.onnx"
_KOKORO_VOICES_URL = f"{_KOKORO_RELEASES}/voices-v1.0.bin"

_VALID_VOICES: frozenset[str] = frozenset({
    "af_heart",
    "af_bella",
    "am_adam",
    "am_michael",
    "if_sara",
    "im_nicola",
})

_LANG_MAP: dict[str, str] = {
    "it": "it",
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
}


@dataclass
class AudioMeta:
    file_path: Path
    voice: str
    language: str
    duration_seconds: float
    sample_rate: int


class TTSEngine:
    """Wraps kokoro-onnx for local text-to-speech synthesis.

    Model files are downloaded automatically on first use if absent from
    Config.KOKORO_MODEL_DIR.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: object | None = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _resolve_model_files(self) -> tuple[Path, Path]:
        """Return (model_path, voices_path), downloading if necessary."""
        model_dir = self._config.KOKORO_MODEL_DIR
        model_path = model_dir / self._config.KOKORO_MODEL_FILE
        voices_path = model_dir / self._config.KOKORO_VOICES_FILE
        model_dir.mkdir(parents=True, exist_ok=True)

        if not voices_path.exists():
            logger.info("Downloading Kokoro voices → %s", voices_path)
            _download_file(_KOKORO_VOICES_URL, voices_path)

        if not model_path.exists():
            logger.info(
                "Downloading Kokoro model (this may take a few minutes) → %s", model_path
            )
            _download_file(_KOKORO_MODEL_URL, model_path)

        return model_path, voices_path

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from kokoro_onnx import Kokoro  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "kokoro-onnx is not installed. Run: pip install kokoro-onnx"
            ) from exc
        model_path, voices_path = self._resolve_model_files()
        self._model = Kokoro(str(model_path), str(voices_path))
        logger.info("Kokoro model loaded from %s", model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_available_voices(self) -> list[str]:
        """Return voices available in the current installation."""
        try:
            self._ensure_model()
            model = self._model
            if hasattr(model, "get_voices"):
                return model.get_voices()  # type: ignore[union-attr]
            if hasattr(model, "voices"):
                return sorted(model.voices)
        except Exception:  # noqa: BLE001
            pass
        return sorted(_VALID_VOICES)

    def synthesize(self, text: str, voice: str, lang: str) -> np.ndarray:
        """Generate PCM audio and return it as a float32 numpy array."""
        if voice not in _VALID_VOICES:
            raise ValueError(
                f"Voice '{voice}' is not valid. "
                f"Choose from: {sorted(_VALID_VOICES)}"
            )
        self._ensure_model()
        kokoro_lang = _LANG_MAP.get(lang, lang)
        samples, _ = self._model.create(  # type: ignore[union-attr]
            text, voice=voice, speed=1.0, lang=kokoro_lang
        )
        return np.array(samples, dtype=np.float32)

    def save(
        self,
        audio: np.ndarray,
        output_path: Path,
        sample_rate: int = 24000,
    ) -> None:
        """Write a float32 numpy array to a WAV file via soundfile."""
        try:
            import soundfile as sf  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "soundfile is not installed. Run: pip install soundfile"
            ) from exc
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, samplerate=sample_rate)
        logger.debug("Saved audio: %s", output_path)

    def synthesize_and_save(
        self,
        text: str,
        voice: str,
        lang: str,
        output_path: Path,
    ) -> AudioMeta:
        """Synthesize, persist to disk, and return audio metadata."""
        sample_rate = self._config.TTS_SAMPLE_RATE
        audio = self.synthesize(text, voice, lang)
        self.save(audio, output_path, sample_rate=sample_rate)
        duration = len(audio) / sample_rate
        return AudioMeta(
            file_path=output_path,
            voice=voice,
            language=lang,
            duration_seconds=duration,
            sample_rate=sample_rate,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a simple progress indicator."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        def _hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                print(f"\r  {dest.name}: {pct}%  ", end="", flush=True)

        urllib.request.urlretrieve(url, str(tmp), reporthook=_hook)
        print()
        tmp.rename(dest)
        logger.info("Download complete: %s", dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
