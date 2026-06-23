"""Unit tests for TTSEngine."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.audio_rag.config import Config
import src.audio_rag.dataset_generator.tts_engine as _tts_mod


_FAKE_VOICES = ["af_bella", "af_heart", "am_adam", "am_michael", "if_sara", "im_nicola"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_kokoro(monkeypatch, tmp_path):
    """Inject a fake kokoro_onnx module and bypass model-file resolution."""
    instance = MagicMock()
    instance.get_voices.return_value = sorted(_FAKE_VOICES)
    instance.voices = _FAKE_VOICES  # fallback attribute used by older path
    instance.create.return_value = (np.zeros(24000, dtype=np.float32), 24000)

    mock_module = MagicMock()
    mock_module.Kokoro.return_value = instance
    monkeypatch.setitem(sys.modules, "kokoro_onnx", mock_module)

    # Skip filesystem/network model resolution
    dummy_model = tmp_path / "kokoro.onnx"
    dummy_voices = tmp_path / "voices.bin"
    dummy_model.write_bytes(b"")
    dummy_voices.write_bytes(b"")
    monkeypatch.setattr(
        _tts_mod.TTSEngine,
        "_resolve_model_files",
        lambda self: (dummy_model, dummy_voices),
    )

    return instance


@pytest.fixture()
def mock_soundfile(monkeypatch):
    """Inject a fake soundfile module."""
    sf = MagicMock()
    sf.info.return_value = MagicMock(duration=1.0)
    monkeypatch.setitem(sys.modules, "soundfile", sf)
    return sf


@pytest.fixture()
def engine(mock_kokoro, mock_soundfile):
    return _tts_mod.TTSEngine(Config())


# ---------------------------------------------------------------------------
# synthesize()
# ---------------------------------------------------------------------------

def test_synthesize_returns_float32_array(engine, mock_kokoro):
    audio = engine.synthesize("Hello world", voice="af_heart", lang="en")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32


def test_synthesize_output_has_nonzero_length(engine, mock_kokoro):
    mock_kokoro.create.return_value = (np.ones(48000, dtype=np.float32), 24000)
    audio = engine.synthesize("Hello world", voice="if_sara", lang="it")
    assert audio.shape == (48000,)


def test_synthesize_invalid_voice_raises(engine):
    with pytest.raises(ValueError, match="not valid"):
        engine.synthesize("text", voice="xx_unknown", lang="en")


def test_synthesize_invalid_voice_does_not_load_model(mock_soundfile, monkeypatch):
    """ValueError must be raised before the model is ever instantiated."""
    load_calls: list[str] = []

    mock_module = MagicMock()
    def recording_constructor(*args, **kwargs):
        load_calls.append("loaded")
        return MagicMock()
    mock_module.Kokoro.side_effect = recording_constructor
    monkeypatch.setitem(sys.modules, "kokoro_onnx", mock_module)

    eng = _tts_mod.TTSEngine(Config())
    with pytest.raises(ValueError):
        eng.synthesize("text", voice="bad_voice", lang="en")
    assert load_calls == [], "Model should not be loaded for an invalid voice"


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

def test_save_calls_soundfile_write(engine, mock_soundfile, tmp_path):
    audio = np.zeros(24000, dtype=np.float32)
    out = tmp_path / "out.wav"
    engine.save(audio, out, sample_rate=24000)
    mock_soundfile.write.assert_called_once_with(str(out), audio, samplerate=24000)


def test_save_creates_parent_dirs(engine, mock_soundfile, tmp_path):
    out = tmp_path / "nested" / "deep" / "out.wav"
    engine.save(np.zeros(10, dtype=np.float32), out)
    assert out.parent.exists()


# ---------------------------------------------------------------------------
# synthesize_and_save()
# ---------------------------------------------------------------------------

def test_synthesize_and_save_returns_correct_meta(engine, mock_kokoro, mock_soundfile, tmp_path):
    mock_kokoro.create.return_value = (np.zeros(48000, dtype=np.float32), 24000)
    out = tmp_path / "audio.wav"
    meta = engine.synthesize_and_save("Hello", voice="af_heart", lang="en", output_path=out)

    assert meta.voice == "af_heart"
    assert meta.language == "en"
    assert meta.sample_rate == 24000
    assert pytest.approx(meta.duration_seconds, rel=1e-3) == 48000 / 24000


def test_synthesize_and_save_duration_from_array_length(engine, mock_kokoro, mock_soundfile, tmp_path):
    mock_kokoro.create.return_value = (np.zeros(72000, dtype=np.float32), 24000)
    meta = engine.synthesize_and_save("x", "im_nicola", "it", tmp_path / "x.wav")
    assert pytest.approx(meta.duration_seconds) == 3.0


# ---------------------------------------------------------------------------
# list_available_voices()
# ---------------------------------------------------------------------------

def test_list_available_voices_not_empty(engine, mock_kokoro):
    voices = engine.list_available_voices()
    assert len(voices) > 0


def test_list_available_voices_uses_get_voices(engine, mock_kokoro):
    """Prefers get_voices() over the .voices attribute."""
    voices = engine.list_available_voices()
    mock_kokoro.get_voices.assert_called()
    assert voices == sorted(_FAKE_VOICES)


def test_list_available_voices_sorted(engine, mock_kokoro):
    voices = engine.list_available_voices()
    assert voices == sorted(voices)


def test_list_available_voices_fallback_when_model_unavailable(monkeypatch):
    """When kokoro_onnx is absent, fall back to the hardcoded set."""
    monkeypatch.setitem(sys.modules, "kokoro_onnx", None)

    from importlib import reload
    reload(_tts_mod)

    eng = _tts_mod.TTSEngine(Config())
    voices = eng.list_available_voices()
    assert isinstance(voices, list)
    assert len(voices) > 0
