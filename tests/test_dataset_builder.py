"""Unit tests for DatasetBuilder."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.audio_rag.config import Config
from src.audio_rag.dataset_generator.manifest import DatasetManifest, ManifestEntry
from src.audio_rag.dataset_generator.script_builder import ScriptBuilder
from src.audio_rag.dataset_generator.tts_engine import AudioMeta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dataset_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.SYNTHETIC_DATASET_DIR = tmp_path / "synthetic_dataset"
    cfg.SYNTHETIC_AUDIO_DIR = tmp_path / "synthetic_dataset" / "audio"
    cfg.MANIFEST_PATH = tmp_path / "synthetic_dataset" / "manifest.json"
    cfg.GROUND_TRUTH_QUERIES_PATH = tmp_path / "synthetic_dataset" / "ground_truth_queries.json"
    return cfg


@pytest.fixture()
def mock_tts() -> MagicMock:
    tts = MagicMock()

    def fake_synthesize_and_save(text, voice, lang, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"FAKEWAV")
        return AudioMeta(
            file_path=output_path,
            voice=voice,
            language=lang,
            duration_seconds=3.5,
            sample_rate=24000,
        )

    tts.synthesize_and_save.side_effect = fake_synthesize_and_save
    return tts


@pytest.fixture()
def builder(mock_tts, dataset_config) -> "DatasetBuilder":
    from src.audio_rag.dataset_generator.dataset_builder import DatasetBuilder
    return DatasetBuilder(
        tts_engine=mock_tts,
        script_builder=ScriptBuilder(),
        config=dataset_config,
    )


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

def test_build_calls_synthesize_for_all_scripts(builder, mock_tts):
    manifest = builder.build()
    assert mock_tts.synthesize_and_save.call_count == 25


def test_build_returns_manifest_with_25_entries(builder):
    manifest = builder.build()
    assert isinstance(manifest, DatasetManifest)
    assert manifest.total_audio == 25


def test_build_saves_manifest_to_disk(builder, dataset_config):
    builder.build()
    assert dataset_config.MANIFEST_PATH.exists()


def test_build_creates_wav_files(builder, dataset_config):
    builder.build()
    wav_files = list(dataset_config.SYNTHETIC_AUDIO_DIR.glob("*.wav"))
    assert len(wav_files) == 25


# ---------------------------------------------------------------------------
# Skip existing files (overwrite=False)
# ---------------------------------------------------------------------------

def test_build_skips_existing_files(builder, mock_tts, dataset_config):
    audio_dir = dataset_config.SYNTHETIC_AUDIO_DIR
    audio_dir.mkdir(parents=True, exist_ok=True)

    scripts = ScriptBuilder().get_all_scripts()
    existing = scripts[0]
    (audio_dir / existing.filename).write_bytes(b"EXISTING")

    builder.build(overwrite=False)
    # Should be called 24 times (25 - 1 existing)
    assert mock_tts.synthesize_and_save.call_count == 24


def test_build_overwrite_true_regenerates_all(builder, mock_tts, dataset_config):
    audio_dir = dataset_config.SYNTHETIC_AUDIO_DIR
    audio_dir.mkdir(parents=True, exist_ok=True)
    scripts = ScriptBuilder().get_all_scripts()
    for s in scripts[:5]:
        (audio_dir / s.filename).write_bytes(b"OLD")

    builder.build(overwrite=True)
    assert mock_tts.synthesize_and_save.call_count == 25


# ---------------------------------------------------------------------------
# build_subset()
# ---------------------------------------------------------------------------

def test_build_subset_only_processes_given_ids(builder, mock_tts):
    ids = ["it_f_001_politica", "en_m_001_technology"]
    manifest = builder.build_subset(ids)
    assert mock_tts.synthesize_and_save.call_count == 2
    assert manifest.total_audio == 2


def test_build_subset_unknown_id_is_silently_ignored(builder, mock_tts):
    manifest = builder.build_subset(["it_f_001_politica", "nonexistent_id"])
    assert mock_tts.synthesize_and_save.call_count == 1
    assert manifest.total_audio == 1


def test_build_subset_empty_list_produces_empty_manifest(builder, mock_tts):
    manifest = builder.build_subset([])
    assert manifest.total_audio == 0
    mock_tts.synthesize_and_save.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling: one failure does not stop the rest
# ---------------------------------------------------------------------------

def test_build_continues_after_single_failure(builder, mock_tts):
    scripts = ScriptBuilder().get_all_scripts()
    failing_filename = scripts[0].filename

    original_side_effect = mock_tts.synthesize_and_save.side_effect

    def selective_failure(text, voice, lang, output_path):
        if output_path.name == failing_filename:
            raise RuntimeError("Simulated TTS failure")
        return original_side_effect(text=text, voice=voice, lang=lang, output_path=output_path)

    mock_tts.synthesize_and_save.side_effect = selective_failure
    manifest = builder.build()
    # 24 succeed, 1 fails → 24 in manifest
    assert manifest.total_audio == 24


# ---------------------------------------------------------------------------
# Manifest entry content
# ---------------------------------------------------------------------------

def test_manifest_entries_have_correct_language(builder):
    manifest = builder.build()
    it_entries = [e for e in manifest.entries if e.language == "it"]
    en_entries = [e for e in manifest.entries if e.language == "en"]
    assert len(it_entries) == 13
    assert len(en_entries) == 12


def test_manifest_entries_have_duration(builder):
    manifest = builder.build()
    for entry in manifest.entries:
        assert entry.duration_seconds > 0
