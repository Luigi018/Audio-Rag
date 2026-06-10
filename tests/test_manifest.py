"""Unit tests for ManifestManager."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.audio_rag.dataset_generator.manifest import (
    DatasetManifest,
    GroundTruthQuery,
    ManifestEntry,
    ManifestManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(**overrides) -> ManifestEntry:
    defaults = dict(
        script_id="it_f_001_politica",
        filename="it_female_001_politica.wav",
        file_path="/data/synthetic_dataset/audio/it_female_001_politica.wav",
        language="it",
        gender="female",
        voice="if_sara",
        topic="politica_italiana",
        duration_seconds=5.2,
        expected_keywords=["politica italiana", "governo"],
    )
    defaults.update(overrides)
    return ManifestEntry(**defaults)


def _make_manifest(entries: list[ManifestEntry] | None = None) -> DatasetManifest:
    entries = entries or [_make_entry()]
    return DatasetManifest(
        version="1.0",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        total_audio=len(entries),
        entries=entries,
    )


@pytest.fixture()
def mgr() -> ManifestManager:
    return ManifestManager()


# ---------------------------------------------------------------------------
# save() + load() round-trip
# ---------------------------------------------------------------------------

def test_save_creates_file(mgr, tmp_path):
    path = tmp_path / "manifest.json"
    mgr.save(_make_manifest(), path)
    assert path.exists()


def test_round_trip_preserves_all_fields(mgr, tmp_path):
    path = tmp_path / "manifest.json"
    original = _make_manifest()
    mgr.save(original, path)
    loaded = mgr.load(path)

    assert loaded.version == original.version
    assert loaded.created_at == original.created_at
    assert loaded.total_audio == original.total_audio
    assert len(loaded.entries) == len(original.entries)
    e = loaded.entries[0]
    assert e.script_id == "it_f_001_politica"
    assert e.filename == "it_female_001_politica.wav"
    assert e.language == "it"
    assert e.gender == "female"
    assert e.voice == "if_sara"
    assert e.topic == "politica_italiana"
    assert e.duration_seconds == pytest.approx(5.2)
    assert e.expected_keywords == ["politica italiana", "governo"]


def test_round_trip_multiple_entries(mgr, tmp_path):
    path = tmp_path / "manifest.json"
    entries = [
        _make_entry(script_id="it_f_001_politica", filename="it_female_001_politica.wav"),
        _make_entry(script_id="en_m_001_technology", filename="en_male_001_technology.wav",
                    language="en", gender="male", voice="am_adam", topic="technology",
                    expected_keywords=["technology", "AI"]),
    ]
    mgr.save(_make_manifest(entries), path)
    loaded = mgr.load(path)
    assert loaded.total_audio == 2
    assert loaded.entries[1].script_id == "en_m_001_technology"


# ---------------------------------------------------------------------------
# load() error handling
# ---------------------------------------------------------------------------

def test_load_missing_file_raises_file_not_found(mgr, tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        mgr.load(tmp_path / "nonexistent.json")


def test_load_corrupted_json_raises_value_error(mgr, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json }", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted"):
        mgr.load(bad)


def test_load_missing_key_raises_value_error(mgr, tmp_path):
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"entries": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        mgr.load(incomplete)


# ---------------------------------------------------------------------------
# created_at is ISO 8601
# ---------------------------------------------------------------------------

def test_created_at_is_iso8601(mgr, tmp_path):
    path = tmp_path / "manifest.json"
    mgr.save(_make_manifest(), path)
    loaded = mgr.load(path)
    # datetime.fromisoformat raises ValueError if not valid ISO 8601
    parsed = datetime.fromisoformat(loaded.created_at)
    assert parsed.tzinfo is not None  # must be timezone-aware


# ---------------------------------------------------------------------------
# get_ground_truth_queries()
# ---------------------------------------------------------------------------

def test_ground_truth_queries_one_per_topic(mgr):
    entries = [
        _make_entry(script_id="a1", filename="a1.wav", topic="politica_italiana"),
        _make_entry(script_id="a2", filename="a2.wav", topic="politica_italiana"),
        _make_entry(script_id="b1", filename="b1.wav", topic="cucina_italiana",
                    language="it", expected_keywords=["cucina italiana", "piatti"]),
    ]
    manifest = _make_manifest(entries)
    queries = mgr.get_ground_truth_queries(manifest)
    assert len(queries) == 2  # two distinct topics


def test_ground_truth_queries_have_expected_source_files(mgr):
    entries = [
        _make_entry(script_id="a1", filename="a1.wav", topic="sport"),
        _make_entry(script_id="a2", filename="a2.wav", topic="sport"),
    ]
    queries = mgr.get_ground_truth_queries(_make_manifest(entries))
    sport_query = next(q for q in queries if q.topic == "sport")
    assert "a1.wav" in sport_query.expected_source_files
    assert "a2.wav" in sport_query.expected_source_files


def test_ground_truth_queries_italian_language(mgr):
    entries = [_make_entry(language="it", expected_keywords=["politica italiana", "governo"])]
    queries = mgr.get_ground_truth_queries(_make_manifest(entries))
    assert queries[0].query.startswith("Dove si parla di")


def test_ground_truth_queries_english_language(mgr):
    entries = [_make_entry(language="en", expected_keywords=["technology", "AI"])]
    queries = mgr.get_ground_truth_queries(_make_manifest(entries))
    assert queries[0].query.startswith("Which audio files")


def test_ground_truth_query_is_dataclass(mgr):
    queries = mgr.get_ground_truth_queries(_make_manifest())
    assert isinstance(queries[0], GroundTruthQuery)


# ---------------------------------------------------------------------------
# make_empty()
# ---------------------------------------------------------------------------

def test_make_empty_returns_valid_manifest(mgr):
    m = ManifestManager.make_empty()
    assert m.total_audio == 0
    assert m.entries == []
    assert m.version == ManifestManager.VERSION
