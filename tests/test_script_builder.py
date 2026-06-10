"""Unit tests for ScriptBuilder."""
from __future__ import annotations

import pytest

from src.audio_rag.dataset_generator.script_builder import AudioScript, ScriptBuilder


@pytest.fixture()
def builder() -> ScriptBuilder:
    return ScriptBuilder()


# ---------------------------------------------------------------------------
# Total count
# ---------------------------------------------------------------------------

def test_total_script_count(builder):
    assert len(builder.get_all_scripts()) == 25


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

def test_distribution_by_language(builder):
    scripts = builder.get_all_scripts()
    it_scripts = [s for s in scripts if s.language == "it"]
    en_scripts = [s for s in scripts if s.language == "en"]
    assert len(it_scripts) >= 6
    assert len(en_scripts) >= 6


def test_distribution_by_gender(builder):
    scripts = builder.get_all_scripts()
    female = [s for s in scripts if s.gender == "female"]
    male = [s for s in scripts if s.gender == "male"]
    assert len(female) >= 6
    assert len(male) >= 6


def test_italian_female_count(builder):
    scripts = [s for s in builder.get_all_scripts() if s.language == "it" and s.gender == "female"]
    assert len(scripts) == 6


def test_italian_male_count(builder):
    scripts = [s for s in builder.get_all_scripts() if s.language == "it" and s.gender == "male"]
    assert len(scripts) == 7


def test_english_female_count(builder):
    scripts = [s for s in builder.get_all_scripts() if s.language == "en" and s.gender == "female"]
    assert len(scripts) == 6


def test_english_male_count(builder):
    scripts = [s for s in builder.get_all_scripts() if s.language == "en" and s.gender == "male"]
    assert len(scripts) == 6


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------

def test_script_ids_are_unique(builder):
    ids = [s.script_id for s in builder.get_all_scripts()]
    assert len(ids) == len(set(ids))


def test_filenames_are_unique(builder):
    names = [s.filename for s in builder.get_all_scripts()]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Content validity
# ---------------------------------------------------------------------------

def test_all_filenames_end_with_wav(builder):
    for script in builder.get_all_scripts():
        assert script.filename.endswith(".wav"), f"{script.script_id} has non-WAV filename"


def test_each_script_has_at_least_two_keywords(builder):
    for script in builder.get_all_scripts():
        assert len(script.expected_keywords) >= 2, (
            f"{script.script_id} has fewer than 2 expected_keywords"
        )


def test_all_voices_are_known(builder):
    known = {"af_heart", "af_bella", "am_adam", "am_michael", "if_sara", "im_nicola"}
    for script in builder.get_all_scripts():
        assert script.voice in known, f"{script.script_id} uses unknown voice '{script.voice}'"


def test_text_length_within_bounds(builder):
    for script in builder.get_all_scripts():
        word_count = len(script.text.split())
        assert 80 <= word_count <= 200, (
            f"{script.script_id}: {word_count} words (expected 80–200)"
        )


def test_script_id_matches_language_prefix(builder):
    for script in builder.get_all_scripts():
        prefix = script.script_id[:2]
        assert prefix == script.language, (
            f"{script.script_id}: id prefix '{prefix}' != language '{script.language}'"
        )


# ---------------------------------------------------------------------------
# get_scripts_by_language()
# ---------------------------------------------------------------------------

def test_get_scripts_by_language_italian(builder):
    scripts = builder.get_scripts_by_language("it")
    assert all(s.language == "it" for s in scripts)
    assert len(scripts) == 13


def test_get_scripts_by_language_english(builder):
    scripts = builder.get_scripts_by_language("en")
    assert all(s.language == "en" for s in scripts)
    assert len(scripts) == 12


def test_get_scripts_by_language_unknown_returns_empty(builder):
    assert builder.get_scripts_by_language("fr") == []


# ---------------------------------------------------------------------------
# get_scripts_by_topic()
# ---------------------------------------------------------------------------

def test_get_scripts_by_topic_sport_returns_multiple(builder):
    scripts = builder.get_scripts_by_topic("sport")
    assert len(scripts) >= 2, "Expected at least 2 scripts with topic='sport'"


def test_get_scripts_by_topic_technology_returns_multiple(builder):
    scripts = builder.get_scripts_by_topic("technology")
    assert len(scripts) >= 2


def test_get_scripts_by_topic_economics_returns_multiple(builder):
    scripts = builder.get_scripts_by_topic("economics")
    assert len(scripts) >= 2


def test_get_scripts_by_topic_unknown_returns_empty(builder):
    assert builder.get_scripts_by_topic("quantum_computing") == []


# ---------------------------------------------------------------------------
# AudioScript dataclass structure
# ---------------------------------------------------------------------------

def test_all_scripts_are_audioscript_instances(builder):
    for script in builder.get_all_scripts():
        assert isinstance(script, AudioScript)
