"""Tests for judge.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.audio_rag.config import Config
from src.audio_rag.generator import GeneratedAnswer, Reference
from src.audio_rag.judge import InlineEvalResult, JudgeResult, LLMJudge


@pytest.fixture()
def judge(test_config: Config) -> LLMJudge:
    return LLMJudge(test_config)


def _mock_response(text: str) -> dict:
    return {"message": {"content": text}}


_VALID_RESPONSE = """
REFERENCE_SCORE: 8
COMPLETENESS_SCORE: 7
HALLUCINATION: NO
FEEDBACK: The answer is accurate and well-structured, with good source citations.
"""

_HALLUCINATION_RESPONSE = """
REFERENCE_SCORE: 3
COMPLETENESS_SCORE: 5
HALLUCINATION: YES
FEEDBACK: The answer contains information not present in the ground truth.
"""


class TestLLMJudgeParsing:
    def test_parse_valid_response(self, judge: LLMJudge) -> None:
        ref, comp, hall, feedback = judge._parse_response(_VALID_RESPONSE)
        assert ref == 8.0
        assert comp == 7.0
        assert hall is False
        assert "accurate" in feedback

    def test_parse_hallucination_yes(self, judge: LLMJudge) -> None:
        _, _, hall, _ = judge._parse_response(_HALLUCINATION_RESPONSE)
        assert hall is True

    def test_parse_score_clamped_above_10(self, judge: LLMJudge) -> None:
        raw = "REFERENCE_SCORE: 15\nCOMPLETENESS_SCORE: 12\nHALLUCINATION: NO\nFEEDBACK: ok"
        ref, comp, _, _ = judge._parse_response(raw)
        assert ref == 10.0
        assert comp == 10.0

    def test_parse_score_clamped_below_0(self, judge: LLMJudge) -> None:
        raw = "REFERENCE_SCORE: -3\nCOMPLETENESS_SCORE: -1\nHALLUCINATION: NO\nFEEDBACK: ok"
        ref, comp, _, _ = judge._parse_response(raw)
        assert ref == 0.0
        assert comp == 0.0

    def test_parse_missing_scores_defaults_to_zero(self, judge: LLMJudge) -> None:
        raw = "HALLUCINATION: NO\nFEEDBACK: fallback"
        ref, comp, _, _ = judge._parse_response(raw)
        assert ref == 0.0
        assert comp == 0.0

    def test_parse_decimal_score(self, judge: LLMJudge) -> None:
        raw = "REFERENCE_SCORE: 7.5\nCOMPLETENESS_SCORE: 8.5\nHALLUCINATION: NO\nFEEDBACK: ok"
        ref, comp, _, _ = judge._parse_response(raw)
        assert ref == 7.5
        assert comp == 8.5

    def test_parse_missing_hallucination_defaults_to_false(self, judge: LLMJudge) -> None:
        raw = "REFERENCE_SCORE: 5\nCOMPLETENESS_SCORE: 5\nFEEDBACK: no info"
        _, _, hall, _ = judge._parse_response(raw)
        assert hall is False

    def test_parse_case_insensitive_hallucination(self, judge: LLMJudge) -> None:
        raw = "REFERENCE_SCORE: 5\nCOMPLETENESS_SCORE: 5\nHALLUCINATION: yes\nFEEDBACK: ok"
        _, _, hall, _ = judge._parse_response(raw)
        assert hall is True


class TestLLMJudgeEvaluate:
    def test_evaluate_happy_path(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_response(_VALID_RESPONSE)

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            result = judge.evaluate("question", sample_answer, "reference ground truth")

        assert isinstance(result, JudgeResult)
        assert result.reference_score == 8.0
        assert result.completeness_score == 7.0
        assert result.hallucination_detected is False
        assert "accurate" in result.feedback

    def test_evaluate_stores_raw_response(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_response(_VALID_RESPONSE)

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            result = judge.evaluate("q", sample_answer, "gt")

        assert result.raw_response.strip() == _VALID_RESPONSE.strip()

    def test_evaluate_prompt_contains_query_and_ground_truth(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        captured = {}
        mock_client = MagicMock()

        def capture(model, messages):
            captured["content"] = messages[0]["content"]
            return _mock_response(_VALID_RESPONSE)

        mock_client.chat.side_effect = capture

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            judge.evaluate("my question", sample_answer, "my ground truth")

        assert "my question" in captured["content"]
        assert "my ground truth" in captured["content"]

    def test_evaluate_missing_ollama_raises(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        with patch("src.audio_rag.judge.ollama", None):
            with pytest.raises(ImportError, match="ollama"):
                judge.evaluate("q", sample_answer, "gt")

    def test_evaluate_hallucination_detected(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_response(_HALLUCINATION_RESPONSE)

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            result = judge.evaluate("q", sample_answer, "gt")

        assert result.hallucination_detected is True
        assert result.reference_score == 3.0


# ── Inline evaluation (no ground truth) ──────────────────────────────────────

_INLINE_VALID_RESPONSE = """
SCORE: 0.85
FAITHFULNESS: YES
RELEVANCE: YES
EXPLANATION: The answer is well-grounded and directly addresses the question.
"""

_INLINE_UNFAITHFUL_RESPONSE = """
SCORE: 0.2
FAITHFULNESS: NO
RELEVANCE: YES
EXPLANATION: The answer contains claims not supported by the retrieved context.
"""


class TestLLMJudgeEvaluateInline:
    def test_happy_path_returns_inline_eval_result(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_response(_INLINE_VALID_RESPONSE)

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            result = judge.evaluate_inline("question", sample_answer)

        assert isinstance(result, InlineEvalResult)
        assert result.score == pytest.approx(0.85)
        assert result.faithfulness is True
        assert result.relevance is True
        assert "grounded" in result.explanation

    def test_faithfulness_false_when_no_in_response(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_response(_INLINE_UNFAITHFUL_RESPONSE)

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            result = judge.evaluate_inline("q", sample_answer)

        assert result.faithfulness is False
        assert result.relevance is True
        assert result.score == pytest.approx(0.2)

    def test_score_clamped_to_1(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        raw = "SCORE: 1.5\nFAITHFULNESS: YES\nRELEVANCE: YES\nEXPLANATION: ok"
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_response(raw)

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            result = judge.evaluate_inline("q", sample_answer)

        assert result.score == 1.0

    def test_score_clamped_to_0(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        raw = "SCORE: -0.5\nFAITHFULNESS: NO\nRELEVANCE: NO\nEXPLANATION: ok"
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_response(raw)

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            result = judge.evaluate_inline("q", sample_answer)

        assert result.score == 0.0

    def test_missing_fields_default_safely(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_response("nothing parseable here")

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            result = judge.evaluate_inline("q", sample_answer)

        assert result.score == 0.0
        assert result.faithfulness is False
        assert result.relevance is False

    def test_missing_ollama_raises(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        with patch("src.audio_rag.judge.ollama", None):
            with pytest.raises(ImportError, match="ollama"):
                judge.evaluate_inline("q", sample_answer)

    def test_prompt_contains_query_and_context(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        captured: dict = {}
        mock_client = MagicMock()

        def capture(model, messages):
            captured["content"] = messages[0]["content"]
            return _mock_response(_INLINE_VALID_RESPONSE)

        mock_client.chat.side_effect = capture

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            judge.evaluate_inline("my specific question", sample_answer)

        assert "my specific question" in captured["content"]
        assert sample_answer.raw_context in captured["content"]

    def test_stores_raw_response(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_response(_INLINE_VALID_RESPONSE)

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            result = judge.evaluate_inline("q", sample_answer)

        assert result.raw_response.strip() == _INLINE_VALID_RESPONSE.strip()

    def test_case_insensitive_yes_no(
        self, judge: LLMJudge, sample_answer: GeneratedAnswer
    ) -> None:
        raw = "SCORE: 0.5\nFAITHFULNESS: yes\nRELEVANCE: no\nEXPLANATION: ok"
        mock_client = MagicMock()
        mock_client.chat.return_value = _mock_response(raw)

        with patch("src.audio_rag.judge.ollama") as mock_ollama:
            mock_ollama.Client.return_value = mock_client
            result = judge.evaluate_inline("q", sample_answer)

        assert result.faithfulness is True
        assert result.relevance is False
