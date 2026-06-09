"""LLM-as-a-Judge evaluation module."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore[assignment]

from .config import Config
from .generator import GeneratedAnswer

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    reference_score: float
    completeness_score: float
    hallucination_detected: bool
    feedback: str
    raw_response: str


_JUDGE_PROMPT = """\
Sei un valutatore imparziale di sistemi RAG (Retrieval-Augmented Generation).

Devi valutare la qualità della risposta generata da un sistema RAG rispetto alla domanda posta e al testo di riferimento (ground truth).

--- DOMANDA ---
{query}

--- RISPOSTA DEL SISTEMA ---
{system_output}

--- RIFERIMENTI CITATI ---
{references}

--- GROUND TRUTH ---
{ground_truth}

---

Valuta la risposta sui seguenti criteri. Rispondi ESATTAMENTE nel formato indicato:

REFERENCE_SCORE: <numero da 0 a 10>
COMPLETENESS_SCORE: <numero da 0 a 10>
HALLUCINATION: <SI o NO>
FEEDBACK: <testo libero di valutazione in italiano>

Definizioni:
- REFERENCE_SCORE: quanto le reference citate corrispondono alle fonti reali nel ground truth (0=nessuna corrispondenza, 10=perfetta).
- COMPLETENESS_SCORE: quanto la risposta copre completamente le informazioni nel ground truth (0=nulla, 10=tutto).
- HALLUCINATION: SI se la risposta contiene informazioni inventate non presenti nel ground truth, NO altrimenti.
- FEEDBACK: commento qualitativo sui punti di forza e debolezza della risposta.
"""


class LLMJudge:
    """Evaluates RAG output quality using Ollama as the judge model."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()

    def _parse_response(self, raw: str) -> tuple[float, float, bool, str]:
        """Extract structured scores from the LLM judge response.

        Returns:
            Tuple of (reference_score, completeness_score, hallucination_detected, feedback).
        """
        def _extract_float(pattern: str, text: str, default: float) -> float:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1).replace(",", "."))
                    return max(0.0, min(10.0, value))
                except ValueError:
                    pass
            return default

        ref_score = _extract_float(r"REFERENCE_SCORE\s*:\s*([\d.,]+)", raw, 0.0)
        comp_score = _extract_float(r"COMPLETENESS_SCORE\s*:\s*([\d.,]+)", raw, 0.0)

        hall_match = re.search(r"HALLUCINATION\s*:\s*(SI|NO)", raw, re.IGNORECASE)
        hallucination = hall_match.group(1).upper() == "SI" if hall_match else False

        feedback_match = re.search(r"FEEDBACK\s*:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else raw.strip()

        return ref_score, comp_score, hallucination, feedback

    def evaluate(
        self,
        query: str,
        system_output: GeneratedAnswer,
        ground_truth: str,
    ) -> JudgeResult:
        """Evaluate system output against a ground truth using an LLM judge.

        Args:
            query: The original user question.
            system_output: GeneratedAnswer produced by the pipeline.
            ground_truth: Expected / reference answer text.

        Returns:
            JudgeResult with numeric scores, hallucination flag and feedback.
        """
        if ollama is None:
            raise ImportError(
                "ollama Python SDK is required. Install with: pip install ollama"
            )

        references_text = system_output.format_references()
        prompt = _JUDGE_PROMPT.format(
            query=query,
            system_output=system_output.summary,
            references=references_text or "(nessuna reference)",
            ground_truth=ground_truth,
        )

        logger.info("Running LLM judge evaluation...")
        client = ollama.Client(host=self._config.OLLAMA_BASE_URL)
        response = client.chat(
            model=self._config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response["message"]["content"].strip()
        logger.debug("Judge raw response: %s", raw[:300])

        ref_score, comp_score, hallucination, feedback = self._parse_response(raw)
        return JudgeResult(
            reference_score=ref_score,
            completeness_score=comp_score,
            hallucination_detected=hallucination,
            feedback=feedback,
            raw_response=raw,
        )
