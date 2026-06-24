"""Tests for evaluate_retrieval.py (root-level benchmark script)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluate_retrieval import _chunk_id, _run_benchmark


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["question", "expected_retrieval"])
        writer.writeheader()
        writer.writerows(rows)


def _make_chunk(filename: str, chunk_index: int) -> MagicMock:
    chunk = MagicMock()
    chunk.source_file = Path(filename)
    chunk.chunk_index = chunk_index
    return chunk


def _run(tmp_path: Path, rows: list[dict], chunks_per_query: list[list], top_k: int = 5) -> dict:
    """Write CSV, mock Retriever, run benchmark, return parsed summary JSON."""
    csv_path = tmp_path / "eval.csv"
    _write_csv(csv_path, rows)
    output_dir = tmp_path / "out"

    mock_retriever = MagicMock()
    mock_retriever.search.side_effect = chunks_per_query

    with patch("src.audio_rag.retriever.Retriever", return_value=mock_retriever):
        _run_benchmark(csv_path, top_k, output_dir)

    summaries = list(output_dir.glob("*_summary.json"))
    assert len(summaries) == 1
    return json.loads(summaries[0].read_text(encoding="utf-8"))


# ── _chunk_id ─────────────────────────────────────────────────────────────────

class TestChunkId:
    def test_basic_format(self) -> None:
        assert _chunk_id(Path("audio.wav"), 3) == "audio.wav::chunk::3"

    def test_uses_filename_not_full_path(self) -> None:
        assert _chunk_id(Path("/deep/path/audio.mp3"), 0) == "audio.mp3::chunk::0"

    def test_chunk_index_zero(self) -> None:
        assert _chunk_id(Path("file.flac"), 0) == "file.flac::chunk::0"


# ── CSV validation ────────────────────────────────────────────────────────────

class TestCsvValidation:
    def test_nonexistent_file_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            _run_benchmark(tmp_path / "missing.csv", 5, tmp_path / "out")

    def test_missing_question_column_exits(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        with bad.open("w") as f:
            f.write("wrong,expected_retrieval\nval,val\n")
        with pytest.raises(SystemExit):
            _run_benchmark(bad, 5, tmp_path / "out")

    def test_missing_expected_retrieval_column_exits(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        with bad.open("w") as f:
            f.write("question,wrong\nval,val\n")
        with pytest.raises(SystemExit):
            _run_benchmark(bad, 5, tmp_path / "out")

    def test_empty_rows_exits(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        _write_csv(empty, [])
        with pytest.raises(SystemExit):
            _run_benchmark(empty, 5, tmp_path / "out")

    def test_rows_with_blank_fields_are_skipped(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sparse.csv"
        with csv_path.open("w", newline="") as f:
            f.write("question,expected_retrieval\n")
            f.write(",\n")           # both blank — skipped
            f.write("q1,a.wav::chunk::0\n")  # valid

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [_make_chunk("a.wav", 0)]

        with patch("src.audio_rag.retriever.Retriever", return_value=mock_retriever):
            _run_benchmark(csv_path, 5, tmp_path / "out")

        # Only 1 valid row — retriever called once
        mock_retriever.search.assert_called_once()


# ── Metric computation ────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_hit_rate(self, tmp_path: Path) -> None:
        rows = [{"question": "q1", "expected_retrieval": "a.wav::chunk::0"}]
        chunks = [[_make_chunk("a.wav", 0)]]
        summary = _run(tmp_path, rows, chunks)
        assert summary["hit_rate"] == 1.0
        assert summary["exact_match"] == 1.0
        assert summary["mrr"] == 1.0

    def test_zero_hit_rate(self, tmp_path: Path) -> None:
        rows = [{"question": "q1", "expected_retrieval": "expected.wav::chunk::0"}]
        chunks = [[_make_chunk("other.wav", 0)]]
        summary = _run(tmp_path, rows, chunks)
        assert summary["hit_rate"] == 0.0
        assert summary["exact_match"] == 0.0
        assert summary["mrr"] == 0.0

    def test_mrr_rank_2(self, tmp_path: Path) -> None:
        rows = [{"question": "q1", "expected_retrieval": "a.wav::chunk::0"}]
        chunks = [[_make_chunk("other.wav", 5), _make_chunk("a.wav", 0)]]
        summary = _run(tmp_path, rows, chunks)
        assert summary["mrr"] == pytest.approx(0.5)
        assert summary["hit_rate"] == 1.0
        assert summary["exact_match"] == 0.0

    def test_mrr_rank_3(self, tmp_path: Path) -> None:
        rows = [{"question": "q1", "expected_retrieval": "a.wav::chunk::0"}]
        chunks = [[
            _make_chunk("x.wav", 0),
            _make_chunk("y.wav", 0),
            _make_chunk("a.wav", 0),
        ]]
        summary = _run(tmp_path, rows, chunks)
        assert summary["mrr"] == pytest.approx(1 / 3, abs=1e-4)

    def test_partial_hits_two_questions(self, tmp_path: Path) -> None:
        rows = [
            {"question": "q1", "expected_retrieval": "a.wav::chunk::0"},
            {"question": "q2", "expected_retrieval": "b.wav::chunk::0"},
        ]
        chunks = [
            [_make_chunk("a.wav", 0)],   # hit
            [_make_chunk("c.wav", 0)],   # miss
        ]
        summary = _run(tmp_path, rows, chunks)
        assert summary["hit_rate"] == pytest.approx(0.5)
        assert summary["total"] == 2
        assert summary["hits"] == 1
        assert summary["exact_matches"] == 1

    def test_empty_retrieval_counts_as_miss(self, tmp_path: Path) -> None:
        rows = [{"question": "q1", "expected_retrieval": "a.wav::chunk::0"}]
        chunks = [[]]
        summary = _run(tmp_path, rows, chunks)
        assert summary["hit_rate"] == 0.0
        assert summary["mrr"] == 0.0

    def test_top_k_forwarded_to_retriever(self, tmp_path: Path) -> None:
        rows = [{"question": "q1", "expected_retrieval": "a.wav::chunk::0"}]
        csv_path = tmp_path / "eval.csv"
        _write_csv(csv_path, rows)

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [_make_chunk("a.wav", 0)]

        with patch("src.audio_rag.retriever.Retriever", return_value=mock_retriever):
            _run_benchmark(csv_path, 10, tmp_path / "out")

        mock_retriever.search.assert_called_once_with("q1", top_k=10)


# ── Output files ──────────────────────────────────────────────────────────────

class TestOutputFiles:
    def test_csv_and_json_written(self, tmp_path: Path) -> None:
        rows = [{"question": "q1", "expected_retrieval": "a.wav::chunk::0"}]
        chunks = [[_make_chunk("a.wav", 0)]]
        _run(tmp_path, rows, chunks)
        out = tmp_path / "out"
        assert len(list(out.glob("retrieval_benchmark_*.csv"))) == 1
        assert len(list(out.glob("*_summary.json"))) == 1

    def test_csv_contains_per_row_detail(self, tmp_path: Path) -> None:
        rows = [{"question": "q1", "expected_retrieval": "a.wav::chunk::0"}]
        csv_path = tmp_path / "eval.csv"
        _write_csv(csv_path, rows)
        output_dir = tmp_path / "out"

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [_make_chunk("a.wav", 0)]

        with patch("src.audio_rag.retriever.Retriever", return_value=mock_retriever):
            _run_benchmark(csv_path, 5, output_dir)

        report = list(output_dir.glob("retrieval_benchmark_*.csv"))[0]
        rows_out = list(csv.DictReader(report.open(encoding="utf-8")))
        assert len(rows_out) == 1
        assert rows_out[0]["question"] == "q1"
        assert rows_out[0]["expected"] == "a.wav::chunk::0"
        assert rows_out[0]["hit"] == "True"
        assert rows_out[0]["rank"] == "1"

    def test_output_dir_created_if_missing(self, tmp_path: Path) -> None:
        rows = [{"question": "q1", "expected_retrieval": "a.wav::chunk::0"}]
        chunks = [[_make_chunk("a.wav", 0)]]
        new_dir = tmp_path / "nested" / "new"
        csv_path = tmp_path / "eval.csv"
        _write_csv(csv_path, rows)

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = chunks

        with patch("src.audio_rag.retriever.Retriever", return_value=mock_retriever):
            _run_benchmark(csv_path, 5, new_dir)

        assert new_dir.exists()

    def test_summary_json_structure(self, tmp_path: Path) -> None:
        rows = [{"question": "q1", "expected_retrieval": "a.wav::chunk::0"}]
        summary = _run(tmp_path, rows, [[_make_chunk("a.wav", 0)]])
        assert set(summary.keys()) == {
            "total", "top_k", "hit_rate", "mrr", "exact_match", "hits", "exact_matches"
        }
        assert summary["top_k"] == 5
