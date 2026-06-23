"""Retrieval benchmark script for audio-rag.

For each row in the input CSV (columns: question, expected_retrieval),
runs the retriever and checks whether the expected chunk appears in the top-k results.

Chunk IDs follow the format:  <filename>::chunk::<index>
Example: it_female_001_politica.wav::chunk::3
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


def _chunk_id(source_file: Path, chunk_index: int) -> str:
    return f"{source_file.name}::chunk::{chunk_index}"


def _run_benchmark(csv_path: Path, top_k: int, output_dir: Path) -> None:
    from src.audio_rag.config import Config
    from src.audio_rag.retriever import Retriever

    cfg = Config()
    retriever = Retriever(config=cfg)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    try:
        with csv_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not {
                "question",
                "expected_retrieval",
            }.issubset(set(reader.fieldnames)):
                print(
                    "Error: CSV must contain columns 'question' and 'expected_retrieval'.",
                    file=sys.stderr,
                )
                sys.exit(1)
            rows = [
                row
                for row in reader
                if row.get("question", "").strip() and row.get("expected_retrieval", "").strip()
            ]
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("Error: CSV contains no valid rows.", file=sys.stderr)
        sys.exit(1)

    # ── Evaluate each question ────────────────────────────────────────────────
    hits = 0
    reciprocal_ranks: list[float] = []
    exact_matches = 0
    details: list[dict] = []

    for i, row in enumerate(rows, 1):
        question = row["question"].strip()
        expected = row["expected_retrieval"].strip()

        try:
            chunks = retriever.search(question, top_k=top_k)
        except Exception as exc:
            logging.warning("Retrieval failed for Q%d (%s): %s", i, question[:60], exc)
            chunks = []

        chunk_ids = [_chunk_id(c.source_file, c.chunk_index) for c in chunks]
        top1_id = chunk_ids[0] if chunk_ids else ""

        hit = expected in chunk_ids
        rank = (chunk_ids.index(expected) + 1) if hit else None
        exact = top1_id == expected

        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        if exact:
            exact_matches += 1

        details.append(
            {
                "question": question,
                "expected": expected,
                "retrieved_top1": top1_id,
                "hit": hit,
                "rank": rank,
            }
        )

    # ── Compute metrics ───────────────────────────────────────────────────────
    total = len(rows)
    hit_rate = hits / total
    mrr = sum(reciprocal_ranks) / total
    exact_rate = exact_matches / total

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nRetrieval Benchmark Results")
    print("=" * 60)
    print(f"Total questions : {total}")
    print(f"Hit Rate @{top_k:<5}: {hit_rate:.2f}  ({hits}/{total})")
    print(f"MRR             : {mrr:.2f}")
    print(f"Exact Match @1  : {exact_rate:.2f}  ({exact_matches}/{total})")

    print("\nPer-question breakdown:")
    for i, d in enumerate(details, 1):
        rank_str = str(d["rank"]) if d["rank"] is not None else "-"
        print(
            f"  Q{i:<3} | expected: {d['expected']:<35} | "
            f"retrieved: {d['retrieved_top1']:<35} | "
            f"hit: {str(d['hit']):<5} | rank: {rank_str}"
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = output_dir / f"retrieval_benchmark_{timestamp}.csv"
    with report_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["question", "expected", "retrieved_top1", "hit", "rank"],
        )
        writer.writeheader()
        writer.writerows(details)

    summary_path = output_dir / f"retrieval_benchmark_{timestamp}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "total": total,
                "top_k": top_k,
                "hit_rate": round(hit_rate, 4),
                "mrr": round(mrr, 4),
                "exact_match": round(exact_rate, 4),
                "hits": hits,
                "exact_matches": exact_matches,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {report_path}")
    print(f"Summary saved to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure retrieval accuracy against a labeled CSV dataset.\n\n"
            "Chunk ID format: <filename>::chunk::<index>\n"
            "Example:         it_female_001_politica.wav::chunk::3"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        required=True,
        metavar="PATH",
        help="Path to the input CSV file (columns: question, expected_retrieval).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of top results to consider for Hit Rate (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results/",
        metavar="DIR",
        help="Directory where the report CSV and summary JSON are saved (default: eval_results/).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    _run_benchmark(Path(args.csv), args.top_k, Path(args.output_dir))


if __name__ == "__main__":
    main()
