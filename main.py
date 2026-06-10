"""CLI entry point for audio-rag."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="audio-rag",
    help="RAG pipeline over local audio files using Whisper + ChromaDB + Ollama.",
    add_completion=False,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.command()
def ingest(
    input_dir: Optional[Path] = typer.Option(
        None, "--input-dir", "-i", help="Directory containing audio files."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Transcribe and index all audio files in the input directory."""
    _setup_logging(verbose)
    from src.audio_rag.config import Config
    from src.audio_rag.pipeline import AudioRAGPipeline

    cfg = Config()
    if input_dir:
        cfg.INPUT_DIR = input_dir

    pipeline = AudioRAGPipeline(cfg)
    n = pipeline.ingest()
    typer.echo(f"Ingestion complete: {n} chunk(s) indexed.")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask about the audio files."),
    input_dir: Optional[Path] = typer.Option(
        None, "--input-dir", "-i", help="Directory containing audio files (for ad-hoc ingest)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save answer as JSON to this file."
    ),
) -> None:
    """Query the indexed audio files with a natural-language question."""
    _setup_logging(verbose)
    from src.audio_rag.config import Config
    from src.audio_rag.pipeline import AudioRAGPipeline

    cfg = Config()
    pipeline = AudioRAGPipeline(cfg)

    if input_dir:
        answer = pipeline.ingest_and_query(input_dir, question)
    else:
        answer = pipeline.query(question)

    typer.echo(f"\nRisposta:\n{answer.summary}")
    if answer.references:
        typer.echo(f"\nReference:\n{answer.format_references()}")

    if output_file:
        data = {
            "query": question,
            "summary": answer.summary,
            "references": [
                {
                    "file_name": r.file_name,
                    "chunk_indices": r.chunk_indices,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                }
                for r in answer.references
            ],
        }
        output_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"\nAnswer saved to '{output_file}'.")


@app.command()
def judge(
    output_json: Path = typer.Option(
        ..., "--output", help="Path to a JSON file with a saved query answer."
    ),
    ground_truth: str = typer.Option(
        ..., "--ground-truth", help="Expected reference answer text."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Evaluate a saved answer against a ground truth using LLM-as-a-Judge."""
    _setup_logging(verbose)
    from src.audio_rag.config import Config
    from src.audio_rag.generator import GeneratedAnswer, Reference
    from src.audio_rag.judge import LLMJudge

    if not output_json.exists():
        typer.echo(f"File not found: {output_json}", err=True)
        raise typer.Exit(code=1)

    data = json.loads(output_json.read_text(encoding="utf-8"))
    refs = [
        Reference(
            file_name=r["file_name"],
            chunk_indices=r["chunk_indices"],
            start_time=r.get("start_time"),
            end_time=r.get("end_time"),
        )
        for r in data.get("references", [])
    ]
    answer = GeneratedAnswer(summary=data["summary"], references=refs)
    cfg = Config()
    result = LLMJudge(cfg).evaluate(data["query"], answer, ground_truth)

    typer.echo(f"\nReference score:    {result.reference_score:.1f}/10")
    typer.echo(f"Completeness score: {result.completeness_score:.1f}/10")
    typer.echo(f"Hallucination:      {'YES' if result.hallucination_detected else 'NO'}")
    typer.echo(f"\nFeedback:\n{result.feedback}")


@app.command("generate-dataset")
def generate_dataset(
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Regenerate files that already exist."
    ),
    lang: Optional[str] = typer.Option(
        None, "--lang", help="Generate only scripts for this language ('it' or 'en')."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Generate the synthetic audio dataset using Kokoro TTS."""
    _setup_logging(verbose)
    from src.audio_rag.config import Config
    from src.audio_rag.dataset_generator.dataset_builder import DatasetBuilder
    from src.audio_rag.dataset_generator.script_builder import ScriptBuilder
    from src.audio_rag.dataset_generator.tts_engine import TTSEngine

    cfg = Config()
    sb = ScriptBuilder()
    db = DatasetBuilder(
        tts_engine=TTSEngine(cfg),
        script_builder=sb,
        config=cfg,
    )

    if lang:
        ids = [s.script_id for s in sb.get_scripts_by_language(lang)]
        if not ids:
            typer.echo(f"No scripts found for language '{lang}'.", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Generating {len(ids)} script(s) for language '{lang}'...")
        manifest = db.build_subset(ids)
    else:
        typer.echo("Generating full dataset (25 audio files)...")
        manifest = db.build(overwrite=overwrite)

    typer.echo(f"Done. {manifest.total_audio} audio file(s) saved to '{cfg.SYNTHETIC_AUDIO_DIR}'.")
    typer.echo(f"Manifest: {cfg.MANIFEST_PATH}")


if __name__ == "__main__":
    app()
