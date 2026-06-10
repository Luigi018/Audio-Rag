"""Standalone CLI for generating the synthetic audio dataset."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

app = typer.Typer(
    name="generate-dataset",
    help="Generate the synthetic audio dataset using Kokoro TTS.",
    add_completion=False,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _make_builder(cfg):
    from src.audio_rag.dataset_generator.dataset_builder import DatasetBuilder
    from src.audio_rag.dataset_generator.script_builder import ScriptBuilder
    from src.audio_rag.dataset_generator.tts_engine import TTSEngine

    return DatasetBuilder(
        tts_engine=TTSEngine(cfg),
        script_builder=ScriptBuilder(),
        config=cfg,
    )


@app.command()
def main(
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Regenerate files that already exist."
    ),
    lang: Optional[str] = typer.Option(
        None, "--lang", help="Generate only scripts for this language ('it' or 'en')."
    ),
    ids: Optional[list[str]] = typer.Option(
        None, "--ids", help="Generate only these script IDs (space-separated)."
    ),
    show_manifest: bool = typer.Option(
        False, "--show-manifest", help="Print the existing manifest and exit."
    ),
    export_queries: Optional[Path] = typer.Option(
        None, "--export-queries", help="Export ground-truth queries to this JSON file."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Generate the synthetic audio dataset using Kokoro TTS."""
    _setup_logging(verbose)

    from src.audio_rag.config import Config
    from src.audio_rag.dataset_generator.manifest import ManifestManager
    from src.audio_rag.dataset_generator.script_builder import ScriptBuilder

    cfg = Config()
    mgr = ManifestManager()

    if show_manifest:
        if not cfg.MANIFEST_PATH.exists():
            typer.echo("No manifest found. Run without --show-manifest to generate the dataset.")
            raise typer.Exit(code=1)
        manifest = mgr.load(cfg.MANIFEST_PATH)
        typer.echo(f"Version: {manifest.version}")
        typer.echo(f"Created: {manifest.created_at}")
        typer.echo(f"Total audio: {manifest.total_audio}")
        for e in manifest.entries:
            typer.echo(
                f"  [{e.language}/{e.gender}] {e.filename}  topic={e.topic}  "
                f"duration={e.duration_seconds:.1f}s"
            )
        raise typer.Exit()

    if export_queries:
        if not cfg.MANIFEST_PATH.exists():
            typer.echo("No manifest found. Generate the dataset first.")
            raise typer.Exit(code=1)
        manifest = mgr.load(cfg.MANIFEST_PATH)
        queries = mgr.get_ground_truth_queries(manifest)
        export_queries.write_text(
            json.dumps([asdict(q) for q in queries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        typer.echo(f"Exported {len(queries)} ground-truth queries to '{export_queries}'.")
        raise typer.Exit()

    builder = _make_builder(cfg)

    if ids:
        typer.echo(f"Generating subset: {ids}")
        manifest = builder.build_subset(list(ids))
    elif lang:
        sb = ScriptBuilder()
        subset_ids = [s.script_id for s in sb.get_scripts_by_language(lang)]
        if not subset_ids:
            typer.echo(f"No scripts found for language '{lang}'.")
            raise typer.Exit(code=1)
        typer.echo(f"Generating {len(subset_ids)} scripts for language '{lang}'.")
        manifest = builder.build_subset(subset_ids)
    else:
        typer.echo("Generating full dataset (25 audio files)...")
        manifest = builder.build(overwrite=overwrite)

    typer.echo(f"\nDone. {manifest.total_audio} audio file(s) generated.")
    typer.echo(f"Manifest: {cfg.MANIFEST_PATH}")
    typer.echo(f"Audio:    {cfg.SYNTHETIC_AUDIO_DIR}")


if __name__ == "__main__":
    app()
