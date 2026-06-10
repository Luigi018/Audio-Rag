"""Orchestrates TTS synthesis to produce the full synthetic audio dataset."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from src.audio_rag.config import Config
from src.audio_rag.dataset_generator.manifest import (
    DatasetManifest,
    ManifestEntry,
    ManifestManager,
)
from src.audio_rag.dataset_generator.script_builder import AudioScript, ScriptBuilder
from src.audio_rag.dataset_generator.tts_engine import TTSEngine

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Iterates over AudioScript definitions, synthesises WAV files, and writes the manifest."""

    def __init__(
        self,
        tts_engine: TTSEngine,
        script_builder: ScriptBuilder,
        config: Config,
    ) -> None:
        self._tts = tts_engine
        self._scripts = script_builder
        self._config = config
        self._manifest_mgr = ManifestManager()

    def build(self, overwrite: bool = False) -> DatasetManifest:
        """Synthesise all 25 scripts and return the saved manifest."""
        return self._synthesise(self._scripts.get_all_scripts(), overwrite=overwrite)

    def build_subset(self, script_ids: list[str]) -> DatasetManifest:
        """Synthesise only the specified script IDs (always overwrites)."""
        by_id = {s.script_id: s for s in self._scripts.get_all_scripts()}
        scripts = [by_id[sid] for sid in script_ids if sid in by_id]
        return self._synthesise(scripts, overwrite=True)

    def _synthesise(
        self, scripts: list[AudioScript], *, overwrite: bool
    ) -> DatasetManifest:
        audio_dir = self._config.SYNTHETIC_AUDIO_DIR
        audio_dir.mkdir(parents=True, exist_ok=True)

        try:
            from tqdm import tqdm  # type: ignore[import]
            iterable = tqdm(scripts, desc="Generating audio", unit="file")
        except ImportError:
            iterable = scripts  # type: ignore[assignment]

        entries: list[ManifestEntry] = []
        for script in iterable:
            out_path = audio_dir / script.filename
            if out_path.exists() and not overwrite:
                logger.info("Skipping existing: %s", out_path.name)
                duration = _read_duration(out_path)
            else:
                try:
                    meta = self._tts.synthesize_and_save(
                        text=script.text,
                        voice=script.voice,
                        lang=script.language,
                        output_path=out_path,
                    )
                    duration = meta.duration_seconds
                    logger.info("Generated: %s (%.1f s)", out_path.name, duration)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to generate '%s': %s", script.filename, exc)
                    continue

            entries.append(
                ManifestEntry(
                    script_id=script.script_id,
                    filename=script.filename,
                    file_path=str(out_path),
                    language=script.language,
                    gender=script.gender,
                    voice=script.voice,
                    topic=script.topic,
                    duration_seconds=duration,
                    expected_keywords=script.expected_keywords,
                )
            )

        manifest = DatasetManifest(
            version=ManifestManager.VERSION,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            total_audio=len(entries),
            entries=entries,
        )
        self._manifest_mgr.save(manifest, self._config.MANIFEST_PATH)
        return manifest


def _read_duration(path: Path) -> float:
    try:
        import soundfile as sf  # type: ignore[import]
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:  # noqa: BLE001
        return 0.0
