"""Dataset manifest: serialization, deserialization, and ground-truth query generation."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    script_id: str
    filename: str
    file_path: str
    language: str
    gender: str
    voice: str
    topic: str
    duration_seconds: float
    expected_keywords: list[str]


@dataclass
class DatasetManifest:
    version: str
    created_at: str
    total_audio: int
    entries: list[ManifestEntry]


@dataclass
class GroundTruthQuery:
    query: str
    expected_source_files: list[str]
    topic: str


class ManifestManager:
    """Serialise / deserialise the dataset manifest and derive ground-truth queries."""

    VERSION = "1.0"

    def save(self, manifest: DatasetManifest, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": manifest.version,
            "created_at": manifest.created_at,
            "total_audio": manifest.total_audio,
            "entries": [asdict(e) for e in manifest.entries],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Manifest saved: %s (%d entries)", path, manifest.total_audio)

    def load(self, path: Path) -> DatasetManifest:
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Manifest at '{path}' is corrupted or not valid JSON."
            ) from exc
        try:
            entries = [ManifestEntry(**e) for e in raw.get("entries", [])]
            return DatasetManifest(
                version=raw["version"],
                created_at=raw["created_at"],
                total_audio=raw["total_audio"],
                entries=entries,
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Manifest at '{path}' has an unexpected structure: {exc}"
            ) from exc

    def get_ground_truth_queries(
        self, manifest: DatasetManifest
    ) -> list[GroundTruthQuery]:
        """Generate one query per topic using the entries' expected_keywords."""
        by_topic: dict[str, list[ManifestEntry]] = {}
        for entry in manifest.entries:
            by_topic.setdefault(entry.topic, []).append(entry)

        queries: list[GroundTruthQuery] = []
        for topic, entries in by_topic.items():
            filenames = [e.filename for e in entries]
            keywords = entries[0].expected_keywords
            keyword = keywords[0] if keywords else topic
            if entries[0].language == "it":
                query_text = f"Dove si parla di {keyword}?"
            else:
                query_text = f"Which audio files discuss {keyword}?"
            queries.append(
                GroundTruthQuery(
                    query=query_text,
                    expected_source_files=filenames,
                    topic=topic,
                )
            )
        return queries

    @staticmethod
    def make_empty(created_at: str | None = None) -> DatasetManifest:
        return DatasetManifest(
            version=ManifestManager.VERSION,
            created_at=created_at or datetime.now(tz=timezone.utc).isoformat(),
            total_audio=0,
            entries=[],
        )
