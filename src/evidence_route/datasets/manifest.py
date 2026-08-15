"""Dataset manifests (spec section 7.5).

The repository stores metadata, never the corpora. A manifest is what lets
someone else confirm they are looking at the same data: source, license,
checksums, transformation steps, split seed and the resulting counts.

Manifests are committed. They are also the only durable record that a given
result was produced against a particular version of a dataset, which matters
because the source repositories can and do change.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evidence_route.evaluation.reproducibility import file_checksum

__all__ = ["DatasetManifest", "build_manifest"]


class DatasetManifest(BaseModel):
    """Provenance for one prepared dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    source_url: str | None = None
    license: str | None = None

    prepared_at: str
    #: path (relative to the repo) -> "sha256:..."
    file_checksums: dict[str, str] = Field(default_factory=dict)

    #: Ordered description of what was done to the source data. Free text on
    #: purpose — the point is that a reader can tell whether their copy went
    #: through the same steps.
    transformations: list[str] = Field(default_factory=list)

    split_seed: int | None = None
    split_group_by: str | None = None
    split_counts: dict[str, int] = Field(default_factory=dict)
    split_group_counts: dict[str, int] = Field(default_factory=dict)

    total_parsed: int = 0
    total_valid: int = 0
    excluded_count: int = 0
    exclusion_reasons: dict[str, int] = Field(default_factory=dict)
    parse_failures: list[str] = Field(default_factory=list)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> DatasetManifest:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def build_manifest(
    *,
    name: str,
    version: str,
    source_url: str | None,
    license_name: str | None,
    source_files: dict[str, Path],
    transformations: list[str],
    split_seed: int | None = None,
    split_group_by: str | None = None,
    split_counts: dict[str, int] | None = None,
    split_group_counts: dict[str, int] | None = None,
    total_parsed: int = 0,
    total_valid: int = 0,
    excluded_count: int = 0,
    exclusion_reasons: dict[str, int] | None = None,
    parse_failures: list[str] | None = None,
) -> DatasetManifest:
    """Assemble a manifest, checksumming whichever source files exist.

    ``excluded_count`` is passed in rather than derived from
    ``exclusion_reasons``: one question can fail several checks at once, so
    summing the per-reason counts would overstate how many questions were
    actually dropped.
    """
    checksums = {
        label: file_checksum(path) for label, path in source_files.items() if path.exists()
    }
    return DatasetManifest(
        name=name,
        version=version,
        source_url=source_url,
        license=license_name,
        prepared_at=datetime.now(timezone.utc).isoformat(),
        file_checksums=checksums,
        transformations=transformations,
        split_seed=split_seed,
        split_group_by=split_group_by,
        split_counts=split_counts or {},
        split_group_counts=split_group_counts or {},
        total_parsed=total_parsed,
        total_valid=total_valid,
        excluded_count=excluded_count,
        exclusion_reasons=exclusion_reasons or {},
        parse_failures=parse_failures or [],
    )
