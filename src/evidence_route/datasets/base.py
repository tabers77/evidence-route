"""Shared dataset types and configuration loading.

A :class:`QuestionRecord` requires a split, but a question does not belong to a
split until the split generator has run. Rather than weaken that constraint with
an optional field, loaders emit :class:`RawQuestion` — everything parsed from the
source, minus the split — and :func:`to_question_record` promotes it once the
assignment exists. The invariant "a QuestionRecord always knows its split" is
worth an extra type, because a record whose split is unknown is exactly the kind
of thing that silently ends up in the wrong half of an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from evidence_route.storage.records import QuestionRecord, QuestionType, SplitName

__all__ = [
    "DatasetConfig",
    "DownloadConfig",
    "RawQuestion",
    "SplitConfig",
    "ValidationConfig",
    "load_dataset_config",
    "to_question_record",
]


@dataclass(frozen=True)
class RawQuestion:
    """A question parsed from a source dataset, before splits are assigned.

    ``group_key`` is what the split generator groups on — company for
    FinanceBench, document for FiQA. It is carried on the question itself so the
    grouping decision travels with the data rather than being recomputed from
    metadata at split time, where a silent mismatch would be hard to notice.
    """

    question_id: str
    dataset: str
    question_text: str
    group_key: str
    document_ids: tuple[str, ...] = ()
    reference_answer: str | None = None
    reference_evidence: tuple[str, ...] = ()
    question_type: QuestionType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def to_question_record(raw: RawQuestion, split: SplitName) -> QuestionRecord:
    """Promote a :class:`RawQuestion` into a split-assigned record."""
    return QuestionRecord(
        question_id=raw.question_id,
        dataset=raw.dataset,
        split=split,
        question_text=raw.question_text,
        question_type=raw.question_type,
        document_ids=list(raw.document_ids),
        reference_answer=raw.reference_answer,
        reference_evidence=list(raw.reference_evidence),
        metadata={**raw.metadata, "group_key": raw.group_key},
    )


# ---------------------------------------------------------------------------
# Configuration (configs/datasets/*.yaml)
# ---------------------------------------------------------------------------
class DownloadConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    manifest_path: Path
    raw_dir: Path
    verify_checksums: bool = True


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy: str = "grouped"
    group_by: str = "company"
    seed: int
    fractions: dict[str, float]
    output_dir: Path
    #: Once true, regenerating splits is refused unless explicitly overridden.
    #: A test split that can be silently regenerated is not frozen.
    freeze_test_ids: bool = True


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    require_reference_answer: bool = True
    require_reference_evidence: bool = True
    report_unmatched_evidence: bool = True


class DatasetConfig(BaseModel):
    """The parts of a dataset config the pipeline actually consumes."""

    model_config = ConfigDict(extra="ignore")

    name: str
    version: str
    source_url: str | None = None
    license: str | None = None
    download: DownloadConfig
    splits: SplitConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    #: Present on supporting datasets (FiQA, RAGTruth); absent on FinanceBench.
    subset: dict[str, Any] | None = None


def load_dataset_config(path: Path) -> DatasetConfig:
    """Parse a dataset YAML config.

    ``extra="ignore"`` throughout: the YAML files carry documentation keys
    (parsing, chunking, scorer_validation) that later stages consume. This
    loader validates only what it uses, so adding a key for a future stage does
    not break the current one.
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a YAML mapping.")
    return DatasetConfig.model_validate(payload)
