"""Dataset loading, validation, splitting and versioning (spec section 7).

Three corpora, with three distinct jobs:

``financebench``
    Primary business-facing benchmark. Public financial documents with
    associated evidence, covering extraction, numerical reasoning and evidence
    synthesis. The public sample is small, so it is treated as a high-quality
    held-out evaluation set — not as an unlimited source of bandit training
    observations.

``fiqa`` (BEIR)
    Supporting retrieval set. Provides enough queries to compare retrieval
    approaches and to fit routing features without consuming FinanceBench test
    examples.

``ragtruth``
    Scorer-validation set only. Used to check whether the hallucination and
    evidence-support scorers actually detect annotated unsupported spans, rather
    than trusting self-generated evaluation examples. It does not become a third
    application branch.

Splitting is grouped by document or company rather than sampled per question:
several questions often share a source, and question-level random splitting
would leak document-specific patterns from training into test (spec 7.4).

This repository stores manifests, checksums and split definitions — not the
source documents, whose licenses generally do not permit redistribution.
"""

from evidence_route.datasets.base import (
    DatasetConfig,
    RawQuestion,
    load_dataset_config,
    to_question_record,
)
from evidence_route.datasets.download import (
    DATASET_SOURCES,
    DatasetSource,
    DownloadResult,
    download_file,
    ensure_dataset,
)
from evidence_route.datasets.financebench import load_financebench
from evidence_route.datasets.manifest import DatasetManifest, build_manifest
from evidence_route.datasets.prepare import PrepareResult, prepare_dataset
from evidence_route.datasets.splits import (
    SplitAssignment,
    assert_no_group_leakage,
    generate_grouped_splits,
    load_frozen_splits,
    write_splits,
)
from evidence_route.datasets.validation import (
    ValidationIssue,
    ValidationReport,
    validate_questions,
)

__all__ = [
    "DATASET_SOURCES",
    "DatasetConfig",
    "DatasetManifest",
    "DatasetSource",
    "DownloadResult",
    "PrepareResult",
    "download_file",
    "ensure_dataset",
    "RawQuestion",
    "SplitAssignment",
    "ValidationIssue",
    "ValidationReport",
    "assert_no_group_leakage",
    "build_manifest",
    "generate_grouped_splits",
    "load_dataset_config",
    "load_financebench",
    "load_frozen_splits",
    "prepare_dataset",
    "to_question_record",
    "validate_questions",
    "write_splits",
]
