"""The `data prepare` pipeline: load, validate, split, freeze, record.

One entry point so that dataset preparation is reproducible from a command
rather than from a sequence of notebook cells (spec section 21, week 2 exit
criteria).

The frozen-split guard is the important behaviour here. Once
``splits.json`` exists and the config declares ``freeze_test_ids: true``,
regeneration is refused unless explicitly forced. Splits that can be
regenerated silently are not frozen, and a benchmark whose test set quietly
changed between runs cannot support any of the comparisons this project makes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evidence_route.config import PROJECT_ROOT
from evidence_route.datasets.base import (
    DatasetConfig,
    RawQuestion,
    load_dataset_config,
    to_question_record,
)
from evidence_route.datasets.financebench import load_financebench
from evidence_route.datasets.manifest import DatasetManifest, build_manifest
from evidence_route.datasets.splits import (
    SplitAssignment,
    assert_no_group_leakage,
    generate_grouped_splits,
    load_frozen_splits,
    write_splits,
)
from evidence_route.datasets.validation import ValidationReport, validate_questions
from evidence_route.storage.records import QuestionRecord

__all__ = ["PrepareResult", "prepare_dataset", "resolve_source_path"]

#: Loaders keyed by dataset name. FiQA and RAGTruth land here as they are built.
_LOADERS = {
    "financebench": load_financebench,
}


@dataclass
class PrepareResult:
    """Everything `data prepare` produced, for the CLI to report."""

    config: DatasetConfig
    report: ValidationReport
    assignment: SplitAssignment
    records: list[QuestionRecord]
    manifest: DatasetManifest
    splits_path: Path
    manifest_path: Path
    parse_failures: list[str]
    reused_frozen_splits: bool
    #: Set when this run fetched the source file rather than finding it present.
    downloaded: bool = False


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


#: Accepted source filenames per dataset, in preference order.
#:
#: FinanceBench has been distributed under more than one name: the original
#: GitHub repository (now removed) shipped ``financebench_open_source.jsonl``,
#: while the HuggingFace dataset that replaced it ships
#: ``financebench_merged.jsonl``. Both are accepted so a checkout obtained from
#: either source works without editing config.
_SOURCE_FILENAMES: dict[str, tuple[str, ...]] = {
    "financebench": (
        "financebench_merged.jsonl",
        "financebench_open_source.jsonl",
    ),
    "fiqa": ("fiqa.jsonl",),
    "ragtruth": ("ragtruth.jsonl",),
}


def resolve_source_path(config: DatasetConfig) -> Path:
    """Locate the raw source file for a dataset.

    Returns the first accepted filename that exists; if none do, returns the
    preferred one so the resulting error names the file the user should fetch.
    Kept separate so tests can point at a fixture without inventing a config.
    """
    raw_dir = _absolute(config.download.raw_dir)
    candidates = _SOURCE_FILENAMES.get(config.name, (f"{config.name}.jsonl",))
    for filename in candidates:
        path = raw_dir / filename
        if path.exists():
            return path
    return raw_dir / candidates[0]


def prepare_dataset(
    config_path: Path,
    *,
    source_path: Path | None = None,
    force_resplit: bool = False,
    download: bool = True,
    force_download: bool = False,
) -> PrepareResult:
    """Download, load, validate, split and freeze a dataset from its config.

    Downloading is on by default so that preparation is genuinely one command
    (spec section 19). It is skipped when ``source_path`` is given explicitly,
    since the caller has then already said where the data is.
    """
    config = load_dataset_config(config_path)

    loader = _LOADERS.get(config.name)
    if loader is None:
        raise NotImplementedError(
            f"No loader implemented for dataset {config.name!r}. "
            f"Available: {', '.join(sorted(_LOADERS))}."
        )

    downloaded = False
    if source_path is not None:
        source = source_path
    else:
        source = resolve_source_path(config)
        if download and (force_download or not source.exists()):
            from evidence_route.config import get_settings
            from evidence_route.datasets.download import ensure_dataset

            result = ensure_dataset(
                config.name,
                _absolute(config.download.raw_dir),
                force=force_download,
                offline=get_settings().offline,
            )
            source = result.path
            downloaded = result.downloaded

    questions, parse_failures = loader(source)

    report = validate_questions(
        questions,
        dataset=config.name,
        require_reference_answer=config.validation.require_reference_answer,
        require_reference_evidence=config.validation.require_reference_evidence,
    )
    if not report.valid:
        raise ValueError(
            f"No valid questions after validation for {config.name}.\n{report.summary()}"
        )

    output_dir = _absolute(config.splits.output_dir)
    splits_path = output_dir / "splits.json"

    reused = False
    if splits_path.exists() and config.splits.freeze_test_ids and not force_resplit:
        # Reuse rather than regenerate. This is what makes the test split frozen
        # in practice and not just in intent.
        assignment = load_frozen_splits(splits_path)
        assignment = _reconcile_frozen(assignment, report.valid, splits_path)
        reused = True
    else:
        assignment = generate_grouped_splits(
            report.valid,
            fractions=config.splits.fractions,
            seed=config.splits.seed,
            group_by=config.splits.group_by,
        )
        assert_no_group_leakage(report.valid, assignment)
        splits_path = write_splits(
            assignment,
            output_dir,
            dataset=config.name,
            dataset_version=config.version,
        )

    records = [
        to_question_record(q, assignment.split_of_question[q.question_id]) for q in report.valid
    ]

    manifest = build_manifest(
        name=config.name,
        version=config.version,
        source_url=config.source_url,
        license_name=config.license,
        source_files={"source": source, "splits": splits_path},
        transformations=[
            (
                f"downloaded {source.name}"
                if downloaded
                else f"used existing {source.name}"
            ),
            f"loaded {config.name} from {source.name}",
            "validated against declared requirements",
            (
                "reused frozen splits"
                if reused
                else f"generated grouped splits (group_by={config.splits.group_by}, "
                f"seed={config.splits.seed})"
            ),
        ],
        split_seed=assignment.seed,
        split_group_by=assignment.group_by,
        split_counts=assignment.counts(),
        split_group_counts=assignment.group_counts(),
        total_parsed=report.total,
        total_valid=len(report.valid),
        excluded_count=report.excluded_count,
        exclusion_reasons=report.counts_by_code(),
        parse_failures=parse_failures,
    )
    manifest_path = manifest.write(_absolute(config.download.manifest_path))

    return PrepareResult(
        config=config,
        report=report,
        assignment=assignment,
        records=records,
        manifest=manifest,
        splits_path=splits_path,
        manifest_path=manifest_path,
        parse_failures=parse_failures,
        reused_frozen_splits=reused,
        downloaded=downloaded,
    )


def _reconcile_frozen(
    assignment: SplitAssignment, questions: list[RawQuestion], splits_path: Path
) -> SplitAssignment:
    """Check a frozen assignment still covers exactly the current questions.

    A frozen split file and a changed source dataset is a silent corruption: the
    questions no longer match the assignment, and whichever ones are missing get
    dropped or misfiled without anyone noticing. Fail loudly instead.
    """
    current = {q.question_id for q in questions}
    frozen = set(assignment.split_of_question)

    missing = current - frozen
    stale = frozen - current
    if missing or stale:
        raise ValueError(
            f"Frozen splits at {splits_path} no longer match the source data: "
            f"{len(missing)} question(s) absent from the split file, "
            f"{len(stale)} split entr(ies) with no matching question. "
            f"The dataset changed after the splits were frozen. Investigate before "
            f"continuing — re-splitting invalidates every result that cites the old "
            f"split. Use --force-resplit only if that is genuinely intended."
        )
    return assignment
