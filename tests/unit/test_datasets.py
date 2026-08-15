"""Tests for the FinanceBench loader, validation and the prepare pipeline."""

from __future__ import annotations

import json

import pytest

from evidence_route.datasets.base import RawQuestion, to_question_record
from evidence_route.datasets.financebench import (
    evidence_reference,
    load_financebench,
    parse_financebench_row,
)
from evidence_route.datasets.manifest import DatasetManifest, build_manifest
from evidence_route.datasets.prepare import prepare_dataset
from evidence_route.datasets.validation import validate_questions


@pytest.fixture
def sample_path(fixtures_dir):
    return fixtures_dir / "questions" / "financebench_sample.jsonl"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def test_loads_every_fixture_row(sample_path):
    questions, failures = load_financebench(sample_path)
    assert len(questions) == 10
    assert failures == []


def test_maps_fields_onto_the_raw_question(sample_path):
    questions, _ = load_financebench(sample_path)
    first = next(q for q in questions if q.question_id == "financebench_id_00001")

    assert first.dataset == "financebench"
    assert first.question_text.startswith("What was 3M's capital expenditure")
    assert first.group_key == "3M"
    assert first.document_ids == ("3M_2018_10K",)
    assert first.reference_answer == "$1,577.00"
    assert first.reference_evidence == ("3M_2018_10K::p60",)


def test_question_type_is_left_unset(sample_path):
    """FinanceBench's label describes how a question was produced, not what
    answering it requires. Mapping it onto the routing taxonomy would fabricate
    a signal the router then learns from."""
    questions, _ = load_financebench(sample_path)
    assert all(q.question_type is None for q in questions)
    # The raw label is preserved rather than discarded.
    assert questions[0].metadata["financebench_question_type"] == "domain-relevant"


def test_multi_evidence_questions_keep_every_reference(sample_path):
    questions, _ = load_financebench(sample_path)
    multi = next(q for q in questions if q.question_id == "financebench_id_00002")
    assert multi.reference_evidence == ("3M_2018_10K::p24", "3M_2018_10K::p25")


def test_grouping_is_by_company_not_document(sample_path):
    questions, _ = load_financebench(sample_path)
    groups = {q.group_key for q in questions}
    assert "3M" in groups
    assert "3M_2018_10K" not in groups


def test_evidence_reference_is_stable():
    assert evidence_reference("DOC", 12) == "DOC::p12"
    assert evidence_reference("DOC", None) == "DOC::p?"
    assert evidence_reference("DOC", 12) == evidence_reference("DOC", 12)


def test_duplicate_evidence_pages_are_deduped():
    row = {
        "financebench_id": "x1",
        "company": "ACME",
        "doc_name": "ACME_10K",
        "question": "q?",
        "answer": "a",
        "evidence": [
            {"doc_name": "ACME_10K", "evidence_page_num": 5},
            {"doc_name": "ACME_10K", "evidence_page_num": 5},
            {"doc_name": "ACME_10K", "evidence_page_num": 6},
        ],
    }
    parsed = parse_financebench_row(row)
    assert parsed.reference_evidence == ("ACME_10K::p5", "ACME_10K::p6")


def test_row_without_an_id_is_rejected():
    with pytest.raises(ValueError, match="no financebench_id"):
        parse_financebench_row({"company": "ACME", "question": "q?"})


def test_malformed_rows_are_reported_not_silently_skipped(tmp_path):
    """A loader that quietly drops what it cannot parse produces a benchmark
    that looks cleaner than it is."""
    path = tmp_path / "broken.jsonl"
    path.write_text(
        '{"financebench_id": "ok1", "company": "A", "doc_name": "D", '
        '"question": "q?", "answer": "a", "evidence": []}\n'
        "{not valid json\n"
        '{"company": "B", "question": "no id"}\n',
        encoding="utf-8",
    )
    questions, failures = load_financebench(path)
    assert len(questions) == 1
    assert len(failures) == 2
    assert any("invalid JSON" in f for f in failures)


def test_missing_file_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data prepare"):
        load_financebench(tmp_path / "absent.jsonl")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _raw(**overrides) -> RawQuestion:
    kwargs = {
        "question_id": "q1",
        "dataset": "financebench",
        "question_text": "What was revenue?",
        "group_key": "ACME",
        "document_ids": ("ACME_10K",),
        "reference_answer": "$1",
        "reference_evidence": ("ACME_10K::p1",),
    }
    kwargs.update(overrides)
    return RawQuestion(**kwargs)


def test_clean_dataset_passes(sample_path):
    questions, _ = load_financebench(sample_path)
    report = validate_questions(questions, dataset="financebench")
    assert report.is_clean
    assert len(report.valid) == 10
    assert report.excluded_count == 0


def test_missing_evidence_is_excluded_and_counted():
    report = validate_questions([_raw(reference_evidence=())], dataset="d")
    assert report.valid == []
    assert report.counts_by_code() == {"missing_reference_evidence": 1}
    assert report.excluded_count == 1


def test_evidence_requirement_can_be_relaxed():
    """FiQA supplies relevance labels rather than answers."""
    report = validate_questions(
        [_raw(reference_answer=None)],
        dataset="fiqa",
        require_reference_answer=False,
    )
    assert len(report.valid) == 1


def test_missing_group_key_is_an_error():
    """Without a grouping key the question cannot be split without leaking."""
    report = validate_questions([_raw(group_key="  ")], dataset="d")
    assert "missing_group_key" in report.counts_by_code()


def test_duplicate_ids_are_flagged():
    report = validate_questions([_raw(), _raw()], dataset="d")
    assert report.counts_by_code()["duplicate_question_id"] == 2


def test_one_question_failing_several_checks_counts_once_as_excluded():
    report = validate_questions(
        [_raw(reference_answer=None, reference_evidence=(), document_ids=())],
        dataset="d",
    )
    assert report.excluded_count == 1
    assert len(report.issues) == 3


def test_summary_is_human_readable():
    report = validate_questions(
        [_raw(), _raw(question_id="q2", reference_evidence=())], dataset="d"
    )
    summary = report.summary()
    assert "1 valid / 2 parsed" in summary
    assert "missing_reference_evidence" in summary


# ---------------------------------------------------------------------------
# RawQuestion -> QuestionRecord
# ---------------------------------------------------------------------------
def test_promotion_attaches_the_split_and_preserves_the_group():
    record = to_question_record(_raw(), "dev")
    assert record.split == "dev"
    assert record.question_id == "q1"
    assert record.metadata["group_key"] == "ACME"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def test_manifest_checksums_existing_files_and_round_trips(tmp_path, sample_path):
    manifest = build_manifest(
        name="financebench",
        version="public-sample-v1",
        source_url="https://example.invalid",
        license_name="CC-BY-NC-4.0",
        source_files={"source": sample_path, "absent": tmp_path / "nope.jsonl"},
        transformations=["loaded", "validated"],
        split_seed=20260815,
        split_group_by="company",
        split_counts={"dev": 3, "validation": 3, "test": 4},
        total_parsed=10,
        total_valid=10,
        excluded_count=0,
    )
    assert manifest.file_checksums["source"].startswith("sha256:")
    assert "absent" not in manifest.file_checksums

    path = manifest.write(tmp_path / "m" / "manifest.json")
    assert DatasetManifest.read(path).split_seed == 20260815


def test_manifest_excluded_count_is_not_derived_from_reason_sums(tmp_path, sample_path):
    """One question can fail several checks; summing reasons would overstate it."""
    manifest = build_manifest(
        name="d",
        version="v1",
        source_url=None,
        license_name=None,
        source_files={"source": sample_path},
        transformations=[],
        excluded_count=1,
        exclusion_reasons={"a": 1, "b": 1, "c": 1},
    )
    assert manifest.excluded_count == 1


# ---------------------------------------------------------------------------
# End-to-end prepare
# ---------------------------------------------------------------------------
@pytest.fixture
def prepare_config(tmp_path, sample_path):
    """A dataset config pointing entirely inside tmp_path."""
    import yaml

    config = {
        "name": "financebench",
        "version": "test-v1",
        "source_url": "https://example.invalid",
        "license": "CC-BY-NC-4.0",
        "download": {
            "manifest_path": str(tmp_path / "manifests" / "financebench.json"),
            "raw_dir": str(sample_path.parent),
            "verify_checksums": True,
        },
        "splits": {
            "strategy": "grouped",
            "group_by": "company",
            "seed": 20260815,
            "fractions": {"dev": 0.3, "validation": 0.25, "test": 0.45},
            "output_dir": str(tmp_path / "splits"),
            "freeze_test_ids": True,
        },
        "validation": {
            "require_reference_answer": True,
            "require_reference_evidence": True,
        },
        # An unrelated key a later stage consumes; the loader must ignore it.
        "chunking": {"strategy": "recursive", "chunk_size_tokens": 512},
    }
    path = tmp_path / "financebench.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_prepare_runs_end_to_end(prepare_config, sample_path, tmp_path):
    result = prepare_dataset(prepare_config, source_path=sample_path)

    assert len(result.records) == 10
    assert result.reused_frozen_splits is False
    assert result.splits_path.exists()
    assert result.manifest_path.exists()
    assert sum(result.assignment.counts().values()) == 10
    assert all(r.split in {"dev", "validation", "test"} for r in result.records)


def test_prepare_is_reproducible(prepare_config, sample_path):
    first = prepare_dataset(prepare_config, source_path=sample_path)
    second = prepare_dataset(prepare_config, source_path=sample_path)
    assert second.reused_frozen_splits is True
    assert first.assignment.split_of_question == second.assignment.split_of_question


def test_prepare_reuses_frozen_splits_rather_than_regenerating(prepare_config, sample_path):
    """Splits that regenerate silently are not frozen."""
    prepare_dataset(prepare_config, source_path=sample_path)
    again = prepare_dataset(prepare_config, source_path=sample_path)
    assert again.reused_frozen_splits is True
    assert "reused frozen splits" in again.manifest.transformations


def test_force_resplit_regenerates(prepare_config, sample_path):
    prepare_dataset(prepare_config, source_path=sample_path)
    forced = prepare_dataset(prepare_config, source_path=sample_path, force_resplit=True)
    assert forced.reused_frozen_splits is False


def test_changed_source_against_frozen_splits_fails_loudly(prepare_config, sample_path, tmp_path):
    """A frozen split file plus a changed dataset is silent corruption."""
    prepare_dataset(prepare_config, source_path=sample_path)

    trimmed = tmp_path / "trimmed.jsonl"
    rows = sample_path.read_text(encoding="utf-8").strip().splitlines()
    trimmed.write_text("\n".join(rows[:-2]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no longer match the source data"):
        prepare_dataset(prepare_config, source_path=trimmed)


def test_prepare_writes_a_manifest_with_provenance(prepare_config, sample_path):
    result = prepare_dataset(prepare_config, source_path=sample_path)
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert payload["name"] == "financebench"
    assert payload["license"] == "CC-BY-NC-4.0"
    assert payload["split_seed"] == 20260815
    assert payload["split_group_by"] == "company"
    assert payload["total_valid"] == 10
    assert payload["file_checksums"]["source"].startswith("sha256:")


def test_unknown_dataset_reports_clearly(tmp_path, prepare_config):
    import yaml

    payload = yaml.safe_load(prepare_config.read_text(encoding="utf-8"))
    payload["name"] = "not_a_dataset"
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(NotImplementedError, match="No loader implemented"):
        prepare_dataset(path)
