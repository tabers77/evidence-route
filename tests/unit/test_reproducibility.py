"""Tests for run provenance capture."""

from __future__ import annotations

import json

from evidence_route.evaluation.reproducibility import (
    RunManifest,
    capture_run_manifest,
    file_checksum,
    git_state,
)


def test_git_state_reports_the_expected_keys(repo_root):
    state = git_state(repo_root)
    assert set(state) == {"commit", "branch", "dirty_worktree"}


def test_file_checksum_is_stable_and_prefixed(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("evidence", encoding="utf-8")
    first = file_checksum(path)
    assert first.startswith("sha256:")
    assert first == file_checksum(path)

    path.write_text("evidence!", encoding="utf-8")
    assert file_checksum(path) != first


def test_capture_records_environment(repo_root):
    manifest = capture_run_manifest("exp-smoke", repo_root=repo_root, random_seed=42)
    assert manifest.experiment_id == "exp-smoke"
    assert manifest.random_seed == 42
    assert manifest.python_version
    assert manifest.platform_info
    assert manifest.package_version


def test_manifest_writes_valid_json(tmp_path, repo_root):
    manifest = capture_run_manifest("exp-1", repo_root=repo_root, random_seed=1)
    out = manifest.write(tmp_path / "nested" / "manifest.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "exp-1"
    assert payload["random_seed"] == 1


def test_dataset_checksums_are_captured(tmp_path, repo_root):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"question_id": "q1"}\n', encoding="utf-8")
    manifest = capture_run_manifest(
        "exp-1", repo_root=repo_root, dataset_paths={"financebench": dataset}
    )
    assert manifest.dataset_checksums["financebench"].startswith("sha256:")


def test_missing_dataset_paths_are_skipped_not_fatal(tmp_path, repo_root):
    manifest = capture_run_manifest(
        "exp-1", repo_root=repo_root, dataset_paths={"absent": tmp_path / "nope.jsonl"}
    )
    assert manifest.dataset_checksums == {}


def test_dirty_worktree_run_is_not_publishable():
    """A result produced from uncommitted changes cannot be reproduced by anyone."""
    manifest = RunManifest(
        experiment_id="exp-1",
        created_at="2026-08-15T00:00:00Z",
        git_commit="abc123",
        dirty_worktree=True,
    )
    assert not manifest.is_publishable


def test_clean_committed_run_is_publishable():
    manifest = RunManifest(
        experiment_id="exp-1",
        created_at="2026-08-15T00:00:00Z",
        git_commit="abc123",
        dirty_worktree=False,
    )
    assert manifest.is_publishable


def test_unknown_git_state_is_not_publishable():
    """None (git unavailable) is treated as unverified, not as clean."""
    manifest = RunManifest(
        experiment_id="exp-1",
        created_at="2026-08-15T00:00:00Z",
        git_commit=None,
        dirty_worktree=None,
    )
    assert not manifest.is_publishable
