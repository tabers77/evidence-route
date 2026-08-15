"""Run provenance capture (spec section 25).

Every reported experiment must be traceable to the exact code, environment,
configuration and seed that produced it. This module snapshots that context at
the start of a run and writes it next to the results.

The ``dirty_worktree`` flag matters more than it looks: a result produced from
uncommitted changes cannot be reproduced by anyone else, including the author a
month later. It is recorded rather than blocked, so exploratory runs stay cheap,
but the reporting layer can refuse to publish a claim backed by a dirty run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["RunManifest", "capture_run_manifest", "file_checksum", "git_state"]


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """Run a git command, returning None if git is unavailable or fails."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state(repo_root: Path | None = None) -> dict[str, Any]:
    """Current commit, branch and worktree cleanliness."""
    commit = _git("rev-parse", "HEAD", cwd=repo_root)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root)
    status = _git("status", "--porcelain", cwd=repo_root)
    return {
        "commit": commit,
        "branch": branch,
        # None (git unavailable) is distinct from False (verified clean).
        "dirty_worktree": None if status is None else bool(status.strip()),
    }


def file_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Checksum a file in chunks, so large corpora do not load into memory."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{algorithm}:{digest.hexdigest()}"


@dataclass
class RunManifest:
    """Everything needed to identify and defend a single experiment run."""

    experiment_id: str
    created_at: str

    # Code
    git_commit: str | None = None
    git_branch: str | None = None
    dirty_worktree: bool | None = None

    # Environment
    python_version: str = ""
    platform_info: str = ""
    package_version: str = ""

    # Inputs
    config_path: str | None = None
    config_checksum: str | None = None
    dataset_checksums: dict[str, str] = field(default_factory=dict)

    # Models. Deployment names are recorded; endpoints and keys never are —
    # a manifest is a published artifact.
    model_configuration: dict[str, Any] = field(default_factory=dict)
    prompt_versions: dict[str, str] = field(default_factory=dict)

    # Determinism
    random_seed: int | None = None

    # Outputs
    raw_results_path: str | None = None
    report_version: str | None = None

    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        """Write the manifest as JSON, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @property
    def is_publishable(self) -> bool:
        """Whether a public claim may cite this run.

        Requires a known commit and a clean worktree. Anything else is an
        illustrative example, not a reproduced result (spec section 25).
        """
        return bool(self.git_commit) and self.dirty_worktree is False


def capture_run_manifest(
    experiment_id: str,
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    dataset_paths: dict[str, Path] | None = None,
    model_configuration: dict[str, Any] | None = None,
    prompt_versions: dict[str, str] | None = None,
    random_seed: int | None = None,
    notes: str | None = None,
) -> RunManifest:
    """Snapshot the current run context.

    Call this once at the start of a run, before any expensive work, so the
    recorded state matches the code that actually executed.
    """
    from evidence_route import __version__

    state = git_state(repo_root)
    manifest = RunManifest(
        experiment_id=experiment_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=state["commit"],
        git_branch=state["branch"],
        dirty_worktree=state["dirty_worktree"],
        python_version=sys.version.split()[0],
        platform_info=f"{platform.system()} {platform.release()} ({platform.machine()})",
        package_version=__version__,
        config_path=str(config_path) if config_path else None,
        config_checksum=file_checksum(config_path)
        if config_path and config_path.exists()
        else None,
        dataset_checksums={
            name: file_checksum(p) for name, p in (dataset_paths or {}).items() if p.exists()
        },
        model_configuration=model_configuration or {},
        prompt_versions=prompt_versions or {},
        random_seed=random_seed,
        notes=notes,
    )
    return manifest
