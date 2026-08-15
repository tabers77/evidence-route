"""Smoke tests: the package installs and the CLI surface exists.

Marked ``smoke`` because this is the minimal end-to-end path that must run
without paid API access (spec section 19). It is the first thing CI runs and the
first thing to check on a fresh clone.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from evidence_route import __version__
from evidence_route.cli import app

pytestmark = pytest.mark.smoke

runner = CliRunner()


def test_package_imports_and_exposes_a_version():
    assert __version__


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_the_documented_command_groups():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("data", "index", "outcomes", "router", "policy", "bandit", "ope", "report"):
        assert group in result.stdout


def test_env_check_runs_without_credentials():
    """`env check` must be safe to run on a fresh clone with no .env."""
    result = runner.invoke(app, ["env", "check"])
    assert result.exit_code == 0
    assert "evidence-route" in result.stdout


def test_actions_list_shows_the_full_action_space():
    result = runner.invoke(app, ["actions", "list"])
    assert result.exit_code == 0
    for value in ("A0_direct", "A1_bm25", "A5_agentic", "A6_abstain"):
        assert value in result.stdout


def test_profiles_lists_every_reward_profile():
    result = runner.invoke(app, ["profiles"])
    assert result.exit_code == 0
    for name in ("quality_first", "balanced_enterprise", "cost_sensitive", "regulated"):
        assert name in result.stdout


@pytest.mark.parametrize(
    "command",
    [
        ["data", "prepare", "--config", "configs/datasets/financebench.yaml"],
        ["index", "build", "--experiment", "configs/experiments/mvp.yaml"],
        ["outcomes", "run", "--experiment", "configs/experiments/mvp.yaml"],
        ["router", "train", "--config", "configs/routers/linucb.yaml"],
        ["bandit", "simulate", "--config", "configs/experiments/ope_simulation.yaml"],
        ["ope", "evaluate", "--config", "configs/experiments/ope_study.yaml"],
        ["report", "build", "--experiment-id", "final-v1"],
    ],
)
def test_unimplemented_commands_exit_cleanly(command: list[str]):
    """Scheduled commands report their roadmap week instead of raising."""
    result = runner.invoke(app, command)
    assert result.exit_code == 2
    assert "not implemented yet" in result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)
