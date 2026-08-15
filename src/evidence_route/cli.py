"""The ``evidence-route`` command line interface (spec section 19).

The command surface is fixed now, ahead of the implementations, because it
defines the reproduction path a reviewer will follow:

    evidence-route data prepare    --config configs/datasets/financebench.yaml
    evidence-route index build     --experiment configs/experiments/mvp.yaml
    evidence-route outcomes run    --experiment configs/experiments/mvp.yaml
    evidence-route router train    --config configs/routers/linucb.yaml
    evidence-route policy evaluate --experiment configs/experiments/final_test.yaml
    evidence-route bandit simulate --config configs/experiments/ope_simulation.yaml
    evidence-route ope evaluate    --config configs/experiments/ope_study.yaml
    evidence-route report build    --experiment-id final-v1

Subcommands that are not implemented yet exit with a clear message naming the
roadmap week that delivers them, rather than a traceback. ``env check`` and
``actions list`` work today and are what the setup instructions rely on.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import typer

from evidence_route import __version__

app = typer.Typer(
    name="evidence-route",
    help="Statistical evaluation and contextual-bandit routing for enterprise QA.",
    no_args_is_help=True,
    add_completion=False,
)

data_app = typer.Typer(
    help="Download, validate and split benchmark datasets.", no_args_is_help=True
)
index_app = typer.Typer(help="Build lexical and dense retrieval indexes.", no_args_is_help=True)
outcomes_app = typer.Typer(
    help="Execute workflows to build the outcome matrix.", no_args_is_help=True
)
router_app = typer.Typer(help="Train and inspect routing policies.", no_args_is_help=True)
policy_app = typer.Typer(help="Evaluate fixed and learned policies.", no_args_is_help=True)
bandit_app = typer.Typer(help="Simulate logged bandit feedback.", no_args_is_help=True)
ope_app = typer.Typer(help="Compare offline policy estimators.", no_args_is_help=True)
report_app = typer.Typer(help="Build tables, figures and the static report.", no_args_is_help=True)
env_app = typer.Typer(help="Inspect the local environment.", no_args_is_help=True)
actions_app = typer.Typer(help="Inspect the routing action space.", no_args_is_help=True)

for sub, name in (
    (data_app, "data"),
    (index_app, "index"),
    (outcomes_app, "outcomes"),
    (router_app, "router"),
    (policy_app, "policy"),
    (bandit_app, "bandit"),
    (ope_app, "ope"),
    (report_app, "report"),
    (env_app, "env"),
    (actions_app, "actions"),
):
    app.add_typer(sub, name=name)


def _not_yet(command: str, week: str) -> NoReturn:
    """Exit cleanly for a command whose implementation is still scheduled."""
    typer.secho(f"`{command}` is not implemented yet.", fg=typer.colors.YELLOW)
    typer.echo(f"Scheduled for {week} of the roadmap (see section 21 of the specification).")
    raise typer.Exit(code=2)


# invoke_without_command lets `--version` work on its own; without it Typer
# treats a bare flag as a missing subcommand and exits 2.
@app.callback(invoke_without_command=True)
def _main(
    version: bool = typer.Option(
        False, "--version", is_eager=True, help="Show the package version and exit."
    ),
) -> None:
    if version:
        typer.echo(f"evidence-route {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# env — works today
# ---------------------------------------------------------------------------
@env_app.command("check")
def env_check() -> None:
    """Verify that the environment is wired up correctly.

    Reports what is configured without printing any secret value.
    """
    from evidence_route.config import PROJECT_ROOT, get_settings

    settings = get_settings()

    typer.secho(f"evidence-route {__version__}", bold=True)
    typer.echo(f"  project root : {PROJECT_ROOT}")
    typer.echo(f"  offline mode : {settings.offline}")
    typer.echo(f"  random seed  : {settings.random_seed}")
    typer.echo(f"  budget ceiling: ${settings.max_experiment_budget_usd:.2f}")
    typer.echo(f"  test split protected: {settings.protect_test_split}")

    # Evallab is an editable local dependency during development, so its
    # absence is a normal state worth reporting clearly rather than a crash.
    try:
        import agent_eval

        typer.secho(
            f"  evallab      : available (agent-eval {agent_eval.__version__})",
            fg=typer.colors.GREEN,
        )
    except ImportError:
        typer.secho(
            "  evallab      : NOT INSTALLED — run `pip install -e ../evallab`",
            fg=typer.colors.YELLOW,
        )

    if settings.azure.is_configured():
        typer.secho("  azure openai : configured", fg=typer.colors.GREEN)
    else:
        missing = ", ".join(settings.azure.missing_fields())
        typer.secho(f"  azure openai : not configured (missing: {missing})", fg=typer.colors.YELLOW)
        typer.echo("                 copy .env.example to .env and fill it in")

    env_file = PROJECT_ROOT / ".env"
    typer.echo(f"  .env present : {env_file.exists()}")


@actions_app.command("list")
def actions_list() -> None:
    """List the routing action space."""
    from evidence_route.workflows.actions import ACTION_SPACE

    typer.secho("Action space:", bold=True)
    for i, action in enumerate(ACTION_SPACE):
        flags = []
        if action.uses_retrieval:
            flags.append("retrieval")
        if not action.produces_answer:
            flags.append("no-answer")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        typer.echo(f"  {i}  {action.value}{suffix}")


@app.command("profiles")
def profiles() -> None:
    """List the declared reward profiles and their weights."""
    from evidence_route.evaluation.rewards import REWARD_PROFILES

    for name, profile in REWARD_PROFILES.items():
        typer.secho(f"{name} ({profile.version})", bold=True)
        typer.echo(f"  {profile.description}")
        typer.echo(
            f"  lambda: cost={profile.lambda_cost} latency={profile.lambda_latency} "
            f"hallucination={profile.lambda_hallucination} violation={profile.lambda_violation}"
        )
        typer.echo("")


# ---------------------------------------------------------------------------
# Scheduled commands
# ---------------------------------------------------------------------------
@data_app.command("prepare")
def data_prepare(
    config: Path = typer.Option(
        ..., "--config", help="Dataset config, e.g. configs/datasets/financebench.yaml"
    ),
) -> None:
    """Download, validate, checksum and split a benchmark dataset."""
    _not_yet("data prepare", "week 2")


@index_app.command("build")
def index_build(
    experiment: Path = typer.Option(..., "--experiment", help="Experiment config path."),
) -> None:
    """Build the BM25 and dense indexes for an experiment."""
    _not_yet("index build", "weeks 3-4")


@outcomes_app.command("run")
def outcomes_run(
    experiment: Path = typer.Option(..., "--experiment", help="Experiment config path."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Estimate cost and token use without calling the provider."
    ),
) -> None:
    """Execute every candidate action per question to build the outcome matrix."""
    _not_yet("outcomes run", "week 7")


@router_app.command("train")
def router_train(
    config: Path = typer.Option(..., "--config", help="Router config path."),
) -> None:
    """Train a routing policy on development/validation outcomes."""
    _not_yet("router train", "weeks 8-9")


@policy_app.command("evaluate")
def policy_evaluate(
    experiment: Path = typer.Option(..., "--experiment", help="Experiment config path."),
) -> None:
    """Evaluate fixed and learned policies on held-out data."""
    _not_yet("policy evaluate", "week 11")


@bandit_app.command("simulate")
def bandit_simulate(
    config: Path = typer.Option(..., "--config", help="Simulation config path."),
) -> None:
    """Replay behavior policies over the outcome matrix to produce logged feedback."""
    _not_yet("bandit simulate", "week 10")


@ope_app.command("evaluate")
def ope_evaluate(
    config: Path = typer.Option(..., "--config", help="OPE study config path."),
) -> None:
    """Compare offline policy estimators against known full-information values."""
    _not_yet("ope evaluate", "week 10")


@report_app.command("build")
def report_build(
    experiment_id: str = typer.Option(..., "--experiment-id", help="Experiment identifier."),
) -> None:
    """Build reproducible tables, figures and the static HTML report."""
    _not_yet("report build", "week 12")


if __name__ == "__main__":  # pragma: no cover
    app()
