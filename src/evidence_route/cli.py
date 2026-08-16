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
corpus_app = typer.Typer(help="Build and fetch document corpora.", no_args_is_help=True)

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
    (corpus_app, "corpus"),
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

    # The generation extra is optional; report it rather than let a run
    # discover it missing partway through an expensive outcome matrix.
    try:
        import openai  # noqa: F401

        typer.secho("  openai sdk   : available", fg=typer.colors.GREEN)
    except ImportError:
        typer.secho(
            '  openai sdk   : NOT INSTALLED — pip install -e ".[generation]"',
            fg=typer.colors.YELLOW,
        )

    if settings.sec_user_agent:
        typer.secho("  sec edgar    : configured", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "  sec edgar    : EVIDENCE_ROUTE_SEC_USER_AGENT unset (needed to fetch filings)",
            fg=typer.colors.YELLOW,
        )

    from evidence_route.generation.cost import PRICING

    if PRICING:
        typer.echo(f"  pricing      : {len(PRICING)} model(s) registered")
    else:
        typer.secho(
            "  pricing      : none registered — cost will be recorded as unknown, not zero",
            fg=typer.colors.YELLOW,
        )

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


@corpus_app.command("build-dev")
def corpus_build_dev(
    source: Path = typer.Option(
        Path("data/raw/financebench/financebench_merged.jsonl"),
        "--source",
        help="FinanceBench metadata file.",
    ),
    chunk_size: int = typer.Option(512, "--chunk-size", help="Chunk size in tokens."),
    overlap: int = typer.Option(64, "--overlap", help="Chunk overlap in tokens."),
) -> None:
    """Build the evidence-page development corpus.

    DEVELOPMENT ONLY. Contains gold evidence pages pooled across questions, so
    retrieval is easier than reality and disproportionately so for BM25. The
    reporting path refuses corpora built this way.
    """
    from evidence_route.documents.chunking import ChunkingConfig
    from evidence_route.documents.evidence_corpus import build_evidence_page_corpus

    try:
        corpus = build_evidence_page_corpus(
            source,
            chunking=ChunkingConfig(chunk_size_tokens=chunk_size, chunk_overlap_tokens=overlap),
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    stats = corpus.stats()
    typer.secho(f"Built {stats['name']}", bold=True)
    typer.echo(f"  documents   : {stats['n_documents']}")
    typer.echo(f"  chunks      : {stats['n_chunks']}")
    typer.echo(f"  tokens      : ~{stats['total_tokens']:,}")
    typer.echo(f"  provenance  : {stats['provenance']}")
    typer.secho(
        f"\n  NOT REPORTABLE. {corpus.provenance.rationale}",
        fg=typer.colors.YELLOW,
    )


@corpus_app.command("fetch")
def corpus_fetch(
    source: Path = typer.Option(
        Path("data/raw/financebench/financebench_merged.jsonl"),
        "--source",
        help="FinanceBench metadata file.",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw/financebench/filings"), "--output-dir", help="Where filings are written."
    ),
    cik_map: Path = typer.Option(
        Path("data/reference/company_cik.json"),
        "--cik-map",
        help="Reviewed company-to-CIK mapping.",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Fetch only the first N documents (use to sanity-check first)."
    ),
) -> None:
    """Fetch the FinanceBench SEC filings from EDGAR.

    Requires EVIDENCE_ROUTE_SEC_USER_AGENT — the SEC mandates a User-Agent with
    contact information and returns 403 without one.
    """
    from evidence_route.config import get_settings
    from evidence_route.datasets.fetch_corpus import fetch_financebench_corpus

    settings = get_settings()
    if settings.offline:
        typer.secho("Offline mode is enabled; refusing to fetch.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not settings.sec_user_agent:
        typer.secho(
            "EVIDENCE_ROUTE_SEC_USER_AGENT is not set.\n"
            "The SEC requires a User-Agent identifying you and including a contact "
            "email, e.g. 'EvidenceRoute research you@example.com'.\n"
            "Add it to .env — see https://www.sec.gov/os/webmaster-faq#developers",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        report = fetch_financebench_corpus(
            source, output_dir, cik_map, user_agent=settings.sec_user_agent, limit=limit
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.secho(report.summary(), bold=True)
    for failure in report.failed[:10]:
        typer.secho(f"  MISSING {failure.document_name}: {failure.reason}", fg=typer.colors.YELLOW)
    if len(report.failed) > 10:
        typer.echo(f"  ... and {len(report.failed) - 10} more")

    path = report.write(output_dir / "coverage.json")
    typer.echo(f"\n  coverage report -> {path}")


# ---------------------------------------------------------------------------
# Scheduled commands
# ---------------------------------------------------------------------------
@data_app.command("prepare")
def data_prepare(
    config: Path = typer.Option(
        ..., "--config", help="Dataset config, e.g. configs/datasets/financebench.yaml"
    ),
    source: Path | None = typer.Option(None, "--source", help="Override the source file location."),
    force_resplit: bool = typer.Option(
        False,
        "--force-resplit",
        help="Regenerate frozen splits. Invalidates every result citing the old split.",
    ),
    download: bool = typer.Option(
        True,
        "--download/--no-download",
        help="Fetch the dataset if it is missing locally.",
    ),
    force_download: bool = typer.Option(
        False, "--force-download", help="Re-fetch even if the file is already present."
    ),
) -> None:
    """Download, validate, split and freeze a benchmark dataset."""
    from evidence_route.datasets import prepare_dataset

    try:
        result = prepare_dataset(
            config,
            source_path=source,
            force_resplit=force_resplit,
            download=download,
            force_download=force_download,
        )
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except (NotImplementedError, ValueError, OSError, RuntimeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if result.downloaded:
        typer.secho("Downloaded source data.", fg=typer.colors.GREEN)

    typer.secho(result.report.summary(), bold=True)

    if result.parse_failures:
        typer.secho(
            f"  {len(result.parse_failures)} row(s) failed to parse:",
            fg=typer.colors.YELLOW,
        )
        for failure in result.parse_failures[:5]:
            typer.echo(f"    {failure}")
        if len(result.parse_failures) > 5:
            typer.echo(f"    ... and {len(result.parse_failures) - 5} more")

    origin = "reused frozen" if result.reused_frozen_splits else "generated"
    typer.echo(
        f"\nSplits ({origin}, group_by={result.assignment.group_by}, "
        f"seed={result.assignment.seed}):"
    )
    group_counts = result.assignment.group_counts()
    for split, count in sorted(result.assignment.counts().items()):
        share = 100 * count / len(result.records) if result.records else 0
        typer.echo(
            f"  {split:<11} {count:>5} questions ({share:4.1f}%)  "
            f"{group_counts.get(split, 0):>3} groups"
        )

    typer.echo(f"\n  splits   -> {result.splits_path}")
    typer.echo(f"  manifest -> {result.manifest_path}")

    if result.reused_frozen_splits:
        typer.secho(
            "\nExisting splits were reused. Pass --force-resplit to regenerate "
            "(this invalidates results citing the old split).",
            fg=typer.colors.YELLOW,
        )


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
