"""Typed runtime configuration.

Two distinct kinds of configuration exist in this project, and they are kept
apart on purpose:

**Environment settings** (this module) — credentials, endpoints, paths, budget
ceilings. These vary per machine, contain secrets, and are read from the
environment or a gitignored ``.env``. They are *never* committed and never
written into experiment artifacts.

**Experiment configuration** (``configs/*.yaml``) — datasets, splits, workflow
parameters, router hyperparameters, reward weights. These are version-controlled
because a result is only reproducible if the configuration that produced it is
in git.

Secrets use :class:`pydantic.SecretStr`, so an accidental ``print(settings)`` or
a config dump into a run manifest yields ``**********`` rather than a key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root, resolved from this file's location (src/evidence_route/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI connection details.

    Every field is supplied by the environment. No endpoint, deployment name,
    tenant or key is hardcoded anywhere in this repository — see ``.env.example``
    for the variable names and ``data/README.md`` for the reasoning.

    Authentication is either an API key or Entra ID (managed identity / service
    principal). Exactly one must be configured before the generation layer runs;
    :meth:`is_configured` reports whether that is the case, so offline paths can
    check rather than crash.
    """

    model_config = SettingsConfigDict(
        env_prefix="EVIDENCE_ROUTE_AZURE_",
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str | None = Field(
        default=None,
        validation_alias="EVIDENCE_ROUTE_AZURE_OPENAI_ENDPOINT",
        description="Base endpoint of the Azure OpenAI resource.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias="EVIDENCE_ROUTE_AZURE_OPENAI_API_KEY",
    )
    api_version: str = Field(
        default="2024-10-21",
        validation_alias="EVIDENCE_ROUTE_AZURE_OPENAI_API_VERSION",
    )

    # Deployment names, not model names. The mapping from deployment to
    # underlying model version is recorded per run for reproducibility
    # (spec section 25), because a deployment can be re-pointed silently.
    chat_deployment: str | None = Field(
        default=None,
        validation_alias="EVIDENCE_ROUTE_AZURE_OPENAI_CHAT_DEPLOYMENT",
    )
    cheap_chat_deployment: str | None = Field(
        default=None,
        validation_alias="EVIDENCE_ROUTE_AZURE_OPENAI_CHEAP_CHAT_DEPLOYMENT",
        description="Smaller model used for pipeline debugging (spec section 26).",
    )
    embedding_deployment: str | None = Field(
        default=None,
        validation_alias="EVIDENCE_ROUTE_AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    )
    judge_deployment: str | None = Field(
        default=None,
        validation_alias="EVIDENCE_ROUTE_AZURE_OPENAI_JUDGE_DEPLOYMENT",
        description=(
            "Deployment used for LLM-judge scoring. Prefer a different model "
            "family from the generator so the judge does not grade its own "
            "outputs (spec section 27)."
        ),
    )

    # Entra ID alternative to the API key.
    use_entra_id: bool = Field(default=False, validation_alias="EVIDENCE_ROUTE_AZURE_USE_ENTRA_ID")
    tenant_id: str | None = Field(default=None, validation_alias="EVIDENCE_ROUTE_AZURE_TENANT_ID")
    client_id: str | None = Field(default=None, validation_alias="EVIDENCE_ROUTE_AZURE_CLIENT_ID")
    client_secret: SecretStr | None = Field(
        default=None, validation_alias="EVIDENCE_ROUTE_AZURE_CLIENT_SECRET"
    )

    def is_configured(self) -> bool:
        """Whether enough is set to attempt a call to Azure OpenAI."""
        if not self.endpoint:
            return False
        if self.use_entra_id:
            return bool(self.tenant_id and self.client_id)
        return self.api_key is not None

    def missing_fields(self) -> list[str]:
        """Human-readable list of what still needs to be set.

        Returns the *variable names*, never their values.
        """
        missing: list[str] = []
        if not self.endpoint:
            missing.append("EVIDENCE_ROUTE_AZURE_OPENAI_ENDPOINT")
        if self.use_entra_id:
            if not self.tenant_id:
                missing.append("EVIDENCE_ROUTE_AZURE_TENANT_ID")
            if not self.client_id:
                missing.append("EVIDENCE_ROUTE_AZURE_CLIENT_ID")
        elif self.api_key is None:
            missing.append("EVIDENCE_ROUTE_AZURE_OPENAI_API_KEY")
        if not self.chat_deployment:
            missing.append("EVIDENCE_ROUTE_AZURE_OPENAI_CHAT_DEPLOYMENT")
        return missing


class Settings(BaseSettings):
    """Top-level environment settings for a local or containerised run."""

    model_config = SettingsConfigDict(
        env_prefix="EVIDENCE_ROUTE_",
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths --------------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    cache_dir: Path = Field(default=PROJECT_ROOT / ".cache")
    runs_dir: Path = Field(default=PROJECT_ROOT / "experiments" / "runs")
    configs_dir: Path = Field(default=PROJECT_ROOT / "configs")

    # --- Local model alternatives ------------------------------------------
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Cost control (spec section 26) ------------------------------------
    max_experiment_budget_usd: float = Field(default=25.0, ge=0.0)
    enable_response_cache: bool = True

    # --- Reproducibility (spec section 25) ---------------------------------
    random_seed: int = 20260815
    protect_test_split: bool = Field(
        default=True,
        description=(
            "When true, any run that would read the frozen test split is "
            "refused unless the protocol is explicitly unlocked. Guards "
            "against final-test-driven tuning (spec section 21, week 11)."
        ),
    )

    # --- SEC EDGAR ----------------------------------------------------------
    #: EDGAR requires a User-Agent identifying the requester and including
    #: contact information, and returns 403 without one. It holds a real email
    #: address, so it comes from the environment rather than the repository.
    #: See https://www.sec.gov/os/webmaster-faq#developers
    sec_user_agent: str | None = Field(
        default=None,
        description="e.g. 'EvidenceRoute research you@example.com'. Required to fetch filings.",
    )

    # --- Execution mode -----------------------------------------------------
    offline: bool = Field(
        default=False,
        description="When true, no network or paid API calls are made; fixtures are used.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    azure: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        """Make every path absolute so behaviour does not depend on cwd."""
        for name in ("data_dir", "cache_dir", "runs_dir", "configs_dir"):
            value: Path = getattr(self, name)
            if not value.is_absolute():
                object.__setattr__(self, name, (PROJECT_ROOT / value).resolve())
        return self

    def require_live_provider(self) -> None:
        """Raise if a paid provider call is about to be made without credentials.

        Called at the entry point of any code path that bills money, so the
        failure is a clear configuration error rather than an opaque HTTP 401
        halfway through an expensive outcome-matrix run.
        """
        if self.offline:
            raise RuntimeError(
                "EVIDENCE_ROUTE_OFFLINE is true, but a live provider call was "
                "requested. Unset offline mode or use a fixture-backed path."
            )
        if not self.azure.is_configured():
            missing = ", ".join(self.azure.missing_fields())
            raise RuntimeError(
                f"Azure OpenAI is not configured. Copy .env.example to .env and set: {missing}"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once.

    Tests that need different settings should construct :class:`Settings`
    directly rather than mutating the cached instance; call
    ``get_settings.cache_clear()`` if the environment is changed deliberately.
    """
    return Settings()
