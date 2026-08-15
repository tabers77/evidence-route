"""Tests for environment settings.

The property under test throughout: credentials come from the environment, are
never hardcoded, and never leak through a repr or a serialized manifest.
"""

from __future__ import annotations

import pytest

from evidence_route.config import PROJECT_ROOT, AzureOpenAISettings, Settings


def test_project_root_points_at_the_repository():
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert (PROJECT_ROOT / "src" / "evidence_route").is_dir()


def test_paths_are_absolute(no_azure_env):
    settings = Settings(_env_file=None)
    assert settings.data_dir.is_absolute()
    assert settings.runs_dir.is_absolute()
    assert settings.cache_dir.is_absolute()


def test_relative_paths_resolve_against_project_root(no_azure_env):
    settings = Settings(_env_file=None, data_dir="data/custom")
    assert settings.data_dir.is_absolute()
    assert settings.data_dir.parts[-2:] == ("data", "custom")


def test_test_split_is_protected_by_default(no_azure_env):
    """Guards against final-test-driven tuning; opting out must be deliberate."""
    assert Settings(_env_file=None).protect_test_split is True


def test_azure_unconfigured_by_default(no_azure_env):
    azure = AzureOpenAISettings(_env_file=None)
    assert not azure.is_configured()
    assert "EVIDENCE_ROUTE_AZURE_OPENAI_ENDPOINT" in azure.missing_fields()


def test_missing_fields_reports_names_never_values(no_azure_env, monkeypatch):
    monkeypatch.setenv("EVIDENCE_ROUTE_AZURE_OPENAI_ENDPOINT", "https://example.invalid/")
    monkeypatch.setenv("EVIDENCE_ROUTE_AZURE_OPENAI_API_KEY", "super-secret-value")
    azure = AzureOpenAISettings(_env_file=None)
    missing = azure.missing_fields()
    assert "super-secret-value" not in " ".join(missing)
    # Endpoint and key are set, but the chat deployment is still required.
    assert "EVIDENCE_ROUTE_AZURE_OPENAI_CHAT_DEPLOYMENT" in missing


def test_api_key_auth_is_recognised(no_azure_env, monkeypatch):
    monkeypatch.setenv("EVIDENCE_ROUTE_AZURE_OPENAI_ENDPOINT", "https://example.invalid/")
    monkeypatch.setenv("EVIDENCE_ROUTE_AZURE_OPENAI_API_KEY", "k")
    assert AzureOpenAISettings(_env_file=None).is_configured()


def test_entra_id_auth_is_recognised_without_an_api_key(no_azure_env, monkeypatch):
    monkeypatch.setenv("EVIDENCE_ROUTE_AZURE_OPENAI_ENDPOINT", "https://example.invalid/")
    monkeypatch.setenv("EVIDENCE_ROUTE_AZURE_USE_ENTRA_ID", "true")
    monkeypatch.setenv("EVIDENCE_ROUTE_AZURE_TENANT_ID", "tenant")
    monkeypatch.setenv("EVIDENCE_ROUTE_AZURE_CLIENT_ID", "client")
    azure = AzureOpenAISettings(_env_file=None)
    assert azure.is_configured()
    assert azure.api_key is None


def test_secrets_do_not_leak_through_repr(no_azure_env, monkeypatch):
    """An accidental print or a config dump into a run manifest must not expose keys."""
    monkeypatch.setenv("EVIDENCE_ROUTE_AZURE_OPENAI_ENDPOINT", "https://example.invalid/")
    monkeypatch.setenv("EVIDENCE_ROUTE_AZURE_OPENAI_API_KEY", "leaked-key-12345")
    azure = AzureOpenAISettings(_env_file=None)

    assert "leaked-key-12345" not in repr(azure)
    assert "leaked-key-12345" not in str(azure)
    assert "leaked-key-12345" not in str(azure.model_dump())
    # The real value is still reachable deliberately.
    assert azure.api_key is not None
    assert azure.api_key.get_secret_value() == "leaked-key-12345"


def test_require_live_provider_refuses_in_offline_mode(no_azure_env):
    settings = Settings(_env_file=None, offline=True)
    with pytest.raises(RuntimeError, match="OFFLINE"):
        settings.require_live_provider()


def test_require_live_provider_names_the_missing_variables(no_azure_env):
    settings = Settings(_env_file=None, offline=False)
    with pytest.raises(RuntimeError, match="EVIDENCE_ROUTE_AZURE_OPENAI_ENDPOINT"):
        settings.require_live_provider()


def test_budget_ceiling_rejects_negative_values(no_azure_env):
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        Settings(_env_file=None, max_experiment_budget_usd=-5.0)


def test_get_settings_is_cached(clean_settings_cache):
    from evidence_route.config import get_settings

    assert get_settings() is get_settings()
