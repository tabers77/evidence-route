"""Tests for generation providers, caching and cost.

No network anywhere. What is checked is the logic that governs money and
comparability — cache identity, offline enforcement, retry accounting and the
distinction between unknown and zero cost.
"""

from __future__ import annotations

import pytest

from evidence_route.generation.cache import CachingProvider, ResponseCache
from evidence_route.generation.client import (
    GenerationRequest,
    GenerationResponse,
    ProviderError,
    RecordedProvider,
    ScriptedProvider,
    build_provider,
)
from evidence_route.generation.cost import (
    PRICING,
    TokenPricing,
    estimate_cost,
    register_pricing,
)
from evidence_route.generation.prompts import (
    DIRECT_V1,
    GROUNDED_V1,
    format_evidence,
    get_template,
)
from evidence_route.storage.records import RetrievedItem, TokenUsage


def _request(**overrides) -> GenerationRequest:
    kwargs = {
        "system": "You are an analyst.",
        "user": "What was revenue?",
        "deployment_ref": "chat",
        "prompt_version": "grounded_v1",
        "temperature": 0.0,
        "max_output_tokens": 512,
    }
    kwargs.update(overrides)
    return GenerationRequest(**kwargs)


# ---------------------------------------------------------------------------
# Cache identity — what makes caching model output scientifically valid
# ---------------------------------------------------------------------------
def test_identical_requests_share_a_key():
    assert _request().cache_key() == _request().cache_key()


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("system", "different system"),
        ("user", "different question"),
        ("deployment_ref", "cheap"),
        ("prompt_version", "grounded_v2"),
        ("temperature", 0.7),
        ("max_output_tokens", 1024),
        ("seed", 42),
    ],
)
def test_every_field_that_changes_output_changes_the_key(field_name, value):
    """A key omitting any of these would serve a response produced under
    different conditions — a silent substitution, not a cache hit."""
    assert _request().cache_key() != _request(**{field_name: value}).cache_key()


def test_cache_key_is_a_stable_hex_digest():
    key = _request().cache_key()
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
def test_cache_round_trip(tmp_path):
    cache = ResponseCache(tmp_path)
    request = _request()
    assert cache.get(request) is None

    cache.put(request, GenerationResponse(text="42", input_tokens=10, output_tokens=2))
    hit = cache.get(request)

    assert hit is not None
    assert hit.text == "42"
    assert hit.from_cache is True
    assert cache.stats()["hits"] == 1


def test_cache_leaves_no_partial_files(tmp_path):
    cache = ResponseCache(tmp_path)
    cache.put(_request(), GenerationResponse(text="x"))
    assert list(tmp_path.rglob("*.partial")) == []


def test_corrupt_entry_is_a_miss_not_a_crash(tmp_path):
    cache = ResponseCache(tmp_path)
    request = _request()
    cache.put(request, GenerationResponse(text="x"))

    path = next(tmp_path.rglob("*.json"))
    path.write_text("{ not json", encoding="utf-8")

    assert cache.get(request) is None


def test_cache_is_sharded(tmp_path):
    """A flat directory of tens of thousands of files is slow to list."""
    cache = ResponseCache(tmp_path)
    cache.put(_request(), GenerationResponse(text="x"))
    written = next(tmp_path.rglob("*.json"))
    assert written.parent != tmp_path


# ---------------------------------------------------------------------------
# CachingProvider
# ---------------------------------------------------------------------------
def test_second_identical_call_is_not_re_billed(tmp_path):
    """The property that makes a partial rerun affordable."""
    inner = ScriptedProvider(responses=['{"answer": "first"}', '{"answer": "second"}'])
    provider = CachingProvider(inner=inner, cache=ResponseCache(tmp_path))

    first = provider.complete(_request())
    second = provider.complete(_request())

    assert first.text == '{"answer": "first"}'
    assert second.text == '{"answer": "first"}'
    assert len(inner.requests) == 1, "the second call reached the provider"


def test_different_requests_both_reach_the_provider(tmp_path):
    inner = ScriptedProvider()
    provider = CachingProvider(inner=inner, cache=ResponseCache(tmp_path))
    provider.complete(_request(user="q1"))
    provider.complete(_request(user="q2"))
    assert len(inner.requests) == 2


def test_truncated_responses_are_not_cached(tmp_path):
    """Caching one would make a too-small max_output_tokens permanent."""
    inner = ScriptedProvider()

    class Truncating:
        name = "truncating"

        def complete(self, request):
            return GenerationResponse(text="cut off", finish_reason="length")

    provider = CachingProvider(inner=Truncating(), cache=ResponseCache(tmp_path))
    provider.complete(_request())
    assert provider.cache.writes == 0

    # And a normal response still is cached.
    ok = CachingProvider(inner=inner, cache=ResponseCache(tmp_path))
    ok.complete(_request())
    assert ok.cache.writes == 1


def test_caching_can_be_disabled(tmp_path):
    inner = ScriptedProvider()
    provider = CachingProvider(inner=inner, cache=ResponseCache(tmp_path), enabled=False)
    provider.complete(_request())
    provider.complete(_request())
    assert len(inner.requests) == 2


def test_truncation_is_detected():
    assert GenerationResponse(text="x", finish_reason="length").was_truncated
    assert not GenerationResponse(text="x", finish_reason="stop").was_truncated


# ---------------------------------------------------------------------------
# Recorded provider and the offline guarantee
# ---------------------------------------------------------------------------
def test_recorded_provider_replays(tmp_path):
    provider = RecordedProvider(fixtures_dir=tmp_path)
    request = _request()
    provider.record(request, GenerationResponse(text='{"answer": "recorded"}', input_tokens=7))

    replayed = provider.complete(request)
    assert replayed.text == '{"answer": "recorded"}'
    assert replayed.from_cache is True


def test_recorded_miss_fails_loudly(tmp_path):
    """Falling through to a live call would make CI a billed job the first
    time a prompt changed."""
    with pytest.raises(ProviderError, match="No recorded response"):
        RecordedProvider(fixtures_dir=tmp_path).complete(_request())


def test_recorded_miss_names_the_prompt_version(tmp_path):
    with pytest.raises(ProviderError, match="grounded_v1"):
        RecordedProvider(fixtures_dir=tmp_path).complete(_request())


def test_non_strict_recorded_provider_returns_a_miss(tmp_path):
    response = RecordedProvider(fixtures_dir=tmp_path, strict=False).complete(_request())
    assert response.finish_reason == "cache_miss"


def test_recorded_fixture_stores_the_request_for_readability(tmp_path):
    """A fixture directory of bare hashes is unreadable."""
    import json

    provider = RecordedProvider(fixtures_dir=tmp_path)
    path = provider.record(_request(), GenerationResponse(text="x"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["request"]["prompt_version"] == "grounded_v1"


def test_offline_settings_select_the_recorded_provider(tmp_path, no_azure_env):
    from evidence_route.config import Settings

    provider = build_provider(Settings(_env_file=None, offline=True), fixtures_dir=tmp_path)
    assert isinstance(provider, RecordedProvider)


def test_offline_without_fixtures_is_refused(no_azure_env):
    from evidence_route.config import Settings

    with pytest.raises(ProviderError, match="fixtures directory"):
        build_provider(Settings(_env_file=None, offline=True))


def test_live_provider_requires_credentials(no_azure_env):
    from evidence_route.config import Settings

    with pytest.raises(RuntimeError, match="Azure OpenAI is not configured"):
        build_provider(Settings(_env_file=None, offline=False))


# ---------------------------------------------------------------------------
# Cost — unknown must not become zero
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_pricing():
    original = dict(PRICING)
    PRICING.clear()
    yield
    PRICING.clear()
    PRICING.update(original)


def test_unregistered_model_yields_unknown_not_zero():
    """Recording 0.0 for an unpriced call would make the budget ceiling
    useless: a run could exhaust a real budget reporting it spent nothing."""
    estimate = estimate_cost(TokenUsage(input_tokens=1000, output_tokens=100), "mystery")
    assert estimate.usd is None
    assert estimate.pricing_known is False
    assert "register_pricing" in (estimate.note or "")


def test_missing_model_key_yields_unknown():
    assert estimate_cost(TokenUsage(input_tokens=10), None).pricing_known is False


def test_registered_pricing_computes_cost():
    register_pricing(
        "test-model",
        TokenPricing(
            input_usd_per_million=2.50,
            output_usd_per_million=10.0,
            source="test fixture",
        ),
    )
    estimate = estimate_cost(
        TokenUsage(input_tokens=1_000_000, output_tokens=100_000), "test-model"
    )
    assert estimate.pricing_known
    assert estimate.usd == pytest.approx(2.50 + 1.0)


def test_retry_tokens_are_billed():
    """Excluding them understates spend on exactly the hardest questions."""
    register_pricing(
        "m", TokenPricing(input_usd_per_million=1.0, output_usd_per_million=1.0, source="t")
    )
    without = estimate_cost(TokenUsage(input_tokens=1_000_000), "m")
    with_retries = estimate_cost(
        TokenUsage(input_tokens=1_000_000, retry_input_tokens=1_000_000), "m"
    )
    assert with_retries.usd == pytest.approx(2 * (without.usd or 0))


def test_model_key_lookup_is_case_insensitive():
    register_pricing(
        "GPT-Test", TokenPricing(input_usd_per_million=1.0, output_usd_per_million=1.0, source="t")
    )
    assert estimate_cost(TokenUsage(input_tokens=100), "gpt-test").pricing_known


def test_usd_or_zero_is_available_for_arithmetic():
    estimate = estimate_cost(TokenUsage(input_tokens=10), "unknown")
    assert estimate.usd_or_zero == 0.0


def test_no_pricing_is_guessed_by_default():
    """An invented price is worse than none: it produces a confident number
    nobody re-checks."""
    assert PRICING == {}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def test_templates_are_versioned():
    assert get_template("grounded_v1") is GROUNDED_V1
    assert get_template("direct_v1") is DIRECT_V1


def test_unknown_prompt_version_raises():
    """A run silently using a different prompt would attribute results wrongly."""
    with pytest.raises(KeyError, match="Unknown prompt version"):
        get_template("grounded_v99")


def test_grounded_prompt_forbids_outside_knowledge():
    system = GROUNDED_V1.system.lower()
    assert "only from the evidence" in system
    assert "never invent" in system


def test_grounded_prompt_permits_abstention():
    assert "abstain" in GROUNDED_V1.system.lower()
    assert "insufficient_evidence" in GROUNDED_V1.system


def test_direct_prompt_states_there_are_no_documents():
    assert "no documents" in DIRECT_V1.system.lower()


def test_both_templates_specify_the_same_response_contract():
    for template in (DIRECT_V1, GROUNDED_V1):
        for field_name in ("answer", "citations", "confidence", "abstained"):
            assert f'"{field_name}"' in template.system


def test_every_abstention_reason_code_is_offered():
    from evidence_route.workflows.actions import AbstentionReason

    for reason in AbstentionReason:
        assert reason.value in GROUNDED_V1.system


def test_evidence_is_rendered_with_citable_ids():
    """Without visible ids the model invents references and no citation is
    checkable."""
    items = [
        RetrievedItem(chunk_id="D::c0001", document_id="D", rank=1, text="Revenue 100"),
        RetrievedItem(chunk_id="D::c0002", document_id="D", rank=2, text="Costs 40"),
    ]
    rendered = format_evidence(items)
    assert "[D::c0001]" in rendered
    assert "Revenue 100" in rendered
    assert "[D::c0002]" in rendered


def test_empty_evidence_is_stated_explicitly():
    assert "no evidence" in format_evidence([]).lower()


def test_user_template_renders():
    rendered = GROUNDED_V1.render_user(evidence="[c1] text", question="What?")
    assert "[c1] text" in rendered
    assert "What?" in rendered


# ---------------------------------------------------------------------------
# Scripted provider
# ---------------------------------------------------------------------------
def test_scripted_provider_records_requests():
    provider = ScriptedProvider(responses=['{"answer": "a"}'])
    provider.complete(_request(user="first"))
    provider.complete(_request(user="second"))

    assert [r.user for r in provider.requests] == ["first", "second"]
    assert provider.requests[0].prompt_version == "grounded_v1"
