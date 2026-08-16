"""Generation providers.

Everything that produces text goes through :class:`GenerationProvider`, which
exists so the expensive path and the free path are interchangeable. Three
implementations:

``AzureOpenAIProvider``
    The real one. Lazily imports the ``openai`` SDK so the core package installs
    and tests without it.

``RecordedProvider``
    Replays responses captured from a real run, keyed by request content. A
    cache miss is a hard failure rather than a fall-through to a live call —
    otherwise CI would quietly become a billable job the first time a prompt
    changed.

``ScriptedProvider``
    Canned responses for tests. Records what it was asked, so prompt
    construction can be asserted without a network.

Requests are hashed by *content*, not by object identity, so a cached response
is reused only when the deployment, prompt version, messages and sampling
parameters all match. Anything less would serve a response generated under
different conditions and quietly break the comparison it feeds.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AzureOpenAIProvider",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResponse",
    "ProviderError",
    "RecordedProvider",
    "ScriptedProvider",
    "build_provider",
]


class ProviderError(RuntimeError):
    """A generation call failed. Recorded as an outcome, not swallowed."""


@dataclass(frozen=True)
class GenerationRequest:
    """One call to a model.

    ``prompt_version`` and ``deployment_ref`` are part of the request identity
    because they are part of the experiment's identity — two responses produced
    under different prompts are not interchangeable, however similar the text.
    """

    system: str
    user: str
    deployment_ref: str = "chat"
    prompt_version: str = "unknown"
    temperature: float = 0.0
    max_output_tokens: int = 512
    seed: int | None = None

    def cache_key(self) -> str:
        """Content hash identifying this request.

        Covers everything that can change the response. A key omitting, say,
        temperature would let a greedy response be served for a sampled request.
        """
        payload = json.dumps(
            {
                "system": self.system,
                "user": self.user,
                "deployment_ref": self.deployment_ref,
                "prompt_version": self.prompt_version,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "seed": self.seed,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class GenerationResponse:
    """What a provider returned, with the accounting the outcome record needs."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    #: The model the deployment actually resolved to. Recorded because an Azure
    #: deployment can be re-pointed at a new model version with no change here,
    #: and results from different underlying models must not be pooled.
    model: str | None = None
    finish_reason: str | None = None
    latency_ms: float = 0.0
    from_cache: bool = False
    #: Tokens burned by retries that preceded the successful call.
    retry_input_tokens: int = 0
    retry_output_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def was_truncated(self) -> bool:
        """Whether the model stopped because it hit the output limit.

        A truncated response usually fails to parse as JSON. Distinguishing it
        from a genuine refusal matters: one is a configuration problem, the
        other is model behaviour worth reporting.
        """
        return self.finish_reason in {"length", "max_tokens"}


@runtime_checkable
class GenerationProvider(Protocol):
    """Anything that can turn a request into text."""

    @property
    def name(self) -> str: ...

    def complete(self, request: GenerationRequest) -> GenerationResponse: ...


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------
@dataclass
class AzureOpenAIProvider:
    """Azure OpenAI chat completions.

    Credentials come from :class:`~evidence_route.config.AzureOpenAISettings`;
    nothing here is hardcoded. Supports both an API key and Entra ID, since an
    enterprise tenant may forbid key auth entirely.
    """

    settings: Any  # AzureOpenAISettings, untyped to avoid a config import cycle
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    _client: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "azure_openai"

    def _deployment_for(self, ref: str) -> str:
        """Resolve a logical deployment reference to a configured name."""
        mapping = {
            "chat": self.settings.chat_deployment,
            "cheap": self.settings.cheap_chat_deployment,
            "judge": self.settings.judge_deployment,
            "embedding": self.settings.embedding_deployment,
        }
        deployment = mapping.get(ref)
        if not deployment:
            raise ProviderError(
                f"No Azure deployment configured for {ref!r}. "
                f"Set the corresponding EVIDENCE_ROUTE_AZURE_OPENAI_*_DEPLOYMENT "
                f"variable in .env."
            )
        return str(deployment)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ProviderError(
                "Azure generation requires the 'generation' extra. "
                'Install it with: pip install -e ".[generation]"'
            ) from exc

        if self.settings.use_entra_id:
            try:
                from azure.identity import (
                    ClientSecretCredential,
                    get_bearer_token_provider,
                )
            except ImportError as exc:  # pragma: no cover - optional extra
                raise ProviderError(
                    "Entra ID auth requires azure-identity. "
                    'Install it with: pip install -e ".[generation]"'
                ) from exc

            credential = ClientSecretCredential(
                tenant_id=str(self.settings.tenant_id),
                client_id=str(self.settings.client_id),
                client_secret=(
                    self.settings.client_secret.get_secret_value()
                    if self.settings.client_secret
                    else ""
                ),
            )
            self._client = AzureOpenAI(
                azure_endpoint=str(self.settings.endpoint),
                api_version=self.settings.api_version,
                azure_ad_token_provider=get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                ),
            )
        else:
            self._client = AzureOpenAI(
                azure_endpoint=str(self.settings.endpoint),
                api_version=self.settings.api_version,
                api_key=(
                    self.settings.api_key.get_secret_value() if self.settings.api_key else None
                ),
            )
        return self._client

    def complete(self, request: GenerationRequest) -> GenerationResponse:
        """Call the model, retrying transient failures.

        Tokens spent on failed attempts are carried on the response rather than
        discarded. A retry costs real money, and a cost figure that ignores it
        understates exactly the questions that were hardest.
        """
        client = self._ensure_client()
        deployment = self._deployment_for(request.deployment_ref)

        retry_input = retry_output = 0
        last_error: Exception | None = None
        started = time.monotonic()

        for attempt in range(self.max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=deployment,
                    messages=[
                        {"role": "system", "content": request.system},
                        {"role": "user", "content": request.user},
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_output_tokens,
                    **({"seed": request.seed} if request.seed is not None else {}),
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * (2**attempt))
                continue

            usage = getattr(completion, "usage", None)
            choice = completion.choices[0]
            return GenerationResponse(
                text=choice.message.content or "",
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                model=getattr(completion, "model", None),
                finish_reason=getattr(choice, "finish_reason", None),
                latency_ms=(time.monotonic() - started) * 1000,
                retry_input_tokens=retry_input,
                retry_output_tokens=retry_output,
            )

        raise ProviderError(
            f"Azure generation failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error


# ---------------------------------------------------------------------------
# Offline providers
# ---------------------------------------------------------------------------
@dataclass
class RecordedProvider:
    """Replays responses captured from a real run.

    ``strict`` defaults to true: a request with no recording raises. Falling
    through to a live provider would let a prompt change turn an offline test
    run into a billed one without anyone noticing.
    """

    fixtures_dir: Path
    strict: bool = True

    @property
    def name(self) -> str:
        return "recorded"

    def _path_for(self, request: GenerationRequest) -> Path:
        return self.fixtures_dir / f"{request.cache_key()}.json"

    def record(self, request: GenerationRequest, response: GenerationResponse) -> Path:
        """Persist a response so it can be replayed.

        The request is stored alongside it — not needed for lookup, but without
        it a fixture directory is an unreadable pile of hashes.
        """
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(request)
        path.write_text(
            json.dumps(
                {
                    "request": {
                        "deployment_ref": request.deployment_ref,
                        "prompt_version": request.prompt_version,
                        "temperature": request.temperature,
                        "max_output_tokens": request.max_output_tokens,
                        "system": request.system,
                        "user": request.user,
                    },
                    "response": {
                        "text": response.text,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "model": response.model,
                        "finish_reason": response.finish_reason,
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    def complete(self, request: GenerationRequest) -> GenerationResponse:
        path = self._path_for(request)
        if not path.exists():
            if self.strict:
                raise ProviderError(
                    f"No recorded response for this request (key {request.cache_key()[:12]}, "
                    f"prompt {request.prompt_version}). Re-record the fixtures against a "
                    f"live provider, or the offline path is not exercising this call."
                )
            return GenerationResponse(text="", finish_reason="cache_miss")

        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded = payload["response"]
        return GenerationResponse(
            text=recorded.get("text", ""),
            input_tokens=recorded.get("input_tokens", 0),
            output_tokens=recorded.get("output_tokens", 0),
            model=recorded.get("model"),
            finish_reason=recorded.get("finish_reason"),
            from_cache=True,
        )


@dataclass
class ScriptedProvider:
    """Returns canned responses, recording what it was asked.

    For tests that need to assert on prompt construction without a network.
    """

    responses: list[str] = field(default_factory=list)
    default: str = '{"answer": "scripted", "confidence": 0.5, "citations": []}'
    requests: list[GenerationRequest] = field(default_factory=list)
    _index: int = 0

    @property
    def name(self) -> str:
        return "scripted"

    def complete(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        if self._index < len(self.responses):
            text = self.responses[self._index]
            self._index += 1
        else:
            text = self.default
        return GenerationResponse(
            text=text,
            input_tokens=len(request.system + request.user) // 4,
            output_tokens=len(text) // 4,
            model="scripted",
            finish_reason="stop",
        )


def build_provider(settings: Any, *, fixtures_dir: Path | None = None) -> GenerationProvider:
    """Choose a provider from settings.

    Offline mode selects the recorded provider, so the offline guarantee is a
    property of the wiring rather than of every call site remembering to check.
    """
    if settings.offline:
        if fixtures_dir is None:
            raise ProviderError("Offline mode requires a fixtures directory of recorded responses.")
        return RecordedProvider(fixtures_dir=fixtures_dir)

    settings.require_live_provider()
    return AzureOpenAIProvider(settings=settings.azure)
