"""Content-addressed response cache (spec section 26, items 4-5).

The outcome matrix runs every action against every question. A run that fails
partway — a rate limit, a bad deployment name, an interrupted laptop — must be
resumable without paying again for the calls that already succeeded. Without a
cache, "avoid rerunning unchanged actions" is not achievable and the budget
ceiling gets consumed by repeated work.

Caching model output is scientifically valid **only** because the cache key
covers everything that can change the output: deployment, prompt version,
messages, temperature, token limit and seed. A key omitting any of those would
serve a response produced under different conditions, which is not a cache hit
but a silent substitution.

What is deliberately *not* cached: failures. A rate-limited or errored call is
retried on the next run rather than replayed, because the failure is a property
of that moment, not of the request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from evidence_route.generation.client import (
    GenerationProvider,
    GenerationRequest,
    GenerationResponse,
)

__all__ = ["CachingProvider", "ResponseCache"]


@dataclass
class ResponseCache:
    """A directory of responses keyed by request content hash."""

    directory: Path
    hits: int = 0
    misses: int = 0
    writes: int = 0

    def _path_for(self, key: str) -> Path:
        # Sharded by the first two hex characters: a flat directory of tens of
        # thousands of files is slow to list on most filesystems.
        return self.directory / key[:2] / f"{key}.json"

    def get(self, request: GenerationRequest) -> GenerationResponse | None:
        path = self._path_for(request.cache_key())
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt entry is a miss, not a crash. The call is simply remade.
            self.misses += 1
            return None

        self.hits += 1
        return GenerationResponse(
            text=payload.get("text", ""),
            input_tokens=payload.get("input_tokens", 0),
            output_tokens=payload.get("output_tokens", 0),
            model=payload.get("model"),
            finish_reason=payload.get("finish_reason"),
            from_cache=True,
        )

    def put(self, request: GenerationRequest, response: GenerationResponse) -> Path:
        """Store a response.

        Written atomically so an interrupted write cannot leave a truncated
        entry that later parses as a valid but wrong response.
        """
        path = self._path_for(request.cache_key())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "model": response.model,
            "finish_reason": response.finish_reason,
            "prompt_version": request.prompt_version,
            "deployment_ref": request.deployment_ref,
        }
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        self.writes += 1
        return path

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


@dataclass
class CachingProvider:
    """Wraps a provider so repeated identical requests are not re-billed."""

    inner: GenerationProvider
    cache: ResponseCache
    #: Truncated responses are not stored. They usually mean max_output_tokens
    #: was too low, and caching one would make that misconfiguration permanent
    #: for every later run against the same request.
    skip_truncated: bool = True
    enabled: bool = True

    @property
    def name(self) -> str:
        return f"cached({self.inner.name})"

    def complete(self, request: GenerationRequest) -> GenerationResponse:
        if not self.enabled:
            return self.inner.complete(request)

        cached = self.cache.get(request)
        if cached is not None:
            return cached

        response = self.inner.complete(request)
        if not (self.skip_truncated and response.was_truncated):
            self.cache.put(request, response)
        return response
