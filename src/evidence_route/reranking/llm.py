"""LLM-based reranking.

The other candidate for protocol decision 6. The model is shown the question and
the numbered candidates, and asked to grade each one's relevance.

**Listwise, not pointwise.** All candidates go in one call. Scoring each
separately would cost one call per candidate — thirty calls per question at
A4's candidate width — and would also deny the model the comparison that makes
grading easy. Cost is the deciding factor: pointwise would make A4 the most
expensive action in the suite by a wide margin, which would settle decision 6 on
billing rather than on retrieval quality.

**A failed rerank returns retrieval's order unchanged**, rather than raising.
Reranking is a refinement; when it fails the candidates are still perfectly
usable, and losing the whole outcome over a formatting error would discard a
retrieval result that cost real money to produce. The fallback is recorded so it
is visible in the outcome rather than passing as a successful rerank.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from evidence_route.generation.client import (
    GenerationProvider,
    GenerationRequest,
    ProviderError,
)
from evidence_route.generation.schema import extract_json_object
from evidence_route.reranking.base import reorder
from evidence_route.storage.records import RetrievedItem

__all__ = ["RERANK_PROMPT_V1", "LLMReranker"]

RERANK_PROMPT_V1 = """You are grading how well each passage answers a question \
about a company filing.

Score every passage from 0 to 10:
  10  contains the exact answer
   7  contains most of what is needed
   4  related but does not answer it
   0  irrelevant

Judge only whether the passage answers THIS question. A passage about the right \
topic but the wrong period or the wrong company scores low.

Respond with a single JSON object and nothing else:

{"scores": [{"id": <passage number>, "score": <0-10>}]}

Score every passage exactly once."""


@dataclass
class LLMReranker:
    """Listwise reranking through a generation provider."""

    provider: GenerationProvider
    deployment_ref: str = "chat"
    prompt_version: str = "rerank_v1"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    #: Set when the last rerank fell back to retrieval order.
    last_fallback_reason: str | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "llm_reranker"

    @property
    def model_version(self) -> str:
        return f"{self.deployment_ref}:{self.prompt_version}"

    @property
    def is_learned(self) -> bool:
        return True

    def _build_user_prompt(self, query: str, items: list[RetrievedItem]) -> str:
        blocks = [f"[{i}] {(item.text or '').strip()}" for i, item in enumerate(items, start=1)]
        passages = "\n\n".join(blocks)
        return f"Question: {query}\n\nPassages:\n\n{passages}"

    def rerank(self, query: str, items: list[RetrievedItem], k: int = 10) -> list[RetrievedItem]:
        self.last_fallback_reason = None
        if not items:
            return []

        request = GenerationRequest(
            system=RERANK_PROMPT_V1,
            user=self._build_user_prompt(query, items),
            deployment_ref=self.deployment_ref,
            prompt_version=self.prompt_version,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        try:
            response = self.provider.complete(request)
            payload = extract_json_object(response.text)
        except (ProviderError, ValueError, json.JSONDecodeError) as exc:
            self.last_fallback_reason = f"rerank_failed: {exc}"
            return self._retrieval_order(items, k)

        scores = self._parse_scores(payload, len(items))
        if scores is None:
            self.last_fallback_reason = "rerank_unusable: no valid scores returned"
            return self._retrieval_order(items, k)

        return reorder(items, scores, k)

    def _parse_scores(self, payload: dict, n_items: int) -> list[float] | None:
        """Read scores, defaulting anything the model omitted.

        A missing passage keeps a neutral score rather than sinking to zero: the
        model failing to mention it is not evidence that it is irrelevant, and
        treating silence as a zero would let a truncated response quietly
        discard good candidates.
        """
        entries = payload.get("scores")
        if not isinstance(entries, list) or not entries:
            return None

        scores = [5.0] * n_items
        seen = False
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry["id"]) - 1
                score = float(entry["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < n_items:
                scores[index] = score
                seen = True
        return scores if seen else None

    def _retrieval_order(self, items: list[RetrievedItem], k: int) -> list[RetrievedItem]:
        """Fall back to retrieval's ranking, preserving its positions."""
        return reorder(items, [-float(item.rank) for item in items], k)
