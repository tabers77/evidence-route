"""Embedding providers for dense retrieval.

Three implementations, and one property that separates them: ``is_semantic``.

``AzureEmbedder`` and ``LocalEmbedder`` produce genuine semantic vectors —
"revenue" and "net sales" land near each other. ``HashingEmbedder`` does not. It
hashes tokens into a fixed-width vector, which makes it deterministic, free and
dependency-free, and therefore ideal for exercising the dense pipeline offline
and in CI.

But a hashing vectorizer is **lexical**. Two documents are close only if they
share literal tokens, which is what BM25 already measures. Running the A1-vs-A2
comparison on it would show dense retrieval adding nothing — and that would be
an artifact of the stub, not a finding about embeddings. It is the kind of
conclusion that looks like a result and is entirely fake.

So ``is_semantic`` is a first-class property, and the retrieval comparison
refuses to draw conclusions from a non-semantic embedder.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AzureEmbedder",
    "Embedder",
    "EmbeddingError",
    "HashingEmbedder",
    "LocalEmbedder",
]

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9,.%$]*")


class EmbeddingError(RuntimeError):
    """An embedding call failed."""


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors."""

    @property
    def name(self) -> str: ...

    @property
    def model_version(self) -> str:
        """Identifies the model. Recorded per run and per index.

        An index built with a different embedding model is a *different index*;
        pooling results across them would compare two things that are not
        comparable.
        """
        ...

    @property
    def dimension(self) -> int: ...

    @property
    def is_semantic(self) -> bool:
        """Whether nearby vectors mean related things.

        False for the hashing stub, which only measures literal token overlap.
        """
        ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalize(vector: list[float]) -> list[float]:
    """Scale to unit length so cosine similarity reduces to a dot product."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


# ---------------------------------------------------------------------------
# Offline / deterministic
# ---------------------------------------------------------------------------
@dataclass
class HashingEmbedder:
    """Deterministic feature-hashing embedder. No model, no network, no cost.

    For exercising the dense code path offline. **Not** a substitute for real
    embeddings — see the module docstring.
    """

    dimensions: int = 256
    seed: int = 0

    @property
    def name(self) -> str:
        return "hashing"

    @property
    def model_version(self) -> str:
        return f"hashing-{self.dimensions}d-seed{self.seed}"

    @property
    def dimension(self) -> int:
        return self.dimensions

    @property
    def is_semantic(self) -> bool:
        return False

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.sha256(f"{self.seed}\x00{token}".encode()).digest()
        index = int.from_bytes(digest[:4], "big") % self.dimensions
        # Signed hashing keeps unrelated collisions from always adding up.
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in _TOKEN_RE.findall(text.lower()):
                index, sign = self._bucket(token)
                vector[index] += sign
            vectors.append(_normalize(vector))
        return vectors


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------
@dataclass
class AzureEmbedder:
    """Azure OpenAI embeddings.

    Credentials come from settings; nothing is hardcoded. Batched because the
    corpus is embedded once and per-call overhead dominates otherwise.
    """

    settings: Any  # AzureOpenAISettings
    batch_size: int = 64
    dimensions: int = 1536
    _client: Any = field(default=None, repr=False)
    _resolved_model: str | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "azure_openai_embedding"

    @property
    def model_version(self) -> str:
        # The deployment name until a call reveals the model behind it. Azure
        # deployments can be re-pointed, so the resolved model is preferred.
        return self._resolved_model or str(self.settings.embedding_deployment or "unknown")

    @property
    def dimension(self) -> int:
        return self.dimensions

    @property
    def is_semantic(self) -> bool:
        return True

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise EmbeddingError(
                "Azure embeddings require the 'generation' extra. "
                'Install it with: pip install -e ".[generation]"'
            ) from exc

        if not self.settings.embedding_deployment:
            raise EmbeddingError(
                "No embedding deployment configured. Set "
                "EVIDENCE_ROUTE_AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env."
            )

        self._client = AzureOpenAI(
            azure_endpoint=str(self.settings.endpoint),
            api_version=self.settings.api_version,
            api_key=(self.settings.api_key.get_secret_value() if self.settings.api_key else None),
        )
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        deployment = str(self.settings.embedding_deployment)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                response = client.embeddings.create(model=deployment, input=batch)
            except Exception as exc:
                raise EmbeddingError(f"Azure embedding call failed: {exc}") from exc
            self._resolved_model = getattr(response, "model", None) or self._resolved_model
            vectors.extend(_normalize(list(item.embedding)) for item in response.data)
        return vectors


# ---------------------------------------------------------------------------
# Local sentence-transformers
# ---------------------------------------------------------------------------
@dataclass
class LocalEmbedder:
    """sentence-transformers embeddings.

    The documented fallback so the benchmark stays reproducible without an Azure
    account. Requires the heavy ``local-models`` extra (torch), which is why it
    is not the default.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _model: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "sentence_transformers"

    @property
    def model_version(self) -> str:
        return self.model_name

    @property
    def dimension(self) -> int:
        model = self._ensure_model()
        return int(model.get_sentence_embedding_dimension())

    @property
    def is_semantic(self) -> bool:
        return True

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise EmbeddingError(
                "Local embeddings require the 'local-models' extra (~3-4 GB, pulls "
                'torch). Install it with: pip install -e ".[local-models]"'
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        encoded = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, vector)) for vector in encoded]
