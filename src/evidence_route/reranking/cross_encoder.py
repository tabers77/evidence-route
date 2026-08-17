"""Cross-encoder reranking.

One of the two candidates for protocol decision 6. A cross-encoder reads the
query and a passage *together* and scores their relevance directly, which is
strictly more informative than comparing two independently-computed embeddings —
it can notice that a passage mentions the right figure for the wrong year.

Its trade-offs against the LLM reranker:

============  ============================  ==========================
              cross-encoder                 Azure LLM reranker
============  ============================  ==========================
per query     free after download           billed per call
determinism   deterministic                 sampling-dependent
offline       yes                           needs an account
install       ~3-4 GB (torch)               none
stability     weights frozen                deployment can be re-pointed
============  ============================  ==========================

Requires the ``local-models`` extra. The import is lazy so the core package
still installs without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evidence_route.reranking.base import reorder
from evidence_route.storage.records import RetrievedItem

__all__ = ["CrossEncoderReranker"]


@dataclass
class CrossEncoderReranker:
    """sentence-transformers cross-encoder."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    _model: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "cross_encoder"

    @property
    def model_version(self) -> str:
        return self.model_name

    @property
    def is_learned(self) -> bool:
        return True

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "Cross-encoder reranking requires the 'local-models' extra "
                '(~3-4 GB, pulls torch). Install it with: pip install -e ".[local-models]"'
            ) from exc
        self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, items: list[RetrievedItem], k: int = 10) -> list[RetrievedItem]:
        if not items:
            return []
        model = self._ensure_model()
        pairs = [(query, item.text or "") for item in items]
        scores = model.predict(pairs, batch_size=self.batch_size)
        return reorder(items, [float(s) for s in scores], k)
