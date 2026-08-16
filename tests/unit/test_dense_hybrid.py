"""Tests for embeddings, dense retrieval, fusion and the paired comparison."""

from __future__ import annotations

import pytest

from evidence_route.documents.models import Chunk
from evidence_route.evaluation.retrieval_comparison import (
    QuestionResult,
    compare_retrievers,
)
from evidence_route.generation.client import ScriptedProvider
from evidence_route.retrieval.bm25 import BM25Retriever
from evidence_route.retrieval.dense import DenseConfig, DenseRetriever
from evidence_route.retrieval.embeddings import HashingEmbedder
from evidence_route.retrieval.hybrid import (
    HybridConfig,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from evidence_route.storage.records import QuestionRecord, RetrievedItem
from evidence_route.workflows.actions import Action
from evidence_route.workflows.base import WorkflowContext
from evidence_route.workflows.dense_workflow import DenseWorkflow, HybridWorkflow


def _chunk(ordinal: int, text: str, document_id: str = "D") -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}::c{ordinal:04d}",
        document_id=document_id,
        text=text,
        page_numbers=(ordinal + 1,),
        ordinal=ordinal,
        token_estimate=max(1, len(text) // 4),
    )


def _corpus() -> list[Chunk]:
    return [
        _chunk(0, "Total revenues 66,608 for the fiscal year"),
        _chunk(1, "Research and development expense 2,852"),
        _chunk(2, "Operating income margin improved to 23.4%"),
        _chunk(3, "Discussion of supply chain and weather"),
    ]


def _item(chunk_id: str, rank: int, document_id: str = "D") -> RetrievedItem:
    return RetrievedItem(chunk_id=chunk_id, document_id=document_id, rank=rank)


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder()
    assert embedder.embed(["total revenues"]) == embedder.embed(["total revenues"])


def test_hashing_embedder_produces_unit_vectors():
    """Normalization is what lets cosine reduce to a dot product."""
    vector = HashingEmbedder(dimensions=64).embed(["revenue growth"])[0]
    assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-6)


def test_hashing_embedder_respects_its_dimension():
    assert len(HashingEmbedder(dimensions=32).embed(["x"])[0]) == 32


def test_hashing_embedder_declares_itself_non_semantic():
    """The guard that stops a stub producing a fake finding about embeddings."""
    assert HashingEmbedder().is_semantic is False


def test_empty_text_does_not_break_normalization():
    assert HashingEmbedder().embed([""])[0] == [0.0] * 256


def test_model_version_identifies_the_configuration():
    assert (
        HashingEmbedder(dimensions=64).model_version
        != HashingEmbedder(dimensions=128).model_version
    )


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------
def test_dense_retrieves_lexically_overlapping_text():
    retriever = DenseRetriever(embedder=HashingEmbedder()).index(_corpus())
    hits = retriever.search("total revenues", k=2)
    assert hits
    assert hits[0].chunk_id == "D::c0000"


def test_dense_search_is_reproducible():
    retriever = DenseRetriever().index(_corpus())
    assert [h.chunk_id for h in retriever.search("revenues", k=3)] == [
        h.chunk_id for h in retriever.search("revenues", k=3)
    ]


def test_dense_ranks_are_one_based_and_scores_descend():
    hits = DenseRetriever().index(_corpus()).search("revenue expense margin", k=4)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_dense_empty_index_returns_nothing():
    assert DenseRetriever().index([]).search("anything", k=5) == []


def test_dense_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be positive"):
        DenseRetriever().index(_corpus()).search("x", k=0)


def test_similarity_floor_excludes_weak_matches():
    strict = DenseRetriever(config=DenseConfig(min_similarity=0.99)).index(_corpus())
    assert strict.search("entirely unrelated zzz", k=4) == []


def test_index_records_its_embedding_model():
    retriever = DenseRetriever(embedder=HashingEmbedder(dimensions=64)).index(_corpus())
    assert "64d" in retriever.model_version
    assert retriever.stats()["is_semantic"] is False


def test_querying_with_a_different_embedding_model_is_refused():
    """Vectors from different models are not comparable — and the mismatch
    would still return a confident-looking ranking."""
    retriever = DenseRetriever(embedder=HashingEmbedder(dimensions=64)).index(_corpus())
    retriever.embedder = HashingEmbedder(dimensions=128)

    with pytest.raises(ValueError, match="different models are meaningless"):
        retriever.search("revenue", k=2)


def test_misaligned_embedding_batch_is_rejected():
    """Every vector would otherwise attach to the wrong chunk."""

    class ShortEmbedder(HashingEmbedder):
        def embed(self, texts):
            return super().embed(texts)[:-1]

    with pytest.raises(ValueError, match="misaligned batch"):
        DenseRetriever(embedder=ShortEmbedder()).index(_corpus())


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------
def test_fusion_rewards_agreement():
    """A chunk both retrievers rank highly must beat one only either likes."""
    fused = reciprocal_rank_fusion(
        {
            "bm25": [_item("both", 1), _item("lex_only", 2)],
            "dense": [_item("both", 1), _item("sem_only", 2)],
        },
        rrf_k=60,
        k=3,
    )
    assert fused[0].chunk_id == "both"


def test_fusion_surfaces_a_chunk_only_one_retriever_found():
    """The reason fusion is worth its cost."""
    fused = reciprocal_rank_fusion({"bm25": [_item("lex", 1)], "dense": [_item("sem", 1)]}, k=5)
    assert {f.chunk_id for f in fused} == {"lex", "sem"}


def test_fusion_records_component_ranks():
    """Without them a hybrid result is unattributable."""
    fused = reciprocal_rank_fusion({"bm25": [_item("c1", 3)], "dense": [_item("c1", 1)]}, k=1)
    assert fused[0].component_ranks == {"bm25": 3, "dense": 1}


def test_fusion_uses_ranks_not_scores():
    """BM25 scores are unbounded and cosines live in [-1, 1].

    Fusing on scores would let whichever retriever produces larger numbers
    dominate, for reasons unrelated to retrieval quality.
    """
    huge = RetrievedItem(chunk_id="huge", document_id="D", rank=2, score=9999.0)
    small = RetrievedItem(chunk_id="small", document_id="D", rank=1, score=0.01)
    fused = reciprocal_rank_fusion({"a": [small, huge]}, k=2)
    assert fused[0].chunk_id == "small"


def test_rrf_k_flattens_top_ranks():
    """A larger constant reduces how much rank 1 dominates rank 2."""
    rankings = {"a": [_item("first", 1)], "b": [_item("second", 2)]}
    flat = reciprocal_rank_fusion(rankings, rrf_k=1000, k=2)
    peaked = reciprocal_rank_fusion(rankings, rrf_k=0, k=2)

    flat_gap = flat[0].score - flat[1].score
    peaked_gap = peaked[0].score - peaked[1].score
    assert flat_gap < peaked_gap


def test_weights_can_favour_a_component():
    fused = reciprocal_rank_fusion(
        {"bm25": [_item("lex", 1)], "dense": [_item("sem", 1)]},
        weights={"dense": 5.0},
        k=2,
    )
    assert fused[0].chunk_id == "sem"


def test_fusion_is_deterministic_under_ties():
    rankings = {"a": [_item("z", 1), _item("y", 1)]}
    assert [f.chunk_id for f in reciprocal_rank_fusion(rankings, k=2)] == [
        f.chunk_id for f in reciprocal_rank_fusion(rankings, k=2)
    ]


def test_empty_rankings_fuse_to_nothing():
    assert reciprocal_rank_fusion({}, k=5) == []


def test_hybrid_config_validates():
    with pytest.raises(ValueError, match="rrf_k must be non-negative"):
        HybridConfig(rrf_k=-1)
    with pytest.raises(ValueError, match="candidate_k must be positive"):
        HybridConfig(candidate_k=0)


def test_hybrid_retriever_combines_components():
    hybrid = HybridRetriever(
        components={
            "bm25": BM25Retriever().index(_corpus()),
            "dense": DenseRetriever().index(_corpus()),
        }
    )
    hits = hybrid.search("total revenues", k=3)
    assert hits
    assert hits[0].component_ranks


def test_hybrid_stats_document_the_fusion_method():
    """ "Hybrid" without a stated method is not a method."""
    hybrid = HybridRetriever(components={"bm25": BM25Retriever().index(_corpus())})
    assert hybrid.stats()["fusion"] == "reciprocal_rank_fusion"
    assert hybrid.stats()["rrf_k"] == 60


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------
def _context() -> WorkflowContext:
    return WorkflowContext(
        question=QuestionRecord(
            question_id="q1",
            dataset="financebench",
            split="dev",
            question_text="What were total revenues?",
            document_ids=["D"],
            reference_answer="66,608",
        ),
        experiment_id="exp",
        chunks=_corpus(),
        random_seed=1,
    )


def _answer() -> str:
    return '{"answer": "66,608", "citations": [{"chunk_id": "D::c0000"}], "confidence": 0.8}'


def test_dense_workflow_runs():
    outcome = DenseWorkflow(ScriptedProvider(responses=[_answer()])).run(_context())
    assert outcome.action_id is Action.DENSE
    assert outcome.succeeded
    assert outcome.retrieved_items


def test_hybrid_workflow_runs():
    outcome = HybridWorkflow(ScriptedProvider(responses=[_answer()])).run(_context())
    assert outcome.action_id is Action.HYBRID
    assert outcome.succeeded
    assert outcome.retrieved_items


def test_dense_workflow_records_its_embedding_model():
    """Results from different embedding models must not be pooled."""
    outcome = DenseWorkflow(ScriptedProvider(responses=[_answer()])).run(_context())
    assert "hashing" in outcome.model_configuration["embedding_model"]
    assert outcome.model_configuration["embedding_is_semantic"] is False


def test_hybrid_workflow_records_the_fusion_configuration():
    outcome = HybridWorkflow(ScriptedProvider(responses=[_answer()])).run(_context())
    assert outcome.model_configuration["fusion"] == "reciprocal_rank_fusion"
    assert outcome.model_configuration["rrf_k"] == 60


def test_hybrid_results_carry_component_attribution():
    outcome = HybridWorkflow(ScriptedProvider(responses=[_answer()])).run(_context())
    assert any(item.component_ranks for item in outcome.retrieved_items)


def test_workflows_with_no_chunks_still_produce_outcomes():
    abstain = '{"abstention_reason": "insufficient_evidence"}'
    for workflow_cls in (DenseWorkflow, HybridWorkflow):
        provider = ScriptedProvider(responses=[abstain])
        context = WorkflowContext(question=_context().question, experiment_id="exp", chunks=[])
        outcome = workflow_cls(provider).run(context)
        assert outcome.succeeded
        assert outcome.retrieved_items == []


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------
def _results(pattern: dict[str, list[bool]]) -> list[QuestionResult]:
    """Build results from {retriever: [hit per question]}."""
    out = []
    for retriever, hits in pattern.items():
        for i, hit in enumerate(hits):
            out.append(
                QuestionResult(
                    question_id=f"q{i}",
                    retriever=retriever,
                    recall=1.0 if hit else 0.0,
                    hit=hit,
                    n_gold=1,
                )
            )
    return out


def test_identical_retrievers_show_no_disagreement():
    """If they never disagree, routing between them cannot help."""
    report = compare_retrievers(
        _results({"a": [True, False, True], "b": [True, False, True]}), k=10
    )
    disagreement = report.disagreements[0]
    assert disagreement.discordant == 0
    assert disagreement.disagreement_rate == 0.0
    assert disagreement.complementarity == 0.0


def test_complementary_retrievers_show_an_oracle_gain():
    """The case that makes routing worth building."""
    report = compare_retrievers(_results({"a": [True, False], "b": [False, True]}), k=10)
    disagreement = report.disagreements[0]
    assert disagreement.a_only == 1
    assert disagreement.b_only == 1
    assert disagreement.disagreement_rate == 1.0
    assert disagreement.complementarity == pytest.approx(0.5)
    assert report.oracle_hit_rate() == 1.0


def test_oracle_cannot_be_beaten_by_any_single_retriever():
    report = compare_retrievers(
        _results({"a": [True, False, False], "b": [False, True, False]}), k=10
    )
    assert report.oracle_hit_rate() >= (report.hit_rate("a") or 0)
    assert report.oracle_hit_rate() >= (report.hit_rate("b") or 0)


def test_unmeasurable_questions_are_excluded_from_pairing():
    """An unbalanced pairing is what paired analysis exists to avoid."""
    results = _results({"a": [True, True], "b": [True, True]})
    results.append(
        QuestionResult(
            question_id="q9", retriever="a", recall=0.0, hit=False, n_gold=0, measurable=False
        )
    )
    report = compare_retrievers(results, k=10)
    assert report.disagreements[0].total == 2


def test_non_semantic_embedder_triggers_a_warning():
    """Otherwise the offline path yields a fake finding about embeddings."""
    report = compare_retrievers(
        _results({"bm25": [True], "dense": [True]}),
        k=10,
        non_semantic_retrievers=["dense"],
    )
    assert report.warnings
    assert "artifact of the stub" in report.warnings[0]


def test_report_serialises():
    report = compare_retrievers(_results({"a": [True, False], "b": [False, True]}), k=10)
    payload = report.to_dict()
    assert payload["oracle_hit_rate"] == 1.0
    assert payload["disagreements"][0]["a_only"] == 1


def test_summary_is_readable():
    report = compare_retrievers(_results({"a": [True, False], "b": [False, True]}), k=5)
    summary = report.summary()
    assert "ORACLE" in summary
    assert "disagreement" in summary
