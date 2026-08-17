"""Tests for reranking (A4) and the bounded agent (A5).

The agent tests lean adversarial on purpose: "agentic execution cannot loop
indefinitely" is week 5's exit criterion, and a bound that is never attacked is
not known to hold.
"""

from __future__ import annotations

import json

import pytest

from evidence_route.agents.loop import (
    AgentLimits,
    BoundedAgent,
    TerminationReason,
)
from evidence_route.agents.tools import ToolBox, ToolCall, safe_arithmetic
from evidence_route.documents.models import Chunk
from evidence_route.generation.client import (
    GenerationResponse,
    ProviderError,
    ScriptedProvider,
)
from evidence_route.reranking.base import LexicalOverlapReranker, reorder
from evidence_route.reranking.llm import LLMReranker
from evidence_route.retrieval.bm25 import BM25Retriever
from evidence_route.storage.records import QuestionRecord, RetrievedItem
from evidence_route.workflows.actions import Action
from evidence_route.workflows.agentic import AgenticWorkflow
from evidence_route.workflows.base import WorkflowContext
from evidence_route.workflows.rerank_workflow import RerankWorkflow


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
        _chunk(0, "Total revenues 66,608 for the fiscal year ended December"),
        _chunk(1, "Research and development expense 2,852 in the period"),
        _chunk(2, "Operating income margin improved to 23.4% year over year"),
        _chunk(3, "Discussion of supply chain disruption and weather effects"),
    ]


def _items(n: int = 4) -> list[RetrievedItem]:
    corpus = _corpus()
    return [
        RetrievedItem(chunk_id=c.chunk_id, document_id=c.document_id, rank=i + 1, text=c.text)
        for i, c in enumerate(corpus[:n])
    ]


# ---------------------------------------------------------------------------
# Safe arithmetic — the tool that must never execute arbitrary code
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expression,expected",
    [
        ("1577 / 2", 788.5),
        ("66,608 - 62,286", 4322.0),
        ("$1,577.00 * 2", 3154.0),
        ("(100 - 40) / 100", 0.6),
        ("2 ** 10", 1024.0),
        ("-5 + 3", -2.0),
    ],
)
def test_arithmetic_evaluates(expression: str, expected: float):
    assert safe_arithmetic(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__",
        "print(1)",
        "x + 1",
        "lambda: 1",
        "[1,2,3]",
    ],
)
def test_arithmetic_refuses_anything_that_is_not_arithmetic(expression: str):
    """eval on model output is arbitrary code execution.

    The expression is parsed and walked against a node allowlist, so these are
    rejected rather than run.
    """
    with pytest.raises(ValueError):
        safe_arithmetic(expression)


def test_arithmetic_rejects_division_by_zero():
    with pytest.raises(ValueError, match="Division by zero"):
        safe_arithmetic("1 / 0")


def test_arithmetic_rejects_a_hanging_exponent():
    """9**9**9 would occupy the process indefinitely."""
    with pytest.raises(ValueError, match="Exponent"):
        safe_arithmetic("9 ** 99999")


def test_arithmetic_rejects_empty_and_oversized_input():
    with pytest.raises(ValueError, match="Empty expression"):
        safe_arithmetic("  ")
    with pytest.raises(ValueError, match="too long"):
        safe_arithmetic("1+" * 300 + "1")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def _toolbox() -> ToolBox:
    chunks = _corpus()
    return ToolBox(chunks=chunks, retriever=BM25Retriever().index(chunks))


def test_search_returns_chunk_ids():
    result = _toolbox().execute(ToolCall("search_collection", {"query": "total revenues"}))
    assert result.ok
    assert result.chunk_ids


def test_search_without_a_query_fails_gracefully():
    result = _toolbox().execute(ToolCall("search_collection", {}))
    assert not result.ok
    assert "query is required" in (result.error or "")


def test_read_chunk_returns_full_text():
    result = _toolbox().execute(ToolCall("read_chunk", {"chunk_id": "D::c0000"}))
    assert result.ok
    assert "66,608" in result.content


def test_read_unknown_chunk_explains_how_to_recover():
    result = _toolbox().execute(ToolCall("read_chunk", {"chunk_id": "nope"}))
    assert not result.ok
    assert "search_collection" in (result.error or "")


def test_read_neighbours_spans_a_boundary():
    result = _toolbox().execute(
        ToolCall("read_neighbouring_chunks", {"chunk_id": "D::c0001", "before": 1, "after": 1})
    )
    assert result.ok
    assert set(result.chunk_ids) == {"D::c0000", "D::c0001", "D::c0002"}


def test_unknown_tool_is_reported_not_raised():
    """So an invalid call counts against the valid-call rate rather than
    ending the run."""
    result = _toolbox().execute(ToolCall("delete_everything", {}))
    assert not result.ok
    assert "Unknown tool" in (result.error or "")


def test_submit_answer_is_terminal():
    result = _toolbox().execute(ToolCall("submit_answer", {"answer": "x"}))
    assert result.terminal


def test_call_signature_detects_repeats():
    a = ToolCall("search_collection", {"query": "revenue", "k": 5})
    b = ToolCall("search_collection", {"k": 5, "query": "revenue"})
    assert a.signature() == b.signature()


# ---------------------------------------------------------------------------
# The bounded loop — the exit criterion
# ---------------------------------------------------------------------------
class _LoopingProvider:
    """Never submits. Always searches the same thing."""

    name = "looping"

    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return GenerationResponse(
            text=json.dumps({"tool": "search_collection", "arguments": {"query": "revenue"}}),
            input_tokens=100,
            output_tokens=20,
        )


def test_an_agent_that_never_submits_is_stopped():
    """The week 5 exit criterion, attacked directly."""
    provider = _LoopingProvider()
    run = BoundedAgent(provider, limits=AgentLimits(max_steps=5)).run(
        "What was revenue?", _toolbox()
    )

    assert not run.succeeded
    assert run.termination_reason == TerminationReason.STEP_LIMIT
    assert provider.calls == 5


def test_tool_call_limit_stops_the_loop():
    run = BoundedAgent(_LoopingProvider(), limits=AgentLimits(max_steps=100, max_tool_calls=3)).run(
        "q", _toolbox()
    )
    assert run.termination_reason == TerminationReason.TOOL_CALL_LIMIT
    assert run.n_tool_calls <= 3


def test_token_limit_stops_the_loop():
    run = BoundedAgent(
        _LoopingProvider(),
        limits=AgentLimits(max_steps=100, max_tool_calls=100, max_total_tokens=250),
    ).run("q", _toolbox())
    assert run.termination_reason == TerminationReason.TOKEN_LIMIT


def test_limits_are_independent():
    """Each fails differently, so one limit cannot cover the others."""
    for limits, expected in [
        (
            AgentLimits(max_steps=2, max_tool_calls=99, max_total_tokens=10**6),
            TerminationReason.STEP_LIMIT,
        ),
        (
            AgentLimits(max_steps=99, max_tool_calls=2, max_total_tokens=10**6),
            TerminationReason.TOOL_CALL_LIMIT,
        ),
        (
            AgentLimits(max_steps=99, max_tool_calls=99, max_total_tokens=200),
            TerminationReason.TOKEN_LIMIT,
        ),
    ]:
        run = BoundedAgent(_LoopingProvider(), limits=limits).run("q", _toolbox())
        assert run.termination_reason == expected


def test_limits_must_be_positive():
    for kwargs in [
        {"max_steps": 0},
        {"max_tool_calls": 0},
        {"max_total_tokens": 0},
        {"max_wall_clock_seconds": 0},
    ]:
        with pytest.raises(ValueError, match="must be positive"):
            AgentLimits(**kwargs)


def test_provider_failure_terminates_cleanly():
    class Failing:
        name = "failing"

        def complete(self, request):
            raise ProviderError("rate limited")

    run = BoundedAgent(Failing()).run("q", _toolbox())
    assert run.termination_reason == TerminationReason.PROVIDER_ERROR
    assert "rate limited" in (run.error_detail or "")


def test_unparseable_turn_is_corrected_not_fatal():
    """The agent is told what went wrong and the turn still counts."""
    provider = ScriptedProvider(
        responses=[
            "I will search now.",
            json.dumps({"tool": "submit_answer", "arguments": {"answer": "42"}}),
        ]
    )
    run = BoundedAgent(provider, limits=AgentLimits(max_steps=4)).run("q", _toolbox())

    assert run.succeeded
    assert run.steps[0].parse_error is not None
    assert run.valid_tool_call_rate < 1.0


def test_successful_submission_is_captured():
    provider = ScriptedProvider(
        responses=[
            json.dumps({"tool": "search_collection", "arguments": {"query": "revenue"}}),
            json.dumps(
                {
                    "tool": "submit_answer",
                    "arguments": {
                        "answer": "66,608",
                        "citations": [{"chunk_id": "D::c0000"}],
                        "confidence": 0.9,
                    },
                }
            ),
        ]
    )
    run = BoundedAgent(provider).run("What were revenues?", _toolbox())

    assert run.succeeded
    assert run.submission["answer"] == "66,608"
    assert run.n_tool_calls == 2


def test_repeated_calls_are_counted():
    """A repeated search wastes a step and is a known agent failure mode."""
    run = BoundedAgent(_LoopingProvider(), limits=AgentLimits(max_steps=4)).run("q", _toolbox())
    assert run.repeated_call_rate > 0.0


def test_evidence_per_call_is_measured():
    provider = ScriptedProvider(
        responses=[
            # "revenues", not "revenue" — the tokenizer does not stem, so the
            # singular would not match. See test_tokenizer_does_not_stem.
            json.dumps({"tool": "search_collection", "arguments": {"query": "revenues"}}),
            json.dumps({"tool": "submit_answer", "arguments": {"answer": "x"}}),
        ]
    )
    run = BoundedAgent(provider).run("q", _toolbox())
    assert run.chunks_seen
    assert run.evidence_per_call > 0


# ---------------------------------------------------------------------------
# A5 workflow
# ---------------------------------------------------------------------------
def _context(chunks=None) -> WorkflowContext:
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
        chunks=_corpus() if chunks is None else chunks,
        random_seed=7,
    )


def test_agentic_workflow_produces_the_standard_outcome_schema():
    """The week 5 exit criterion: every action shares one outcome shape."""
    provider = ScriptedProvider(
        responses=[
            json.dumps({"tool": "search_collection", "arguments": {"query": "revenues"}}),
            json.dumps(
                {
                    "tool": "submit_answer",
                    "arguments": {
                        "answer": "66,608",
                        "citations": [{"chunk_id": "D::c0000"}],
                        "confidence": 0.9,
                    },
                }
            ),
        ]
    )
    outcome = AgenticWorkflow(provider).run(_context())

    assert outcome.action_id is Action.AGENTIC
    assert outcome.succeeded
    assert outcome.answer == "66,608"
    assert outcome.citations
    assert outcome.retrieved_items


def test_limit_breach_becomes_an_explicit_error_outcome():
    """Not a silent truncation scored as an ordinary wrong answer."""
    outcome = AgenticWorkflow(_LoopingProvider(), limits=AgentLimits(max_steps=3)).run(_context())

    assert not outcome.succeeded
    assert outcome.error_status == "step_limit_exceeded"
    assert outcome.metadata["termination_reason"] == TerminationReason.STEP_LIMIT


def test_agent_metrics_are_recorded():
    """They cannot be recovered from the outcome record alone."""
    outcome = AgenticWorkflow(_LoopingProvider(), limits=AgentLimits(max_steps=3)).run(_context())
    for key in (
        "n_steps",
        "n_tool_calls",
        "valid_tool_call_rate",
        "repeated_call_rate",
        "evidence_per_call",
        "tools_used",
        "limits",
    ):
        assert key in outcome.metadata


def test_evidence_is_what_the_agent_actually_read():
    """Scoring it against evidence it never saw would make citation validation
    meaningless."""
    provider = ScriptedProvider(
        responses=[
            json.dumps({"tool": "read_chunk", "arguments": {"chunk_id": "D::c0002"}}),
            json.dumps(
                {
                    "tool": "submit_answer",
                    "arguments": {"answer": "23.4%", "citations": [{"chunk_id": "D::c0002"}]},
                }
            ),
        ]
    )
    outcome = AgenticWorkflow(provider).run(_context())
    assert [i.chunk_id for i in outcome.retrieved_items] == ["D::c0002"]


def test_citing_an_unread_chunk_is_flagged():
    provider = ScriptedProvider(
        responses=[
            json.dumps({"tool": "read_chunk", "arguments": {"chunk_id": "D::c0000"}}),
            json.dumps(
                {
                    "tool": "submit_answer",
                    "arguments": {"answer": "x", "citations": [{"chunk_id": "D::c0003"}]},
                }
            ),
        ]
    )
    outcome = AgenticWorkflow(provider).run(_context())
    assert outcome.metadata["n_hallucinated_citations"] == 1


def test_agent_can_abstain():
    provider = ScriptedProvider(
        responses=[
            json.dumps(
                {
                    "tool": "submit_answer",
                    "arguments": {"abstention_reason": "insufficient_evidence"},
                }
            )
        ]
    )
    outcome = AgenticWorkflow(provider).run(_context())
    assert outcome.succeeded
    assert outcome.abstained


def test_no_chunks_yields_an_abstention_not_a_crash():
    outcome = AgenticWorkflow(ScriptedProvider()).run(_context(chunks=[]))
    assert outcome.abstained


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
def test_reorder_preserves_the_pre_rerank_position():
    """Without it, "did reranking move anything" is unanswerable."""
    items = _items()
    reranked = reorder(items, [0.1, 0.9, 0.5, 0.2], k=4)

    assert reranked[0].chunk_id == "D::c0001"
    assert reranked[0].component_ranks["retrieval"] == 2


def test_reorder_ties_keep_retrieval_order():
    """Arbitrary shuffling would look like reranker effect while being noise."""
    items = _items()
    reranked = reorder(items, [1.0, 1.0, 1.0, 1.0], k=4)
    assert [r.chunk_id for r in reranked] == [i.chunk_id for i in items]


def test_reorder_rejects_a_misaligned_score_batch():
    with pytest.raises(ValueError, match="misaligned batch"):
        reorder(_items(), [1.0, 2.0], k=4)


def test_lexical_reranker_declares_itself_unlearned():
    """A stub cannot support a conclusion about whether reranking helps."""
    assert LexicalOverlapReranker().is_learned is False


def test_lexical_reranker_promotes_overlap():
    reranked = LexicalOverlapReranker().rerank("operating income margin", _items(), k=4)
    assert reranked[0].chunk_id == "D::c0002"


def test_llm_reranker_applies_returned_scores():
    provider = ScriptedProvider(
        responses=[json.dumps({"scores": [{"id": 1, "score": 1}, {"id": 3, "score": 10}]})]
    )
    reranked = LLMReranker(provider).rerank("q", _items(), k=4)
    assert reranked[0].chunk_id == "D::c0002"


def test_llm_reranker_falls_back_to_retrieval_order_on_failure():
    """Losing an expensive retrieval result over a formatting error would be
    worse than an unreranked answer."""
    provider = ScriptedProvider(responses=["not json at all"])
    reranker = LLMReranker(provider)
    reranked = reranker.rerank("q", _items(), k=4)

    assert [r.chunk_id for r in reranked] == [i.chunk_id for i in _items()]
    assert reranker.last_fallback_reason is not None


def test_llm_reranker_defaults_omitted_passages_to_neutral():
    """Silence is not evidence of irrelevance; a truncated response would
    otherwise discard good candidates."""
    provider = ScriptedProvider(responses=[json.dumps({"scores": [{"id": 1, "score": 10}]})])
    reranked = LLMReranker(provider).rerank("q", _items(), k=4)
    assert len(reranked) == 4
    assert reranked[0].chunk_id == "D::c0000"


def test_llm_reranker_is_listwise():
    """Pointwise would cost one call per candidate and settle decision 6 on
    billing rather than retrieval quality."""
    provider = ScriptedProvider(responses=[json.dumps({"scores": [{"id": 1, "score": 5}]})])
    LLMReranker(provider).rerank("q", _items(), k=4)
    assert len(provider.requests) == 1


# ---------------------------------------------------------------------------
# A4 workflow
# ---------------------------------------------------------------------------
def test_rerank_workflow_runs():
    provider = ScriptedProvider(
        responses=['{"answer": "66,608", "citations": [{"chunk_id": "D::c0000"}]}']
    )
    outcome = RerankWorkflow(provider).run(_context())

    assert outcome.action_id is Action.HYBRID_RERANK
    assert outcome.succeeded
    assert outcome.retrieved_items


def test_rerank_latency_is_reported_separately():
    """So A3-vs-A4 answers what reranking cost, not just that A4 is slower."""
    provider = ScriptedProvider(responses=['{"answer": "x"}'])
    outcome = RerankWorkflow(provider).run(_context())

    assert outcome.latency_breakdown is not None
    assert outcome.latency_breakdown.reranking_ms > 0


def test_rerank_workflow_records_its_reranker():
    provider = ScriptedProvider(responses=['{"answer": "x"}'])
    outcome = RerankWorkflow(provider).run(_context())

    config = outcome.model_configuration
    assert config["reranker"] == "lexical_overlap"
    assert config["reranker_is_learned"] is False
    assert config["candidate_k"] == 30


def test_rerank_results_keep_their_retrieval_rank():
    provider = ScriptedProvider(responses=['{"answer": "x"}'])
    outcome = RerankWorkflow(provider).run(_context())
    assert all("retrieval" in i.component_ranks for i in outcome.retrieved_items)
