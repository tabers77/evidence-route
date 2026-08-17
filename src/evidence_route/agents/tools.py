"""The agent's allowlisted tool set (spec section 6, A5).

Five tools, and no more. A larger catalogue does not make a better experiment —
it makes the action harder to interpret, because any effect could come from the
agent's reasoning or from having handed it a better instrument.

    search_collection          find candidate chunks
    read_chunk                 read one chunk in full
    read_neighbouring_chunks   read what surrounds it
    compute                    deterministic arithmetic
    submit_answer              terminate with an evidence-grounded answer

**Arithmetic is parsed, never evaluated.** ``eval`` on model-generated text is
arbitrary code execution: a model that emits ``__import__('os').system(...)``
instead of ``1577/2`` would run it. The expression is walked as an AST with an
explicit node allowlist, so anything that is not arithmetic is rejected rather
than executed. The tool exists at all because asking a language model to divide
two nine-digit numbers is a known failure mode, and section 16 lists incorrect
calculation as its own category.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Any

from evidence_route.documents.models import Chunk
from evidence_route.retrieval.base import Retriever

__all__ = [
    "TOOL_NAMES",
    "ToolBox",
    "ToolCall",
    "ToolResult",
    "safe_arithmetic",
]

TOOL_NAMES = (
    "search_collection",
    "read_chunk",
    "read_neighbouring_chunks",
    "compute",
    "submit_answer",
)

#: Only these AST nodes may appear in an expression. Anything else — a call, an
#: attribute access, a name — is rejected before evaluation.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)

_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

#: Guards against an exponent that would hang the process computing it.
_MAX_EXPONENT = 100


def safe_arithmetic(expression: str) -> float:
    """Evaluate an arithmetic expression without executing arbitrary code."""
    cleaned = expression.replace(",", "").replace("$", "").strip()
    if not cleaned:
        raise ValueError("Empty expression.")
    if len(cleaned) > 500:
        raise ValueError("Expression too long.")

    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Not a valid expression: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"{type(node).__name__} is not permitted. Only arithmetic on "
                f"numbers is allowed — no names, calls or attribute access."
            )

    return float(_evaluate(tree.body))


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numbers are allowed, got {node.value!r}.")
        return float(node.value)

    if isinstance(node, ast.UnaryOp):
        return float(_OPERATORS[type(node.op)](_evaluate(node.operand)))

    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"Exponent {right} exceeds the limit of {_MAX_EXPONENT}.")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("Division by zero.")
        return float(_OPERATORS[type(node.op)](left, right))

    raise ValueError(f"Unsupported expression element: {type(node).__name__}.")


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the agent."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        """Identity used to detect repeated calls.

        Repeating a call wastes a step and is a recognised agent failure mode
        (spec section 16, "repeated search loop"), so it has to be detectable.
        """
        parts = sorted(f"{key}={value!r}" for key, value in self.arguments.items())
        return f"{self.name}({', '.join(parts)})"


@dataclass
class ToolResult:
    """What a tool returned."""

    name: str
    ok: bool
    content: Any = None
    error: str | None = None
    #: Chunks this call surfaced, for the evidence-per-call metric.
    chunk_ids: list[str] = field(default_factory=list)
    terminal: bool = False


@dataclass
class ToolBox:
    """Executes the allowlisted tools against one question's corpus."""

    chunks: list[Chunk]
    retriever: Retriever
    max_search_results: int = 5
    max_chunk_chars: int = 4000

    def __post_init__(self) -> None:
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self._by_ordinal = {(chunk.document_id, chunk.ordinal): chunk for chunk in self.chunks}

    def describe(self) -> str:
        """Tool documentation for the agent's prompt."""
        return (
            "Available tools:\n"
            '  search_collection {"query": "...", "k": 5}\n'
            "      Find passages. Returns chunk ids with short previews.\n"
            '  read_chunk {"chunk_id": "..."}\n'
            "      Read one passage in full.\n"
            '  read_neighbouring_chunks {"chunk_id": "...", "before": 1, "after": 1}\n'
            "      Read the passages surrounding one, for context that spans a boundary.\n"
            '  compute {"expression": "1577 / 2"}\n'
            "      Exact arithmetic. Use this for every calculation rather than\n"
            "      doing it yourself. Numbers only — no variables or functions.\n"
            '  submit_answer {"answer": "...", "citations": [{"chunk_id": "..."}], '
            '"confidence": 0.8}\n'
            "      Finish. Every claim must cite a chunk id you actually read.\n"
            "      To decline, submit_answer with "
            '"abstention_reason": "insufficient_evidence".'
        )

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call, converting any failure into a result.

        An unknown or malformed call returns ``ok=False`` with an explanation
        rather than raising, so the agent can correct itself — and so an invalid
        call is counted in the valid-tool-call rate rather than ending the run.
        """
        if call.name not in TOOL_NAMES:
            return ToolResult(
                name=call.name,
                ok=False,
                error=(f"Unknown tool {call.name!r}. Available: {', '.join(TOOL_NAMES)}."),
            )

        handler = {
            "search_collection": self._search,
            "read_chunk": self._read_chunk,
            "read_neighbouring_chunks": self._read_neighbours,
            "compute": self._compute,
            "submit_answer": self._submit,
        }[call.name]

        try:
            return handler(call.arguments)
        except Exception as exc:
            return ToolResult(name=call.name, ok=False, error=f"{type(exc).__name__}: {exc}")

    # -- individual tools ---------------------------------------------------
    def _search(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolResult("search_collection", ok=False, error="A query is required.")

        k = min(int(arguments.get("k") or self.max_search_results), self.max_search_results)
        hits = self.retriever.search(query, k=max(1, k))
        if not hits:
            return ToolResult(
                "search_collection",
                ok=True,
                content="No passages matched that query. Try different wording.",
            )

        lines = [f"[{hit.chunk_id}] {(hit.text or '')[:200].strip()}..." for hit in hits]
        return ToolResult(
            "search_collection",
            ok=True,
            content="\n".join(lines),
            chunk_ids=[hit.chunk_id for hit in hits],
        )

    def _read_chunk(self, arguments: dict[str, Any]) -> ToolResult:
        chunk_id = str(arguments.get("chunk_id") or "").strip()
        chunk = self._by_id.get(chunk_id)
        if chunk is None:
            return ToolResult(
                "read_chunk",
                ok=False,
                error=f"No chunk {chunk_id!r}. Use search_collection to find valid ids.",
            )
        return ToolResult(
            "read_chunk",
            ok=True,
            content=chunk.text[: self.max_chunk_chars],
            chunk_ids=[chunk.chunk_id],
        )

    def _read_neighbours(self, arguments: dict[str, Any]) -> ToolResult:
        chunk_id = str(arguments.get("chunk_id") or "").strip()
        chunk = self._by_id.get(chunk_id)
        if chunk is None:
            return ToolResult("read_neighbouring_chunks", ok=False, error=f"No chunk {chunk_id!r}.")

        before = max(0, min(int(arguments.get("before") or 1), 3))
        after = max(0, min(int(arguments.get("after") or 1), 3))

        collected: list[Chunk] = []
        for offset in range(-before, after + 1):
            neighbour = self._by_ordinal.get((chunk.document_id, chunk.ordinal + offset))
            if neighbour is not None:
                collected.append(neighbour)

        return ToolResult(
            "read_neighbouring_chunks",
            ok=True,
            content="\n\n".join(
                f"[{c.chunk_id}]\n{c.text[: self.max_chunk_chars]}" for c in collected
            ),
            chunk_ids=[c.chunk_id for c in collected],
        )

    def _compute(self, arguments: dict[str, Any]) -> ToolResult:
        expression = str(arguments.get("expression") or "")
        try:
            value = safe_arithmetic(expression)
        except ValueError as exc:
            return ToolResult("compute", ok=False, error=str(exc))
        return ToolResult("compute", ok=True, content=f"{expression} = {value}")

    def _submit(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult("submit_answer", ok=True, content=arguments, terminal=True)
