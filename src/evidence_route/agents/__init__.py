"""The bounded multi-step agentic workflow (action A5).

A minimal custom loop, not a framework comparison — comparing agent frameworks
is not this project's research question. The agent exists to answer one thing:
does multi-step decomposition and evidence synthesis improve complex questions
enough to justify its cost?

Two constraints are structural rather than optional:

*Bounded execution.* Four independent limits — steps, tool calls, tokens and
wall-clock time — because they fail differently. An agent looping between two
searches burns steps; one summarising a long document burns tokens. Hitting any
of them produces an explicit error outcome rather than a silent truncation, so
resource exhaustion stays visible in the failure taxonomy instead of being
scored as an ordinary wrong answer.

*A small allowlisted tool set.* Search, read, read neighbours, compute exact
arithmetic, submit. More tools do not make a better experiment; they make the
action harder to interpret, because any effect could come from the reasoning or
from having handed the agent a better instrument.

Every tool call and observation is captured, so the agent-specific metrics of
section 11.6 — valid tool-call rate, repeated calls, evidence per call, invalid
termination — are computable after the fact.
"""

from evidence_route.agents.loop import (
    AGENT_PROMPT_V1,
    AgentLimits,
    AgentRun,
    AgentStep,
    BoundedAgent,
    TerminationReason,
)
from evidence_route.agents.tools import (
    TOOL_NAMES,
    ToolBox,
    ToolCall,
    ToolResult,
    safe_arithmetic,
)

__all__ = [
    "AGENT_PROMPT_V1",
    "TOOL_NAMES",
    "AgentLimits",
    "AgentRun",
    "AgentStep",
    "BoundedAgent",
    "TerminationReason",
    "ToolBox",
    "ToolCall",
    "ToolResult",
    "safe_arithmetic",
]
