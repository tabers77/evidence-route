"""The bounded agent loop (spec section 6, A5).

Four independent limits, each of which terminates the run: steps, tool calls,
total tokens and wall-clock time. Independent because they fail differently — an
agent stuck alternating between two searches burns steps, one summarising a long
document burns tokens, one waiting on a slow provider burns time. A single limit
would leave the other three unbounded.

**Hitting a limit is an explicit error outcome, never a silent truncation.** An
agent that quietly returns whatever it had when it ran out looks like an agent
that answered, and its failures would be scored as ordinary wrong answers rather
than as the resource exhaustion they are. The termination reason is recorded and
carried into the outcome.

The loop is a minimal custom implementation rather than a framework. Comparing
agent frameworks is not this project's research question (spec section 17), and
a framework would put its own retry, parsing and prompting behaviour between the
measurement and the thing being measured.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from evidence_route.agents.tools import ToolBox, ToolCall, ToolResult
from evidence_route.generation.client import (
    GenerationProvider,
    GenerationRequest,
    ProviderError,
)
from evidence_route.generation.schema import extract_json_object

__all__ = [
    "AGENT_PROMPT_V1",
    "AgentLimits",
    "AgentRun",
    "AgentStep",
    "BoundedAgent",
    "TerminationReason",
]

#: ``__TOOLS__`` is substituted by plain replacement, not ``str.format``. The
#: prompt is full of literal JSON braces, and format() would try to interpret
#: every one of them as a field.
AGENT_PROMPT_V1 = """You are a financial analyst answering a question about company \
filings. You work by calling tools, one at a time.

Each turn, respond with a single JSON object and nothing else:

{"tool": "<tool name>", "arguments": {...}, "thought": "<why this step>"}

__TOOLS__

Rules:
- Search before you answer. Do not answer from memory.
- Read a passage in full before citing it; previews are truncated.
- Use compute for every calculation. Do not do arithmetic yourself.
- Do not repeat a call you have already made — you have its result.
- Finish with submit_answer. Every claim must cite a chunk id you read.
- If the filings do not contain the answer, submit_answer with an \
abstention_reason. Declining is correct when the evidence is absent; guessing \
is not."""


class TerminationReason(str):
    """Why a run ended. A plain string subclass so it serialises directly."""

    SUBMITTED = "submitted"
    STEP_LIMIT = "step_limit_exceeded"
    TOOL_CALL_LIMIT = "tool_call_limit_exceeded"
    TOKEN_LIMIT = "token_limit_exceeded"
    TIME_LIMIT = "time_limit_exceeded"
    INVALID_TERMINATION = "invalid_termination"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True)
class AgentLimits:
    """Execution bounds. Part of the versioned experiment configuration."""

    max_steps: int = 8
    max_tool_calls: int = 12
    max_total_tokens: int = 20_000
    max_wall_clock_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive.")
        if self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive.")
        if self.max_wall_clock_seconds <= 0:
            raise ValueError("max_wall_clock_seconds must be positive.")


@dataclass
class AgentStep:
    """One turn: what the agent asked for and what came back."""

    index: int
    call: ToolCall | None
    result: ToolResult | None
    thought: str | None = None
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    parse_error: str | None = None
    is_repeat: bool = False

    @property
    def valid(self) -> bool:
        """Whether this turn produced a well-formed, successful tool call."""
        return (
            self.parse_error is None
            and self.call is not None
            and self.result is not None
            and self.result.ok
        )


@dataclass
class AgentRun:
    """The complete trace of one agent execution."""

    steps: list[AgentStep] = field(default_factory=list)
    termination_reason: str = TerminationReason.INVALID_TERMINATION
    submission: dict | None = None
    elapsed_seconds: float = 0.0
    error_detail: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.termination_reason == TerminationReason.SUBMITTED

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.call is not None)

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def valid_tool_call_rate(self) -> float:
        """Share of turns that produced a usable tool call.

        A low rate means the agent is fighting the tool contract rather than
        the question — a prompt problem, not a reasoning one.
        """
        attempted = [s for s in self.steps if s.call is not None or s.parse_error]
        if not attempted:
            return 0.0
        return sum(1 for s in attempted if s.valid) / len(attempted)

    @property
    def repeated_call_rate(self) -> float:
        """Share of tool calls that repeated an earlier one."""
        calls = [s for s in self.steps if s.call is not None]
        if not calls:
            return 0.0
        return sum(1 for s in calls if s.is_repeat) / len(calls)

    @property
    def chunks_seen(self) -> list[str]:
        """Every chunk the agent actually read, in order of first sight."""
        seen: list[str] = []
        for step in self.steps:
            if step.result is None:
                continue
            for chunk_id in step.result.chunk_ids:
                if chunk_id not in seen:
                    seen.append(chunk_id)
        return seen

    @property
    def evidence_per_call(self) -> float:
        """New chunks surfaced per tool call. Measures search efficiency."""
        calls = self.n_tool_calls
        return len(self.chunks_seen) / calls if calls else 0.0


@dataclass
class BoundedAgent:
    """Runs the tool loop under hard limits."""

    provider: GenerationProvider
    limits: AgentLimits = field(default_factory=AgentLimits)
    deployment_ref: str = "chat"
    prompt_version: str = "agentic_v1"
    temperature: float = 0.0
    max_output_tokens: int = 512

    def run(self, question: str, toolbox: ToolBox, *, seed: int | None = None) -> AgentRun:
        """Execute until submission or a limit.

        Every exit path sets a termination reason, so a run can never end in an
        ambiguous state that the outcome record would have to guess about.
        """
        run = AgentRun()
        started = time.perf_counter()
        system = AGENT_PROMPT_V1.replace("__TOOLS__", toolbox.describe())
        transcript: list[str] = [f"Question: {question}"]
        seen_signatures: set[str] = set()

        for index in range(self.limits.max_steps):
            if time.perf_counter() - started > self.limits.max_wall_clock_seconds:
                run.termination_reason = TerminationReason.TIME_LIMIT
                break
            if run.total_tokens >= self.limits.max_total_tokens:
                run.termination_reason = TerminationReason.TOKEN_LIMIT
                break
            if run.n_tool_calls >= self.limits.max_tool_calls:
                run.termination_reason = TerminationReason.TOOL_CALL_LIMIT
                break

            request = GenerationRequest(
                system=system,
                user="\n\n".join(transcript),
                deployment_ref=self.deployment_ref,
                prompt_version=self.prompt_version,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                seed=seed,
            )

            try:
                response = self.provider.complete(request)
            except ProviderError as exc:
                run.termination_reason = TerminationReason.PROVIDER_ERROR
                run.error_detail = str(exc)
                break

            step = AgentStep(
                index=index,
                call=None,
                result=None,
                raw_response=response.text,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

            try:
                payload = extract_json_object(response.text)
                tool_name = str(payload["tool"])
                arguments = payload.get("arguments") or {}
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                step.thought = payload.get("thought")
                step.call = ToolCall(name=tool_name, arguments=arguments)
            except (ValueError, KeyError, TypeError) as exc:
                step.parse_error = f"could not read a tool call: {exc}"
                run.steps.append(step)
                # Told, not silently retried — the agent can correct itself, and
                # the failed turn still counts against its limits.
                transcript.append(
                    f"Your response could not be parsed ({exc}). Reply with a single "
                    f"JSON object containing 'tool' and 'arguments'."
                )
                continue

            signature = step.call.signature()
            step.is_repeat = signature in seen_signatures
            seen_signatures.add(signature)

            step.result = toolbox.execute(step.call)
            run.steps.append(step)

            if step.result.terminal:
                run.submission = (
                    step.result.content if isinstance(step.result.content, dict) else {}
                )
                run.termination_reason = TerminationReason.SUBMITTED
                break

            transcript.append(
                json.dumps({"tool": step.call.name, "arguments": step.call.arguments})
            )
            observation = step.result.content if step.result.ok else f"ERROR: {step.result.error}"
            transcript.append(f"Observation: {observation}")
        else:
            # The for-loop finished without submitting.
            run.termination_reason = TerminationReason.STEP_LIMIT

        run.elapsed_seconds = time.perf_counter() - started
        return run
