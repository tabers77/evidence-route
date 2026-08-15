"""The bounded multi-step agentic workflow (action A5).

A minimal custom loop, not a framework comparison — comparing agent frameworks
is not this project's research question. The agent exists to answer one thing:
does multi-step decomposition and evidence synthesis improve complex questions
enough to justify its cost?

Two constraints are structural rather than optional:

*Bounded execution.* Step, token and wall-clock limits are enforced, and hitting
one produces an explicit ``step_limit_exceeded`` outcome rather than a silent
truncation. An agent that can loop indefinitely can consume the project's entire
budget on a single question.

*Small allowlisted tool set.* Search the collection, read a chunk, fetch
neighbouring chunks, do deterministic arithmetic, submit an evidence-grounded
answer. More tools do not make a better experiment; they make the action harder
to interpret.

Every tool call and observation is captured in the trace so agent-specific
metrics (valid tool-call rate, repeated calls, evidence discovered per call,
invalid termination) can be computed after the fact.
"""
