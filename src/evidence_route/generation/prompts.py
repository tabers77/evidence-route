"""Versioned prompt templates.

A prompt is part of the experimental configuration, not an implementation
detail. Changing wording changes results, so every template carries a version
that is recorded in the outcome record and the run manifest (spec section 25).
Editing a template in place would silently make old and new results
incomparable; the rule is that a changed prompt gets a new version.

Two templates for the MVP:

``DIRECT_V1``
    Action A0. No evidence at all. Its job is to measure what the model answers
    from parametric memory — including how often it produces a confident,
    plausible answer with nothing supporting it. A high score here is a warning
    about pretraining contamination, not a success.

``GROUNDED_V1``
    Actions A1-A4. Evidence is supplied with chunk identifiers, and the model is
    required to cite them. Citations are what make evidence support and citation
    precision measurable at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_route.storage.records import RetrievedItem

__all__ = [
    "DIRECT_V1",
    "GROUNDED_V1",
    "PromptTemplate",
    "format_evidence",
    "get_template",
]

#: The response contract, repeated in every template. Kept as one constant so
#: the schema the model is asked for cannot drift between prompts.
_RESPONSE_FORMAT = """Respond with a single JSON object and nothing else:

{
  "answer": "<your answer, or null if you cannot answer>",
  "citations": [{"chunk_id": "<id>", "quoted_text": "<the exact supporting text>"}],
  "confidence": <number between 0 and 1>,
  "abstained": <true or false>,
  "abstention_reason": "<reason code, or null>",
  "reasoning": "<brief explanation>"
}

Valid abstention_reason values:
  insufficient_evidence     nothing provided supports an answer
  retrieval_disagreement    the evidence points in conflicting directions
  ambiguous_question        the question admits several different answers
  conflicting_documents     the documents contradict each other
  calculation_unverified    a figure could not be verified from the evidence
  policy_low_confidence     confidence is too low to answer responsibly"""


@dataclass(frozen=True)
class PromptTemplate:
    """A versioned system/user prompt pair."""

    version: str
    system: str
    user_template: str
    description: str = ""

    def render_user(self, **kwargs: object) -> str:
        """Fill the user template."""
        return self.user_template.format(**kwargs)


def format_evidence(items: list[RetrievedItem]) -> str:
    """Render retrieved chunks with the identifiers the model must cite.

    Identifiers are shown explicitly because a citation is only checkable if it
    names something the workflow actually retrieved — see
    ``StructuredAnswer.validate_citations``. Without visible ids the model would
    invent references and every citation would be unverifiable.
    """
    if not items:
        return "(no evidence was retrieved)"

    blocks = []
    for item in items:
        text = (item.text or "").strip()
        blocks.append(f"[{item.chunk_id}]\n{text}")
    return "\n\n".join(blocks)


DIRECT_V1 = PromptTemplate(
    version="direct_v1",
    description="Action A0: answer with no retrieval, from parametric memory only.",
    system=(
        "You are a financial analyst answering questions about company filings.\n\n"
        "You have NO documents available. Answer only from your own knowledge.\n\n"
        "Rules:\n"
        "- If you do not know the answer, abstain. Do not guess.\n"
        "- Your confidence must reflect that you have no source to check against.\n"
        "- Leave citations empty: you have no evidence to cite.\n"
        "- Give figures in the units the question asks for.\n\n" + _RESPONSE_FORMAT
    ),
    user_template="Question: {question}",
)


GROUNDED_V1 = PromptTemplate(
    version="grounded_v1",
    description="Actions A1-A4: answer strictly from the retrieved evidence.",
    system=(
        "You are a financial analyst answering questions about company filings.\n\n"
        "Answer ONLY from the evidence provided. Do not use outside knowledge, "
        "even if you believe you know the answer.\n\n"
        "Rules:\n"
        "- Every claim must be supported by a citation to a provided chunk id.\n"
        "- Quote the exact supporting text in each citation.\n"
        "- Cite only chunk ids that appear in the evidence below. Never invent one.\n"
        "- If the evidence does not contain the answer, abstain with "
        "insufficient_evidence. An abstention is a correct response when the "
        "evidence is absent; a guess is not.\n"
        "- For calculations, show the figures you used and cite each one. If a "
        "figure you need is missing, abstain with calculation_unverified.\n"
        "- Give figures in the units the question asks for.\n\n" + _RESPONSE_FORMAT
    ),
    user_template="Evidence:\n\n{evidence}\n\n---\n\nQuestion: {question}",
)


_TEMPLATES: dict[str, PromptTemplate] = {
    DIRECT_V1.version: DIRECT_V1,
    GROUNDED_V1.version: GROUNDED_V1,
}


def get_template(version: str) -> PromptTemplate:
    """Look up a template by version.

    Raises on an unknown version rather than falling back to a default: a run
    silently using a different prompt from the one its config names would
    produce results attributed to the wrong prompt.
    """
    try:
        return _TEMPLATES[version]
    except KeyError:
        known = ", ".join(sorted(_TEMPLATES))
        raise KeyError(f"Unknown prompt version {version!r}. Known: {known}") from None
