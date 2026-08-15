"""Evidence-grounded answer generation via Azure OpenAI.

Responses are structured, not free text: every generation returns an answer, its
citations, a confidence value and an optional abstention with a reason code.
Free-text answers would make citation precision and abstention unmeasurable.

Provider credentials come exclusively from the environment through
``evidence_route.config.Settings``. No endpoint, deployment name or key appears
in this package.

Modules:
    ``schema``    the structured answer contract (implemented)
    ``client``    Azure OpenAI client construction (planned)
    ``prompts``   versioned prompt templates; the version is recorded per run
    ``cache``     content-addressed response cache (spec section 26)
"""

from evidence_route.generation.schema import (
    AnswerCitation,
    StructuredAnswer,
    extract_json_object,
    parse_structured_answer,
)

__all__ = [
    "AnswerCitation",
    "StructuredAnswer",
    "extract_json_object",
    "parse_structured_answer",
]
