"""Evidence-grounded answer generation via Azure OpenAI.

Responses are structured, not free text: every generation returns an answer, its
citations, a confidence value and an optional abstention with a reason code.
Free-text answers would make citation precision and abstention unmeasurable.

Provider credentials come exclusively from the environment through
``evidence_route.config.Settings``. No endpoint, deployment name or key appears
in this package.

Modules:
    ``schema``    the structured answer contract
    ``prompts``   versioned prompt templates; the version rides on every request
    ``client``    provider protocol, Azure OpenAI, and offline replay providers
    ``cache``     content-addressed response cache (spec section 26)
    ``cost``      token pricing, where unknown cost stays distinct from zero
"""

from evidence_route.generation.cache import CachingProvider, ResponseCache
from evidence_route.generation.client import (
    AzureOpenAIProvider,
    GenerationProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderError,
    RecordedProvider,
    ScriptedProvider,
    build_provider,
)
from evidence_route.generation.cost import (
    CostEstimate,
    TokenPricing,
    estimate_cost,
    register_pricing,
)
from evidence_route.generation.prompts import (
    DIRECT_V1,
    GROUNDED_V1,
    PromptTemplate,
    format_evidence,
    get_template,
)
from evidence_route.generation.schema import (
    AnswerCitation,
    StructuredAnswer,
    extract_json_object,
    parse_structured_answer,
)

__all__ = [
    "DIRECT_V1",
    "GROUNDED_V1",
    "AnswerCitation",
    "AzureOpenAIProvider",
    "CachingProvider",
    "CostEstimate",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResponse",
    "PromptTemplate",
    "ProviderError",
    "RecordedProvider",
    "ResponseCache",
    "ScriptedProvider",
    "StructuredAnswer",
    "TokenPricing",
    "build_provider",
    "estimate_cost",
    "extract_json_object",
    "format_evidence",
    "get_template",
    "parse_structured_answer",
    "register_pricing",
]
