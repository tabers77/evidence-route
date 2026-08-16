"""Token cost estimation (spec section 26).

The distinction this module exists to preserve: **unknown cost is not zero
cost.** Azure deployment names are arbitrary — a deployment called ``gpt4o-prod``
could point at anything — so the model behind a deployment cannot be inferred
reliably. When pricing is unknown, :func:`estimate_cost` returns ``None`` rather
than a number, and the caller records that the cost is unmeasured.

Recording 0.0 for an unpriced call would make the budget ceiling silently
useless: a run could exhaust a real budget while reporting that it had spent
nothing, and the cost dimension of every reward would be wrong in the direction
that flatters expensive workflows.

Prices are per million tokens and carry the date they were read, because
provider pricing changes and a result's cost figures are only interpretable
against the prices in force when it ran.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_route.storage.records import TokenUsage

__all__ = [
    "PRICING",
    "CostEstimate",
    "TokenPricing",
    "estimate_cost",
    "register_pricing",
]


@dataclass(frozen=True)
class TokenPricing:
    """USD per million tokens for one model."""

    input_usd_per_million: float
    output_usd_per_million: float
    #: Where the figures came from and when. Pricing changes; a cost is only
    #: interpretable against the prices in force at the time.
    source: str

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_million + output_tokens * self.output_usd_per_million
        ) / 1_000_000


@dataclass(frozen=True)
class CostEstimate:
    """A cost, or an explicit statement that it is unknown."""

    usd: float | None
    pricing_known: bool
    model_key: str | None = None
    note: str | None = None

    @property
    def usd_or_zero(self) -> float:
        """For arithmetic that cannot carry None.

        Callers using this must also record ``pricing_known``, or an unpriced
        run will look free.
        """
        return self.usd or 0.0


#: Model pricing, keyed by a normalized model name.
#:
#: Deliberately left empty of guesses. Populate it from the Azure pricing page
#: for the deployments actually in use, and record the date. An entry invented
#: from memory is worse than no entry, because it produces a confident number
#: that no one will re-check.
PRICING: dict[str, TokenPricing] = {}


def register_pricing(model_key: str, pricing: TokenPricing) -> None:
    """Register pricing for a model.

    Call this at startup from configuration rather than editing the module, so
    the prices used by a run are part of that run's recorded config.
    """
    PRICING[model_key.strip().lower()] = pricing


def estimate_cost(usage: TokenUsage, model_key: str | None) -> CostEstimate:
    """Estimate the USD cost of one call, retries included.

    Retry tokens are billed and are therefore counted. Excluding them would
    understate spend precisely on the questions that were hardest, which is
    where a workflow's true cost matters most (spec section 26, item 8).
    """
    if not model_key:
        return CostEstimate(
            usd=None,
            pricing_known=False,
            note="No model key supplied; cost cannot be attributed.",
        )

    key = model_key.strip().lower()
    pricing = PRICING.get(key)
    if pricing is None:
        return CostEstimate(
            usd=None,
            pricing_known=False,
            model_key=key,
            note=(
                f"No pricing registered for {key!r}. Register it with "
                f"register_pricing() so cost is measured rather than assumed zero."
            ),
        )

    total_input = usage.input_tokens + usage.retry_input_tokens
    total_output = usage.output_tokens + usage.retry_output_tokens
    return CostEstimate(
        usd=pricing.cost_for(total_input, total_output),
        pricing_known=True,
        model_key=key,
        note=pricing.source,
    )
