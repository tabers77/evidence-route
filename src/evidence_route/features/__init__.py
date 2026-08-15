"""Context features for routing (spec section 8).

**The feature-availability rule.** The router may only use information available
*before* the answer is known. A feature computed from the answer, from the
reference answer, or from any workflow the router did not pay for makes the
routing result undeployable — the system could not compute it at decision time.
This is the single most important correctness property in the project, and it is
enforced by schema tests rather than by convention (spec section 27, "router
uses leaked post-answer features").

**Probe cost counts.** A cheap first-stage retrieval probe supplies the most
predictive features (score margins, lexical-dense agreement, rank overlap), but
running it is not free. Its cost is attributed to the router, not amortised away
— a router that secretly executes every expensive workflow before choosing is
not a cheap router.

Planned modules:
    ``query``      length, entities, numerals, question-type, ambiguity signals
    ``probe``      retrieval-probe features and their cost accounting
    ``confidence`` router-confidence features, also used for abstention
    ``schema``     the feature contract and its availability assertions
"""
