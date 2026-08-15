"""Contextual-bandit policies (spec section 9.4).

Three algorithms for the MVP — epsilon-greedy, LinUCB and contextual Thompson
sampling. The goal is to demonstrate correct problem formulation and honest
evaluation, not to reimplement every known bandit algorithm.

Correctness here is verifiable in a way that end-to-end results are not: on a
small stationary synthetic environment with a known optimal arm, a correct
implementation must converge to it. Those fixtures are the first line of defence
against a subtly wrong update rule producing plausible-looking results
(spec section 24, "statistical tests").

Exploration parameters are selected without touching the final test set.
"""
