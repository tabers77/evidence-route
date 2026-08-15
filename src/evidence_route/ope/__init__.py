"""Offline policy evaluation (spec section 10).

In production, only the reward of the *selected* action is observed; the rest is
counterfactual. During the controlled research phase EvidenceRoute executes every
action on every question, producing a full-information outcome matrix. Logged
bandit feedback is then simulated from that matrix under several behavior
policies — which means each estimate can be checked against a policy value that
is actually known.

That check is the point. Estimators are usually evaluated by argument; here they
are evaluated by bias, variance, MSE, confidence-interval coverage and
policy-ranking accuracy against ground truth.

The framing matters and must not drift in write-ups: this is *a controlled
logged-feedback simulation constructed from full-information benchmark
outcomes*. It is not production user feedback and not a deployed online RL
system.

Planned modules:
    ``behavior``   behavior-policy simulators, including a deliberately
                   poor-coverage policy that rarely picks expensive actions
    ``estimators`` Direct Method, IPS, self-normalized IPS, doubly robust,
                   and optional switch/clipped variants
    ``diagnostics`` propensity coverage checks and clipping sensitivity
    ``study``      the bias/variance experiment across sample sizes and seeds
"""
