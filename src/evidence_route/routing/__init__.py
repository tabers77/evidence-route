"""Routing policies: fixed, rule-based, supervised and oracle (spec section 9).

The ladder of baselines exists so that any claim about the learned router is
measured against something honest:

``fixed``
    Always the same action. Seven of these, one per action, plus "cheapest
    non-abstaining" and "empirically strongest on validation".

``rule_based``
    Interpretable thresholds fitted on development data. Frequently competitive,
    and a learned router that cannot beat it has not earned its complexity.

``supervised``
    Oracle-imitation. Because the experimental phase can run every action on
    every development question, full-information best-action labels exist.
    Logistic regression and trees are treated as serious baselines, not
    placeholders — a neural router is not assumed superior.

``oracle``
    Selects the best action *after* observing all outcomes. Not deployable; it
    bounds the value available from perfect routing. The gap between the learned
    router and the oracle diagnoses whether the limitation is the features, the
    sample size, the reward or the model capacity.

Thresholds and hyperparameters are selected on development or validation data.
Never on the final test set.
"""
