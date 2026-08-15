"""The statistical analysis protocol (spec section 14).

The unit of analysis is the question. Because every workflow is evaluated on the
same questions, comparisons are paired — and paired comparisons on a small
benchmark are far more informative than unpaired ones.

Repeated generations of the same question are *not* independent questions. When
a workflow is stochastic and run with several seeds, the hierarchy
(question → workflow → replicate) is preserved and resampling happens at the
question level. Collapsing that hierarchy would shrink confidence intervals by
pretending to have more data than exists.

Effect sizes and confidence intervals are reported ahead of isolated p-values,
and the primary comparison is declared before the final test set is touched.

Planned modules:
    ``bootstrap``    paired bootstrap intervals, clustered at question level
    ``tests``        paired permutation, McNemar, policy-value bootstrap
    ``calibration``  reliability diagrams, ECE, Brier, isotonic/Platt fitting
    ``corrections``  multiple-comparison handling for secondary comparisons
    ``power``        practical-significance thresholds and sample-size notes
"""
