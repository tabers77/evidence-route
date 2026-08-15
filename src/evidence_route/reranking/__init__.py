"""Cross-encoder reranking (action A4).

Reranking is included as a candidate whose cost must be justified, not as an
assumed improvement. The measurement that matters is whether it raises evidence
quality *enough* to pay for its additional latency, which is why the reranking
stage has its own entry in ``LatencyBreakdown``.

Planned modules:
    ``cross_encoder``   the reranker itself
    ``base``            reranker protocol shared with any future alternative
"""
