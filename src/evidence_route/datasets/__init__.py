"""Dataset loading, validation, splitting and versioning (spec section 7).

Three corpora, with three distinct jobs:

``financebench``
    Primary business-facing benchmark. Public financial documents with
    associated evidence, covering extraction, numerical reasoning and evidence
    synthesis. The public sample is small, so it is treated as a high-quality
    held-out evaluation set — not as an unlimited source of bandit training
    observations.

``fiqa`` (BEIR)
    Supporting retrieval set. Provides enough queries to compare retrieval
    approaches and to fit routing features without consuming FinanceBench test
    examples.

``ragtruth``
    Scorer-validation set only. Used to check whether the hallucination and
    evidence-support scorers actually detect annotated unsupported spans, rather
    than trusting self-generated evaluation examples. It does not become a third
    application branch.

Splitting is grouped by document or company rather than sampled per question:
several questions often share a source, and question-level random splitting
would leak document-specific patterns from training into test (spec 7.4).

This repository stores manifests, checksums and split definitions — not the
source documents, whose licenses generally do not permit redistribution.
"""
