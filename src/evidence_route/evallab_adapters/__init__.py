"""The boundary between EvidenceRoute and Evallab (spec section 13).

Everything crossing into Evallab passes through this package, so the dependency
has exactly one seam. Each EvidenceRoute execution becomes one Evallab
``Episode``; each workflow trace event becomes a ``Step``.

Evallab's canonical types live in ``agent_eval.core``, not at the top level of
``agent_eval``. They are imported lazily, so the schema, reward and OPE layers
install and test without Evallab present::

    from agent_eval.core import Episode, Step, StepKind, ScoreVector, ScoreDimension

Episode metadata carries the experimental identity that Evallab itself does not
model: ``question_id``, ``dataset``, ``split``, ``document_ids``,
``workflow_action``, ``router_name``, ``router_version``, ``model_name``,
``embedding_model``, ``reranker_model``, ``experiment_id``, ``random_seed``.

Direction of travel is one-way by design: if EvidenceRoute needs a generally
useful capability, it is implemented and tested in Evallab first and then
consumed here. Domain-specific logic never migrates upstream.

Planned modules:
    ``episode``   trace → canonical Episode
    ``steps``     workflow events → Step, including agent tool calls
    ``scoring``   running EvidenceRoute scorers through Evallab's pipeline
    ``reward``    ScoreVector → scalar reward via the declared profiles
"""
