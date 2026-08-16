# Research protocol v1

**Status:** DRAFT — not yet frozen
**Created:** 2026-08-15
**Frozen on:** _(record the date and commit SHA when this is frozen)_
**Applies to:** experiment `final-v1`

---

## Why this document exists

This protocol is written **before** the final test set is analysed. That
ordering is the entire point. Hypotheses chosen after seeing results are not
hypotheses, and confidence intervals computed on data that guided the modelling
do not mean what they appear to mean.

Once frozen, changes to this document require a new version (`protocol_v2.md`)
with the reason for the change recorded. The frozen version and its commit SHA
are cited in the technical report.

---

## 1. Research question

> Can an adaptive routing policy select the least expensive sufficient
> question-answering workflow while preserving or improving evidence-grounded
> answer quality relative to fixed RAG and fixed agentic baselines?

## 2. Hypotheses

Recorded in advance. Each will be reported as supported, not supported, or
inconclusive — including the ones that fail.

| ID | Hypothesis |
| --- | --- |
| H1 | No single fixed workflow dominates across correctness, cost and latency for every query type. |
| H2 | Hybrid retrieval plus reranking improves evidence recall on difficult questions, but not enough to justify itself on simple high-overlap questions. |
| H3 | A contextual router achieves a better quality-cost Pareto position than the best fixed workflow. |
| H4 | Features derived from retrieval agreement, retrieval margin and question type are more predictive than query length alone. |
| H5 | An explicit abstention action reduces unsupported answers at an acceptable reduction in coverage. |
| H6 | The fixed agentic workflow has higher average cost and latency, with value concentrated on multi-step and synthesis questions. |
| H7 | Doubly robust OPE is more stable than basic IPS when the reward model is reasonably specified and behavior-policy coverage is adequate. |
| H8 | LLM-judge results alone overestimate performance on at least one important failure category relative to human or deterministic evaluation. |

## 3. Non-goals

This project is not, and will not become:

- a general-purpose chat interface
- a demonstration of many frameworks with no controlled experiment
- a collection of unrelated notebooks
- a benchmark reporting only the most favourable result
- a claim that agentic RAG is automatically superior
- a large web-development project
- a GRPO fine-tuning project before routing and evaluation baselines work

## 4. Action space

Frozen for the MVP. Adding an action widens the outcome matrix, changes the
propensity denominators in the logged-feedback simulation, and invalidates
previously trained routers — so it requires a protocol revision, not a code
change.

| ID | Action |
| --- | --- |
| A0 | Direct answer, no retrieval |
| A1 | BM25 retrieval + generation |
| A2 | Dense retrieval + generation |
| A3 | Hybrid retrieval (RRF) + generation |
| A4 | Hybrid + cross-encoder reranking + generation |
| A5 | Bounded multi-step agentic retrieval |
| A6 | Abstain with a machine-readable reason code |

## 5. Primary metrics

**Primary outcome:** policy value under the `balanced_enterprise` reward
profile, measured on the frozen test split.

**Reported alongside, always, never collapsed into the scalar:**

- answer correctness (exact match / numeric tolerance / semantic equivalence)
- evidence support and citation precision
- retrieval recall@k — measured separately from answer quality, because a
  correct answer can conceal poor retrieval and vice versa
- hallucination / unsupported-claim rate
- mean cost per question (USD)
- mean end-to-end latency
- answer coverage and abstention precision

## 6. Reward profiles

Four declared profiles: `quality_first`, `balanced_enterprise` (primary),
`cost_sensitive`, `regulated`. Weights are in `configs/rewards/profiles.yaml`
and mirrored in `evidence_route.evaluation.rewards`.

Every headline result is reported under **all four**. A finding that holds under
only one weighting is a finding about the weighting, and will be described that
way.

## 7. Split policy

- Grouped by company/document, never randomised per question — several
  questions share filings, and question-level splitting leaks.
- Seed `20260815`, recorded in every manifest.
- Test identifiers frozen and committed before any test execution.
- `protect_test_split` defaults to true; `final_test.yaml` requires an explicit
  unlock.

## 8. Feature-availability rule

The router may use only information available **before the answer is known**.

Any feature computed from the answer, the reference answer, or a workflow the
router did not pay for makes the routing result undeployable. This is enforced
by schema tests, not by convention. Retrieval-probe cost is charged to the
router: a router that runs every expensive workflow before choosing is not a
cheap router.

## 9. Primary comparison

| | |
| --- | --- |
| Treatment | the learned router (`supervised_v1`) |
| Comparator | `strongest_validation` — the fixed workflow with the highest mean reward on validation |
| Metric | policy value under `balanced_enterprise` |
| Interval | paired bootstrap, 10 000 resamples, clustered by `question_id`, 95% |

Secondary comparisons (LinUCB, Thompson, rule-based, each against the same
comparator) are corrected with Holm. The primary comparison is not corrected —
it was declared in advance.

## 10. Practical significance

Declared before the test, so "statistically detectable" and "worth doing" stay
separable:

| Threshold | Value |
| --- | --- |
| Minimum quality improvement justifying extra cost | +0.03 |
| Maximum acceptable mean latency | 15 000 ms |
| Maximum acceptable hallucination rate | 0.05 |
| Minimum acceptable answer coverage | 0.80 |
| Cost saving considered operationally meaningful | 20% |

### Numeric tolerance

`NumericalAnswerScorer` uses a **relative tolerance of 1%** by default.

Declared here because it decides which answers count as correct, and because it
is more generous than it first appears: on FinanceBench's larger figures 1%
accepts ±666 on a reported 66,608. That is wide enough to admit a genuinely
different figure while excluding only gross errors.

The argument for keeping it: filings round, models restate figures in different
units, and a tolerance tight enough to catch every near-miss also fails correct
answers for formatting. The argument for tightening it to 0.1%: these are
extraction questions with an exact answer printed in the document, so a 1%
discrepancy is usually a wrong number rather than rounding.

**Open — decide before the outcome matrix runs**, since changing it afterwards
alters every correctness score. Whichever value is chosen, the human-reviewed
sample (section 15) is what establishes whether this scorer's verdicts match a
reader's.

## 11. Statistical protocol

- **Unit of analysis:** the question. All workflows run on the same questions,
  so comparisons are paired.
- **Hierarchy preserved:** repeated generations of one question are replicates,
  not independent questions. Resampling clusters at question level.
- **Tests:** paired permutation for continuous differences, McNemar for paired
  binary correctness, bootstrap for policy-value differences.
- **Emphasis:** effect sizes and intervals ahead of isolated p-values.
- **Fixed within comparisons:** temperature 0.0, identical prompt versions,
  recorded model and API versions.

## 12. Null results

The project succeeds even if the learned router does not beat the strongest
fixed baseline, provided the experiment is correctly designed, the outcome is
reproducible, the reason is investigated, uncertainty is reported, and the
limitations of features, sample size or reward are explained.

A sound negative result is a stronger research signal than an unverified
improvement, and will be published as the headline if that is what the data
shows.

---

## 13. Immediate implementation decisions

Recorded per spec section 33. Fill in the remaining rows before freezing.

| # | Decision | Value |
| --- | --- | --- |
| 1 | Repository name | `evidence-route` |
| 2 | Primary generation model | _TBD — one Azure OpenAI chat deployment_ |
| 3 | Embedding model | Azure embedding deployment (primary); local MiniLM documented as fallback |
| 4 | Dense index | Local FAISS (flat inner product) |
| 5 | BM25 implementation | `rank-bm25` |
| 6 | Reranker | **OPEN** — cross-encoder vs Azure LLM reranker, decided in week 5 on measured evidence (see below) |
| 7 | Agent implementation | Minimal custom loop, bounded, 5 allowlisted tools |
| 8 | Experiment budget | USD 25 for the outcome matrix, USD 40 for the final test |
| 9 | FinanceBench split | Grouped by company, seed 20260815, test frozen |
| 10 | Primary reward profile | `balanced_enterprise` |
| 11 | Primary comparison | Learned router vs strongest fixed validation workflow |
| 12 | Evallab version strategy | Local editable during development; pinned commit for publication |

### Open decision 6 — reranker choice

Deliberately not settled at scaffolding time. The spec permits either a
cross-encoder or an LLM reranker (section 6, A4), and the two differ in ways
that bear directly on what A4 measures:

| | Cross-encoder | Azure LLM reranker |
| --- | --- | --- |
| Per-query cost | free | billed per query |
| Determinism | deterministic | sampling-dependent |
| Offline reproduction | yes | needs an Azure account |
| Local install | ~3–4 GB (torch) | none |
| Long-term stability | weights frozen | deployment can be re-pointed |

Since A4's entire purpose is to measure whether reranking earns its added
latency and cost, choosing the reranker on install convenience would bias the
answer to the question the action exists to ask. The decision is made in roadmap
week 5, on a like-for-like comparison on development data, and recorded here
before the outcome matrix runs.

Dependency handling in the meantime: `local-models` is an isolated extra, so
neither path is foreclosed and neither is installed by default.

---

## 14. Freeze checklist

Every box must be ticked before `configs/experiments/final_test.yaml` runs.

- [ ] Hypotheses recorded and unchanged since drafting
- [ ] Action space frozen
- [ ] Reward profiles and weights frozen
- [ ] Split identifiers generated and committed
- [ ] Primary comparison and metric declared
- [ ] Practical-significance thresholds declared
- [ ] All routers and thresholds fitted on dev/validation only
- [ ] Decisions table above completed
- [ ] Cost estimated on a subset and within budget
- [ ] Worktree clean; this document committed
- [ ] Freeze date and commit SHA recorded at the top of this file
