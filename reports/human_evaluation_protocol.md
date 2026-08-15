# Human evaluation protocol

**Status:** skeleton. Executed in roadmap week 11, before the technical report
is written.

---

## Purpose

Automated judges are graded here, not trusted. Human review exists to validate
the automated scorers and to surface failure modes that exact-match and
LLM-judge metrics cannot see — particularly the case where a judge rewards
fluent, well-structured writing that is factually unsupported.

## Sample

Stratified, not random. A random sample of a benchmark this size would be
dominated by easy questions that every workflow answers correctly, which teaches
nothing about where the scorers fail.

Target: 60 examples, stratified across:

- automated score: correct and incorrect
- every workflow action
- high and low judge confidence
- abstentions
- numerical questions
- multi-step agentic questions
- cases where two automated scorers disagree

## Annotation dimensions

Each answer is rated for:

| Dimension | Scale |
| --- | --- |
| Correctness | correct / partially correct / incorrect |
| Completeness | complete / partial / missing |
| Evidence support | fully supported / partially / unsupported |
| Citation accuracy | accurate / imprecise / wrong |
| Unsupported claims | count |
| Appropriate abstention | appropriate / over-cautious / should have abstained |
| Error severity | none / minor / material / critical |

## Annotation guide

_A short written rubric with worked examples for each level of each dimension,
written before annotation begins. Examples come from the dev split, never from
test._

## Reliability

With a single reviewer, inter-rater agreement cannot be measured. That is stated
as a limitation rather than glossed over, and mitigated by re-annotating a
blinded subset after a delay to estimate intra-rater consistency.

If a second reviewer becomes available, agreement is measured on a shared subset
and reported with the appropriate coefficient.

## Judge validation

Automated scores are compared against the human labels using:

- accuracy / F1 for categorical labels
- correlation for ordinal ratings
- confusion matrices
- false-positive and false-negative analysis, reported separately — a judge that
  misses unsupported claims fails differently from one that flags correct
  answers
- agreement broken down by question type

## Blinding

Annotations are made without visibility of which workflow produced the answer or
what the automated scorer concluded. Knowing either would contaminate the
comparison the annotation exists to make.
