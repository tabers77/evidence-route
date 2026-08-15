# EvidenceRoute — technical report

**Status:** skeleton. Sections are filled in as the corresponding roadmap weeks
complete. Nothing here may state a result before the frozen protocol has run.

Every number that appears in this report must be traceable to a run manifest
under `experiments/runs/`. Claims without that trace do not go in.

---

## 1. Abstract

_To be written last._

## 2. Motivation

Enterprise question answering is usually built as "vector database plus LLM",
applied uniformly to every question. That is an architectural assumption, not a
measured decision. This project treats workflow selection as a sequential
decision problem to be formulated, measured and evaluated.

## 3. Related concepts and prior work

- Classical vs. neural information retrieval; hybrid fusion
- Contextual bandits and exploration/exploitation
- Offline policy evaluation: DM, IPS, SNIPS, doubly robust
- LLM-as-judge evaluation and its known biases
- Selective prediction, calibration and abstention

## 4. Problem formulation

_Action space, context features, reward definition, and the constraint that the
router may use only pre-answer information._

## 5. Datasets

_FinanceBench, FiQA, RAGTruth. Split policy, grouping, sizes, and the stated
limitation that the FinanceBench public sample is small._

## 6. Candidate workflows

_A0-A6, their parameters, and the cost/latency profile of each._

## 7. Routing methods

_Fixed, rule-based, supervised oracle-imitation, LinUCB, contextual Thompson
sampling, and the oracle upper bound._

## 8. Reward and offline evaluation

_Reward profiles, normalization, the logged-feedback simulation, and the
estimators under test._

## 9. Experimental protocol

_Reference `experiments/protocols/protocol_v1.md` and its freeze commit SHA._

## 10. Results

_Principal table with paired uncertainty intervals. Reported under all four
reward profiles._

## 11. Statistical analysis

_Paired bootstrap intervals, primary and secondary comparisons, multiple-
comparison correction, calibration analysis._

## 12. Failure analysis

_Failure taxonomy with representative examples, per spec section 16._

## 13. Limitations and threats to validity

_Sample size, judge bias, provider drift, parsing quality, propensity coverage,
and anything the results themselves expose._

## 14. Operational implications

_What a team building enterprise QA should take from this — including where the
answer is "the simple baseline was fine"._

## 15. Conclusion

_Including negative results. A sound negative finding is reported as the
headline if that is what the data shows._
