# EvidenceRoute — system card

**Status:** skeleton. Completed alongside the final evaluation (roadmap week 12).

---

## Intended use

A reproducible research benchmark for studying adaptive workflow routing in
evidence-grounded enterprise question answering. It exists to produce measured
comparisons, not to serve production traffic.

## Out-of-scope use

- Production question answering over private or regulated company data
- Financial, legal or medical advice
- Any deployment where an unsupported answer carries material risk without human
  review
- Treating the reported policy values as transferable to a different corpus,
  provider or reward weighting without re-measurement

## Datasets

_FinanceBench, BEIR/FiQA, RAGTruth. See `data/README.md` for licenses,
provenance and split policy._

## Model providers

_Azure OpenAI deployments for generation, embedding and judging; local
cross-encoder for reranking. Deployment-to-model-version mapping is recorded per
run, because an Azure deployment can be re-pointed at a new model version
without any change in this repository._

## Evaluation coverage

_Which question types, retrieval conditions and failure categories were actually
measured — and which were not._

## Known failure modes

_Populated from the failure taxonomy (spec section 16): retrieval, generation,
routing, agent and evaluation failures._

## Human oversight

_The abstention action routes to human review. The human evaluation protocol and
its inter-rater agreement are documented in `human_evaluation_protocol.md`._

## Cost and latency characteristics

_Per-action mean cost and latency, and the cost of reproducing the benchmark._

## Privacy and security considerations

No private or company data is used. Credentials are read from the environment
and never committed; run manifests record deployment names but never endpoints
or keys.

## Limitations of automated judges

_Judge validation results against the RAGTruth annotations and the human-reviewed
sample, including the failure categories where the judge and humans disagree._
