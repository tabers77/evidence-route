# Data card

## What is and is not stored here

This repository stores **metadata**: manifests, checksums, split definitions and
transformation notes. It does **not** store the source documents. The
FinanceBench corpus is public but its license does not grant redistribution, and
committing PDFs to a portfolio repository would be both a licensing problem and
a repository-size problem.

Everything under `raw/`, `interim/`, `processed/` and `cache/` is gitignored and
produced locally by:

```bash
evidence-route data prepare --config configs/datasets/financebench.yaml
```

| Path | Tracked | Contents |
| --- | --- | --- |
| `manifests/` | yes | Source URLs, licenses, checksums, download timestamps |
| `splits/` | yes | Frozen example identifiers per split |
| `raw/` | no | Downloaded source documents |
| `interim/` | no | Parsed text and extracted tables |
| `processed/` | no | Chunked corpora ready for indexing |
| `cache/` | no | Cached provider responses |

## Datasets

### FinanceBench — primary benchmark

- Source: <https://github.com/patronus-ai/financebench>
- License: CC BY-NC 4.0 (non-commercial)
- Role: the principal business-facing evaluation set

Questions concern public financial documents, answers have associated evidence,
and the mix spans extraction, numerical reasoning and evidence synthesis. It
resembles enterprise decision support far more closely than a trivia benchmark.

**Stated limitation.** The public sample is small. It is used primarily as a
high-quality held-out evaluation set, not as an unlimited source of bandit
training observations. Routing features are fitted on FiQA instead, and every
FinanceBench result is reported with paired uncertainty intervals wide enough to
be honest about the sample size.

### BEIR / FiQA — supporting retrieval set

- Source: <https://github.com/beir-cellar/beir>
- License: Apache 2.0
- Role: retrieval comparison and routing-feature fitting at usable sample sizes

Provides enough queries to compare retrieval approaches statistically and to
train routing features without consuming FinanceBench test examples. One BEIR
dataset in the MVP, not all of them.

### RAGTruth — scorer validation only

- Source: <https://github.com/ParticleMedia/RAGTruth>
- License: MIT
- Role: check whether the hallucination and evidence-support scorers detect
  externally annotated unsupported spans

Without an external annotation source, judge quality can only be validated
against self-generated examples, which is circular. RAGTruth breaks that circle.
It does not become a third application branch.

## Splitting policy

Splits are **grouped by document or company**, not sampled per question.

Several FinanceBench questions reference the same filing. Under question-level
random splitting, a model can learn document-specific patterns in training and
be rewarded for them at test time — the split would leak, and the held-out
estimate would be optimistic in a way no confidence interval would reveal.

Splits are generated once with a recorded seed and then frozen. The test-set
identifiers are committed under `splits/`, and regenerating them with a
different seed invalidates every result that cites the old split.

| Split | Purpose |
| --- | --- |
| `dev` | Feature development, prompt iteration, router fitting |
| `validation` | Threshold selection, hyperparameters, reward-profile choice |
| `test` | Frozen. Read once, under `configs/experiments/final_test.yaml` |
| `challenge` | Optional. Difficult or adversarial examples, reported separately |

## Versioning

Each manifest under `manifests/` records, per dataset:

- name and version
- original source URL
- license
- download timestamp
- per-file checksums
- transformation steps applied
- split-generation seed
- the final example identifiers in each split

That set is the minimum needed for someone else to confirm they are looking at
the same data — see spec section 7.5.

## Reproducing without the corpus

The offline smoke experiment (`configs/experiments/smoke.yaml`) runs against a
handful of fixture questions checked into `tests/fixtures/`. It needs no
download, no API key and no cost, and it is what CI runs on every push.
