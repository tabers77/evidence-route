# Configuration

Everything in this directory is version-controlled, because **a result is only
reproducible if the configuration that produced it is in git**.

Secrets are the opposite: no endpoint, deployment name, key or tenant appears in
any file here. Those live in a gitignored `.env` and are read at runtime through
`evidence_route.config.Settings`. Configs refer to credentials indirectly, via
`deployment_ref: chat`, which resolves against the environment.

## Layout

| Directory | Contents |
| --- | --- |
| `datasets/` | Source, license, parsing, chunking and split rules per corpus |
| `workflows/` | The candidate action space and each action's parameters |
| `routers/` | Fixed, rule-based, supervised and contextual-bandit policies |
| `rewards/` | Reward profiles, normalization scales and sensitivity sweeps |
| `experiments/` | What to run, on which splits, under which budget |

## Conventions

**Placeholders are explicit.** A config with unfitted thresholds carries
`fitted: false` and `null` values, so a run using arbitrary numbers is obvious
rather than quiet. The training step writes the values back and flips the flag.

**Versions are part of identity.** Reward profiles, prompts, normalization rules
and feature sets are versioned. Changing a weight produces a new version rather
than editing the old one, because rewards are only comparable within a version.

**Splits are asymmetric on purpose.** `mvp.yaml` runs on dev and validation.
Only `final_test.yaml` touches the test split, it requires an explicit unlock,
and running it is a one-way door — see the warning at the top of that file.

## The experiment sequence

```
smoke.yaml           offline wiring check; no API key, no cost, runs in CI
   ↓
mvp.yaml             full-information outcome matrix on dev + validation
   ↓
ope_simulation.yaml  replay behavior policies to produce logged feedback
   ↓
ope_study.yaml       score estimators against known policy values
   ↓
final_test.yaml      frozen evaluation on the held-out test split
```
