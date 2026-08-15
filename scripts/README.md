# Scripts

Thin entry points. Every script here is a wrapper around the CLI, which is a
wrapper around `src/evidence_route/`. Logic lives in the package, not in these
files — a pipeline that only works when driven by a script is not testable.

| Script | CLI equivalent |
| --- | --- |
| `download_data.py` | `evidence-route data prepare` |
| `build_indexes.py` | `evidence-route index build` |
| `run_outcome_matrix.py` | `evidence-route outcomes run` |
| `simulate_bandit_logs.py` | `evidence-route bandit simulate` |
| `run_ope_study.py` | `evidence-route ope evaluate` |
| `build_report.py` | `evidence-route report build` |

They exist because the full pipeline is a sequence, and a script is easier to
schedule, log and resume than a shell one-liner.

## Full sequence

```bash
python scripts/download_data.py       --config configs/datasets/financebench.yaml
python scripts/build_indexes.py       --experiment configs/experiments/mvp.yaml
python scripts/run_outcome_matrix.py  --experiment configs/experiments/mvp.yaml
python scripts/simulate_bandit_logs.py --config configs/experiments/ope_simulation.yaml
python scripts/run_ope_study.py       --config configs/experiments/ope_study.yaml
python scripts/build_report.py        --experiment-id final-v1
```

`run_outcome_matrix.py` is the expensive one. Run it with `--dry-run` first: it
estimates token use and cost on a subset before anything is billed.
