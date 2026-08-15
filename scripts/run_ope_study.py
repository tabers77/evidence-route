"""Score offline policy estimators against known full-information policy values.

    python scripts/run_ope_study.py --config configs/experiments/ope_study.yaml

Compares DM, IPS, clipped IPS, SNIPS and doubly robust estimation on bias,
variance, MSE, confidence-interval coverage and policy-ranking accuracy — all
measured against values that are actually known, because every action was run on
every question.

Includes a deliberately poor-coverage behavior policy so the small-propensity
failure mode is documented rather than avoided.

Cheap: no model calls, only estimation over existing logs.
"""

from __future__ import annotations

from _common import run_cli

if __name__ == "__main__":
    run_cli("ope", "evaluate")
