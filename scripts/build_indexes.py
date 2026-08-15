"""Build the BM25 and dense retrieval indexes for an experiment.

    python scripts/build_indexes.py --experiment configs/experiments/mvp.yaml

Indexes are gitignored: they are large and fully rebuildable from the prepared
corpus plus the experiment config. The embedding model version used to build a
dense index is recorded, because an index built with a different embedding model
is a different index.
"""

from __future__ import annotations

from _common import run_cli

if __name__ == "__main__":
    run_cli("index", "build")
