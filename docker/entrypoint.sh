#!/usr/bin/env bash
# Container entrypoint.
#
# Evallab is a local editable dependency during development (spec section 2.3),
# and docker-compose mounts the sibling checkout at /evallab. That mount only
# exists at RUN time, so the install cannot happen during the image build —
# it happens here instead.
#
# Absence of the mount is not an error: the schema, reward and OPE layers do not
# import Evallab, so the image stays useful standalone.

set -euo pipefail

if [ -d /evallab ] && [ -f /evallab/pyproject.toml ]; then
    if ! python -c "import agent_eval" >/dev/null 2>&1; then
        echo "[entrypoint] Installing Evallab (editable) from /evallab"
        pip install --quiet -e /evallab
    fi
else
    echo "[entrypoint] Evallab not mounted at /evallab; Evallab-dependent paths unavailable." >&2
fi

# git refuses to read a repository owned by a different uid. The bind-mounted
# worktree comes from the host, so mark it safe — otherwise every run manifest
# would record an unknown commit and no result would be publishable.
git config --global --add safe.directory /workspace 2>/dev/null || true

exec "$@"
