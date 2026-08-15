# EvidenceRoute developer commands.
#
# Windows users without `make`: an equivalent PowerShell entrypoint is provided
# in tasks.ps1 (e.g. `.\tasks.ps1 test`), or use the Docker path (`make docker-*`).

SHELL := /bin/bash

# Virtualenv layout differs between Windows and POSIX.
ifeq ($(OS),Windows_NT)
	VENV_BIN := .venv/Scripts
	PY       := py -3
else
	VENV_BIN := .venv/bin
	PY       := python3
endif

PYTHON := $(VENV_BIN)/python
PIP    := $(PYTHON) -m pip

# Path to the sibling Evallab checkout, installed as an editable dependency
# during the development phase (spec section 2.3). Override if it lives elsewhere:
#   make install EVALLAB_PATH=/some/other/evallab
EVALLAB_PATH ?= ../evallab

.DEFAULT_GOAL := help
.PHONY: help venv install install-all install-evallab lint format typecheck test \
        test-all smoke clean docker-build docker-test docker-shell env-check tree

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
venv:  ## Create the local virtual environment
	$(PY) -m venv .venv
	$(PIP) install --upgrade pip

install: venv install-evallab  ## One-command setup: venv + core package + dev tools
	$(PIP) install -e ".[dev]"

install-all: venv install-evallab  ## Full experiment environment (retrieval, generation, analysis)
	$(PIP) install -e ".[all,dev]"

install-evallab:  ## Install the sibling Evallab checkout as an editable dependency
	@if [ -d "$(EVALLAB_PATH)" ]; then \
		echo "Installing Evallab (editable) from $(EVALLAB_PATH)"; \
		$(PIP) install -e "$(EVALLAB_PATH)"; \
	else \
		echo "WARNING: Evallab not found at $(EVALLAB_PATH)."; \
		echo "         Evallab-dependent code will be unavailable."; \
		echo "         Clone it next to this repo or pass EVALLAB_PATH=..."; \
	fi

env-check:  ## Verify the environment is wired up (imports, config, evallab)
	$(PYTHON) -m evidence_route.cli env check

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
lint:  ## Run ruff checks
	$(PYTHON) -m ruff check src tests scripts

format:  ## Apply ruff formatting and import ordering
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

typecheck:  ## Run mypy
	$(PYTHON) -m mypy

test:  ## Run the default test suite (no paid API calls)
	$(PYTHON) -m pytest -m "not llm and not slow"

test-all:  ## Run every test, including live-model tests (incurs cost)
	$(PYTHON) -m pytest

smoke:  ## Run the offline end-to-end smoke experiment
	$(PYTHON) -m pytest -m smoke

check: lint typecheck test  ## Lint + typecheck + test

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build:  ## Build the container image
	docker compose build

docker-test:  ## Run the default test suite inside the container
	docker compose run --rm app python -m pytest -m "not llm and not slow"

docker-shell:  ## Open an interactive shell in the container
	docker compose run --rm app bash

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
clean:  ## Remove caches and build artifacts (keeps .venv and data)
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
