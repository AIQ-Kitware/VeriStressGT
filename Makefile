# VeriStressGT Makefile
# ═══════════════════════════════════════════════════════════════════════
# Usage:
#   make install          Install core (minimal, no torch)
#   make install-generate Install core + torch/onnx for building instances
#   make install-all      Install everything pip can handle
#   make install-verifiers Set up all verifier submodules + conda envs
#   make doctor           Run dependency health check
#   make test-fresh       Simulate a brand-new user (uses Docker)
#   make docker           Build the all-in-one Docker image
# ═══════════════════════════════════════════════════════════════════════

SHELL := /bin/bash
PYTHON ?= python3
PIP ?= pip

.PHONY: help install install-generate install-all install-dev \
        install-verifiers install-abcrown install-neuralsat install-nnenum \
        install-marabou install-nnv install-pyrat \
        doctor test test-fresh docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Installation targets ─────────────────────────────────────────────

install:  ## Install core only (numpy, pyyaml — no torch)
	$(PIP) install -e .

install-generate:  ## Install core + instance generation (torch, onnx)
	$(PIP) install -e '.[generate]'

install-all:  ## Install everything pip can handle
	$(PIP) install -e '.[all]'

install-dev:  ## Install with dev/test tools
	$(PIP) install -e '.[dev]'

# ── Verifier setup ───────────────────────────────────────────────────

install-verifiers: install-submodules install-abcrown install-neuralsat install-nnenum install-marabou install-nnv install-pyrat  ## Set up all verifiers

install-submodules:  ## Pull all git submodules
	git submodule update --init --recursive

install-abcrown: install-submodules  ## Set up α-β-CROWN conda env
	@echo "═══ Setting up α-β-CROWN ═══"
	@ABCROWN_DIR=src/VeriStressGT/verifiers/alpha-beta-CROWN; \
	if [ ! -d "$$ABCROWN_DIR/complete_verifier" ]; then \
		echo "ERROR: $$ABCROWN_DIR not populated. Run: make install-submodules"; \
		exit 1; \
	fi; \
	echo "Creating conda env 'alpha-beta-crown' from environment.yaml..."; \
	conda env create -f $$ABCROWN_DIR/complete_verifier/environment.yaml \
		--name alpha-beta-crown --force 2>/dev/null || \
	conda env update -f $$ABCROWN_DIR/complete_verifier/environment.yaml \
		--name alpha-beta-crown; \
	echo "Installing VeriStressGT into alpha-beta-crown env..."; \
	conda run -n alpha-beta-crown pip install -e . || true; \
	echo "Installing auto_LiRPA from submodule..."; \
	conda run -n alpha-beta-crown pip install -e $$ABCROWN_DIR/auto_LiRPA || true; \
	echo ""; \
	echo "Done. Now set env vars:"; \
	echo "  export ABCROWN_VNNCOMP2024_DIR=$$(pwd)/$$ABCROWN_DIR"; \
	echo "  export ABCROWN_CONDA_ENV=alpha-beta-crown"

install-neuralsat: install-submodules  ## Install NeuralSAT deps
	@echo "═══ Setting up NeuralSAT ═══"
	@NS_DIR=src/VeriStressGT/verifiers/neuralsat; \
	if [ ! -d "$$NS_DIR" ] || [ -z "$$(ls -A $$NS_DIR 2>/dev/null)" ]; then \
		echo "ERROR: $$NS_DIR not populated. Run: make install-submodules"; \
		exit 1; \
	fi; \
	if [ -f "$$NS_DIR/requirements.txt" ]; then \
		$(PIP) install -r $$NS_DIR/requirements.txt; \
	fi; \
	echo "Done. Set: export NEURALSAT_DIR=$$(pwd)/$$NS_DIR"

install-nnenum: install-submodules  ## Install nnenum
	@echo "═══ Setting up nnenum ═══"
	@NNENUM_DIR=src/VeriStressGT/verifiers/nnenum; \
	if [ ! -d "$$NNENUM_DIR" ] || [ -z "$$(ls -A $$NNENUM_DIR 2>/dev/null)" ]; then \
		echo "ERROR: $$NNENUM_DIR not populated. Run: make install-submodules"; \
		exit 1; \
	fi; \
	cd $$NNENUM_DIR && $(PIP) install -e . 2>/dev/null || \
		$(PIP) install . 2>/dev/null || \
		echo "nnenum install: check $$NNENUM_DIR manually"

install-marabou: install-submodules  ## Build and install Marabou (requires cmake)
	@echo "═══ Setting up Marabou ═══"
	@MARABOU_DIR=src/VeriStressGT/verifiers/Marabou; \
	if [ ! -d "$$MARABOU_DIR" ] || [ -z "$$(ls -A $$MARABOU_DIR 2>/dev/null)" ]; then \
		echo "ERROR: $$MARABOU_DIR not populated. Run: make install-submodules"; \
		exit 1; \
	fi; \
	echo "Building Marabou (requires cmake)..."; \
	cd $$MARABOU_DIR && mkdir -p build && cd build && cmake .. && cmake --build . && \
	cd .. && $(PIP) install . 2>/dev/null || \
		echo ""; \
		echo "Marabou build failed. Prerequisites: cmake, a C++ compiler."; \
		echo "See $$MARABOU_DIR/README.md for platform-specific instructions."; \
	echo ""; \
	echo "Done. Set: export MARABOU_DIR=$$(pwd)/$$MARABOU_DIR"

install-nnv: install-submodules  ## Set up NNV (requires MATLAB)
	@echo "═══ Setting up NNV ═══"
	@NNV_DIR=src/VeriStressGT/verifiers/nnv; \
	if [ ! -d "$$NNV_DIR" ] || [ -z "$$(ls -A $$NNV_DIR 2>/dev/null)" ]; then \
		echo "ERROR: $$NNV_DIR not populated. Run: make install-submodules"; \
		exit 1; \
	fi; \
	if command -v matlab >/dev/null 2>&1; then \
		echo "MATLAB found: $$(which matlab)"; \
		echo "NNV repo ready at $$NNV_DIR"; \
		echo "Add NNV to your MATLAB path — see $$NNV_DIR/README.md"; \
	else \
		echo "MATLAB not found on PATH."; \
		echo "NNV requires a MATLAB installation."; \
		echo "Once MATLAB is available, add $$NNV_DIR to your MATLAB path."; \
	fi

install-pyrat: install-submodules  ## Install PyRAT
	@echo "═══ Setting up PyRAT ═══"
	@PYRAT_DIR=src/VeriStressGT/verifiers/pyrat; \
	if [ ! -d "$$PYRAT_DIR" ] || [ -z "$$(ls -A $$PYRAT_DIR 2>/dev/null)" ]; then \
		echo "ERROR: $$PYRAT_DIR not populated. Run: make install-submodules"; \
		exit 1; \
	fi; \
	cd $$PYRAT_DIR && $(PIP) install -e . 2>/dev/null || \
		$(PIP) install . 2>/dev/null || \
		echo "PyRAT install failed. See $$PYRAT_DIR/README.md"

# ── Diagnostics ──────────────────────────────────────────────────────

doctor:  ## Run dependency health check
	$(PYTHON) -m VeriStressGT.cli.doctor

# ── Testing ──────────────────────────────────────────────────────────

test:  ## Run test suite
	$(PYTHON) -m pytest tests/ -v

test-fresh:  ## Simulate a fresh user install via Docker
	docker build -t VeriStressGT-fresh-test -f Dockerfile.test .
	docker run --rm VeriStressGT-fresh-test

# ── Docker ───────────────────────────────────────────────────────────

docker:  ## Build the all-in-one Docker image
	docker build -t VeriStressGT -f Dockerfile .

# ── Cleanup ──────────────────────────────────────────────────────────

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true