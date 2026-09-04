# syntax=docker/dockerfile:1.7
# VeriStressGT: the image the mini-sweep card's node runs in.
#
# The card's kwdagger DAG has a single node, mini_sweep, whose command is
# `bash aiq/run_mini_sweep_env.sh`. That node runs as one `docker run` of this
# image, with this checkout bind-mounted at its own absolute path and the
# working directory set there. The copy baked below provides the environment
# (torch, the verifier stacks, the alpha-beta-CROWN conda env); the mounted
# checkout provides the code, so editing the card or the runner does not mean
# rebuilding.
#
# Before building, the verifier submodules must be populated:
#   git submodule update --init --recursive
#
# Build, from the repository root:
#   docker build -t veristressgt-gpu .
#
# MAGNET_REF is the aiq-magnet commit the evaluator runs against. It is
# on AIQ-Kitware/aiq-magnet main (the kwdagger execution merge, PR #94);
# `--build-arg MAGNET_REF=main` builds against the tip of main instead.
#
# KNOWN GAP: alpha-beta-CROWN gets its own conda env here. pyrat and nnenum are
# plain pip installs of their submodule directories. In the last build checked
# (2026-09-02) nnenum and swiglpk import, but pyrat's console script fails with
# "No module named 'pyrat.main'": the editable install registers a namespace
# package, not the real one under verifiers/pyrat/pyrat. A run from this image
# alone therefore scores pyrat 0/50. See docs/containerized_evaluation.md.

# ── Stage 1: the alpha-beta-CROWN conda env ───────────────────────────────────
# alpha-beta-CROWN invokes itself through `conda run -n alpha-beta-crown`, so
# it needs a real conda env built from its own environment.yaml. mambaforge
# builds it; the final stage copies the finished env in.
FROM condaforge/mambaforge:24.3.0-0 AS abcrown-env-builder

COPY . /opt/src/VeriStressGT
WORKDIR /opt/src/VeriStressGT

RUN --mount=type=cache,target=/opt/conda/pkgs \
    ABCROWN_DIR=src/VeriStressGT/verifiers/alpha-beta-CROWN && \
    ENV_YAML="$ABCROWN_DIR/complete_verifier/environment.yaml" && \
    if [ ! -f "$ENV_YAML" ]; then \
        echo "ERROR: $ENV_YAML not found. Run: git submodule update --init --recursive"; \
        exit 1; \
    fi && \
    mamba env create -n alpha-beta-crown -f "$ENV_YAML" && \
    conda run -n alpha-beta-crown pip install --no-cache-dir \
        -e "$ABCROWN_DIR/auto_LiRPA" && \
    conda clean -afy

# ── Stage 2: the node image ───────────────────────────────────────────────────
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS final

ARG MAGNET_REF=5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv \
    # alpha-beta-CROWN locations inside the image; the evaluator overrides
    # ABCROWN_VNNCOMP2024_DIR with the mounted checkout's path at run time.
    ABCROWN_VNNCOMP2024_DIR=/opt/src/VeriStressGT/src/VeriStressGT/verifiers/alpha-beta-CROWN \
    ABCROWN_CONDA_ENV=alpha-beta-crown \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        git \
        jq \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The conda env comes in at ITS OWN prefix, not /opt/conda: the base image
# keeps its interpreter there, and copying mambaforge over it would replace
# Python 3.11 and everything installed with it, torch included.
COPY --from=abcrown-env-builder /opt/conda /opt/abcrown-conda

# So the base image's conda still finds the env by name for
# `conda run -n alpha-beta-crown`.
ENV CONDA_ENVS_PATH=/opt/abcrown-conda/envs \
    ABCROWN_CONDA_ROOT=/opt/abcrown-conda

RUN python -m pip install --no-cache-dir --upgrade uv

WORKDIR /opt/src

# MAGNET first, pinned, so source edits below never rebuild it. kwdagger,
# cmd_queue and infer-stack arrive through magnet's own dependency list.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        "aiq-magnet[optional] @ git+https://github.com/AIQ-Kitware/aiq-magnet@${MAGNET_REF}"

COPY . /opt/src/VeriStressGT
WORKDIR /opt/src/VeriStressGT

# VeriStressGT with its pip-installable extras.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        'scriptconfig>=0.8' \
        'ubelt>=1.3' \
        'matplotlib>=3.7' \
        'onnxruntime>=1.16' \
        'sortedcontainers' \
        'coloredlogs' \
        'termcolor' \
        'beartype' && \
    uv pip install --system -e '.[generate]'

# nnenum and pyrat from their submodule directories. Both steps are allowed to
# fail so the image still builds for abcrown; a failure prints here rather
# than being hidden. An install that succeeds is not yet a verifier that
# runs: see the KNOWN GAP above for pyrat.
RUN --mount=type=cache,target=/root/.cache/uv \
    NNENUM_DIR=src/VeriStressGT/verifiers/nnenum && \
    if [ -f "$NNENUM_DIR/setup.py" ] || [ -f "$NNENUM_DIR/pyproject.toml" ]; then \
        uv pip install --system -e "$NNENUM_DIR" || echo "[nnenum] INSTALL FAILED; nnenum will score 0/50"; \
    else \
        echo "[nnenum] submodule not populated; nnenum will score 0/50"; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    PYRAT_DIR=src/VeriStressGT/verifiers/pyrat && \
    if [ -f "$PYRAT_DIR/setup.py" ] || [ -f "$PYRAT_DIR/pyproject.toml" ]; then \
        uv pip install --system -e "$PYRAT_DIR" || echo "[pyrat] INSTALL FAILED; pyrat will score 0/50"; \
    else \
        echo "[pyrat] submodule not populated; pyrat will score 0/50"; \
    fi

# Gurobi, for the milp.exact_radius construction when a licence is mounted.
# The committed mini_sweep_bench already holds the 50 instances, so the card
# never needs it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system 'gurobipy' || echo "[gurobipy] install skipped"

WORKDIR /opt/src/VeriStressGT
CMD ["bash"]
