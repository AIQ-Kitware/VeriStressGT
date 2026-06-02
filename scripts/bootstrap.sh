#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# bootstrap.sh — one-command setup for VeriStressGT
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/bootstrap.sh              # full setup
#   bash scripts/bootstrap.sh --no-conda   # skip conda/verifier setup
#
# What it does:
#   1. Checks Python version (fails if < 3.9)
#   2. Installs Miniconda if conda is missing
#   3. Pulls git submodules
#   4. Installs VeriStressGT[generate]
#   5. Sets up verifier conda envs (α-β-CROWN, NeuralSAT, nnenum)
#   6. Writes a .env file with all env vars
#   7. Runs VeriStressGT-doctor
#
# After this script, the user is fully set up.
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
err()  { echo -e "  ${RED}✗${NC} $1"; }
step() { echo -e "\n${BOLD}[$1/$TOTAL_STEPS] $2${NC}"; }

SKIP_CONDA=false
FRESH_CONDA=false
if [[ "${1:-}" == "--no-conda" ]]; then
    SKIP_CONDA=true
fi

TOTAL_STEPS=8
if $SKIP_CONDA; then
    TOTAL_STEPS=4
fi

# Find repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo -e "${BOLD}VeriStressGT bootstrap${NC}"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Check Python ─────────────────────────────────────────────
step 1 "Checking Python"

if ! command -v python3 &>/dev/null; then
    err "python3 not found. Install Python 3.9+ first:"
    echo "    https://www.python.org/downloads/"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
    err "Python $PY_VER found, but 3.9+ is required."
    echo "    Install a newer Python: https://www.python.org/downloads/"
    exit 1
fi
ok "Python $PY_VER"

# ── Step 2: Ensure conda ─────────────────────────────────────────────
if ! $SKIP_CONDA; then
    step 2 "Checking conda"

    if command -v conda &>/dev/null; then
        CONDA_PATH=$(which conda)
        ok "conda found at $CONDA_PATH"
    else
        warn "conda not found — installing Miniconda..."
        FRESH_CONDA=true
        echo ""

        MINICONDA_DIR="$HOME/miniconda3"

        # Detect platform
        OS=$(uname -s)
        ARCH=$(uname -m)
        if [ "$OS" = "Darwin" ]; then
            if [ "$ARCH" = "arm64" ]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
            else
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
            fi
        elif [ "$OS" = "Linux" ]; then
            if [ "$ARCH" = "aarch64" ]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
            else
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            fi
        else
            err "Unsupported OS: $OS. Install conda manually: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi

        INSTALLER="/tmp/miniconda_installer.sh"
        echo "    Downloading from $MINICONDA_URL ..."
        curl -fsSL "$MINICONDA_URL" -o "$INSTALLER"
        bash "$INSTALLER" -b -p "$MINICONDA_DIR"
        rm -f "$INSTALLER"

        # Add to current session
        export PATH="$MINICONDA_DIR/bin:$PATH"

        # Add to shell profile if not already there
        SHELL_RC=""
        if [ -f "$HOME/.zshrc" ]; then
            SHELL_RC="$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            SHELL_RC="$HOME/.bashrc"
        elif [ -f "$HOME/.bash_profile" ]; then
            SHELL_RC="$HOME/.bash_profile"
        fi

        if [ -n "$SHELL_RC" ]; then
            if ! grep -q "miniconda3/bin" "$SHELL_RC" 2>/dev/null; then
                echo "" >> "$SHELL_RC"
                echo "# Added by VeriStressGT bootstrap" >> "$SHELL_RC"
                echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "$SHELL_RC"
                ok "Added conda to $SHELL_RC"
            fi
        fi

        # Initialize conda for the current shell
        eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"
        # Initialize conda for future shells
        "$MINICONDA_DIR/bin/conda" init bash >/dev/null 2>&1 || true
        "$MINICONDA_DIR/bin/conda" init zsh >/dev/null 2>&1 || true

        ok "Miniconda installed at $MINICONDA_DIR"
    fi
fi

# ── Step 3: Git submodules ───────────────────────────────────────────
CONDA_STEP=2
if $SKIP_CONDA; then
    step 2 "Pulling git submodules"
else
    step 3 "Pulling git submodules"
fi

if [ ! -f "$REPO_ROOT/.gitmodules" ]; then
    warn "No .gitmodules found — skipping submodule pull"
else
    git submodule update --init --recursive 2>/dev/null && \
        ok "Submodules updated" || \
        warn "Submodule update had issues (some verifiers may be missing)"
fi

# ── Step 4: Create VeriStressGT conda env ──────────────────────
if ! $SKIP_CONDA; then
    step 4 "Creating VeriStressGT conda env"

    if conda env list 2>/dev/null | grep -q "VeriStressGT"; then
        ok "VeriStressGT conda env already exists"
    else
        conda create -n VeriStressGT python=3.11 -y --quiet 2>/dev/null && \
            ok "VeriStressGT conda env created (python 3.11)" || \
            { err "Failed to create conda env"; exit 1; }
    fi

    # Activate it for the rest of this script
    eval "$(conda shell.bash hook)"
    conda activate VeriStressGT
    ok "Activated conda env: VeriStressGT"
fi

# ── Step 5: Install VeriStressGT[generate] ─────────────────────
if $SKIP_CONDA; then
    step 3 "Installing VeriStressGT[generate] (this may take a few minutes)"
else
    step 5 "Installing VeriStressGT[generate] (this may take a few minutes)"
fi

pip install -e '.[generate]' --retries 5 --timeout 120 || \
    { warn "First attempt failed (network timeout?), retrying..."; \
      pip install -e '.[generate]' --retries 5 --timeout 120; }
ok "VeriStressGT[generate] installed"

if $SKIP_CONDA; then
    step 4 "Running doctor"
    echo ""
    VeriStressGT-doctor || true
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Bootstrap complete (--no-conda)${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo ""
    echo "Generate instances:  python -m VeriStressGT.cli.create_benchmark --help"
    echo "Check setup:         VeriStressGT-doctor"
    exit 0
fi

# ── Step 5: Set up verifier conda envs ───────────────────────────────
step 6 "Setting up verifier environments"

# α-β-CROWN
ABCROWN_DIR="$REPO_ROOT/src/VeriStressGT/verifiers/alpha-beta-CROWN"
if [ -d "$ABCROWN_DIR/complete_verifier" ]; then
    echo "  Setting up α-β-CROWN conda env..."
    if conda env list 2>/dev/null | grep -q "alpha-beta-crown"; then
        ok "α-β-CROWN conda env already exists"
    else
        echo "  Creating conda env (this may take a few minutes)..."
        conda env create -f "$ABCROWN_DIR/complete_verifier/environment.yaml" \
            --name alpha-beta-crown 2>&1 && \
            ok "α-β-CROWN conda env created" || \
            warn "α-β-CROWN conda env creation failed (see above)"
    fi
    # Install VeriStressGT into the abcrown env
    echo "  Installing VeriStressGT in alpha-beta-crown env..."
    conda run -n alpha-beta-crown pip install -e . 2>&1 && \
        ok "VeriStressGT installed in alpha-beta-crown env" || \
        warn "Could not install VeriStressGT in alpha-beta-crown env"
    # Install auto_LiRPA from the local submodule (not on PyPI as >=0.4)
    if [ -d "$ABCROWN_DIR/auto_LiRPA" ]; then
        echo "  Installing auto_LiRPA from submodule..."
        conda run -n alpha-beta-crown pip install -e "$ABCROWN_DIR/auto_LiRPA" 2>&1 && \
            ok "auto_LiRPA installed in alpha-beta-crown env" || \
            warn "Could not install auto_LiRPA"
    fi
else
    warn "α-β-CROWN submodule not populated — skipping"
fi

# NeuralSAT
NS_DIR="$REPO_ROOT/src/VeriStressGT/verifiers/neuralsat"
if [ -d "$NS_DIR" ] && [ -n "$(ls -A "$NS_DIR" 2>/dev/null)" ]; then
    if [ -f "$NS_DIR/requirements.txt" ]; then
        pip install -r "$NS_DIR/requirements.txt" --quiet 2>/dev/null && \
            ok "NeuralSAT requirements installed" || \
            warn "NeuralSAT requirements install had issues"
    fi
else
    warn "NeuralSAT submodule not populated — skipping"
fi

# nnenum (own conda env to avoid onnx version conflict with onnxscript)
NNENUM_DIR="$REPO_ROOT/src/VeriStressGT/verifiers/nnenum"
if [ -d "$NNENUM_DIR" ] && [ -n "$(ls -A "$NNENUM_DIR" 2>/dev/null)" ]; then
    if conda env list 2>/dev/null | grep -q "nnenum"; then
        ok "nnenum conda env already exists"
    else
        echo "  Creating nnenum conda env..."
        conda create -n nnenum python=3.10 -y --quiet 2>/dev/null && \
            ok "nnenum conda env created" || \
            warn "nnenum conda env creation failed"
    fi
    echo "  Installing nnenum..."
    conda run -n nnenum pip install -e "$NNENUM_DIR" --quiet 2>/dev/null && \
        ok "nnenum installed (in conda env 'nnenum')" || \
        warn "nnenum install had issues"
else
    warn "nnenum submodule not populated — skipping"
fi

# ── Step 6: Write .env ───────────────────────────────────────────────
step 7 "Writing .env"

ENV_FILE="$REPO_ROOT/.env"
cat > "$ENV_FILE" << EOF
# VeriStressGT environment variables (auto-generated by bootstrap.sh)
# source this file: source .env

export ABCROWN_VNNCOMP2024_DIR=$REPO_ROOT/src/VeriStressGT/verifiers/alpha-beta-CROWN
export ABCROWN_CONDA_ENV=alpha-beta-crown
export NEURALSAT_DIR=$REPO_ROOT/src/VeriStressGT/verifiers/neuralsat
export MARABOU_DIR=$REPO_ROOT/src/VeriStressGT/verifiers/Marabou
EOF

ok "Wrote $ENV_FILE"
echo "    Run: source .env"

# Source it for the current session
source "$ENV_FILE"

# ── Step 7: Doctor ───────────────────────────────────────────────────
step 8 "Running doctor"
echo ""
VeriStressGT-doctor || true

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Bootstrap complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
if $FRESH_CONDA; then
    echo "  Restart your terminal first (conda was just installed)"
fi
echo "  conda activate VeriStressGT                 # activate the env"
echo "  source .env                                      # load env vars"
echo "  VeriStressGT-doctor                         # verify setup"
echo "  python3 -m VeriStressGT.cli.create_benchmark --help"
echo ""