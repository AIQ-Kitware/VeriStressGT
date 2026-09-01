#!/usr/bin/env bash
# Environment-carrying wrapper for aiq/mini_sweep_runner.py.
#
# New file. cards/evaluation.yaml and mini_sweep_runner.py are untouched.
#
# WHY THIS EXISTS. Under the legacy route MAGNET runs the node in a process that
# inherited the caller's environment, so exporting PATH and PYTHONPATH in the
# launching script was enough. Under kwdagger the node runs in a **cmd_queue
# tmux worker, which inherits nothing**. The exports were still printed by the
# orchestrator's preflight -- "VeriStressGT ok" -- while the node itself could
# not import VeriStressGT, so every verifier was skipped and the card saw
# correct_fraction: None for all three.
#
# magnet/containers.py already names this trap for the containerized path
# (DEFAULT_CAPTURED_ENV: "PYTHONPATH ... left as a bare name it arrives empty in
# a cmd_queue tmux worker that did not inherit it"). The host path has the same
# problem and no equivalent mechanism, so the node command carries its own.
#
# Everything is derived from this script's own location, because nothing about
# the caller survives the worker boundary.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB="$(cd "$HERE/.." && pwd)"

# The package itself, plus nnenum's source root: both conda envs carry editable
# installs pointing at the machine they were built on. See SMELL-VST-05.
PP="$SUB/src:$SUB"
[[ -d "$SUB/src/VeriStressGT/verifiers/nnenum/src" ]] && \
    PP="$PP:$SUB/src/VeriStressGT/verifiers/nnenum/src"
export PYTHONPATH="$PP${PYTHONPATH:+:$PYTHONPATH}"

# A relocated miniconda's own `conda` launcher does not execute (dead shebang),
# so a shim is required. Resolution order: an inherited value if the worker did
# get one, then a file the runner script drops beside this wrapper, then the
# known mount.
CONDA_ROOT="${VST_CONDA_ROOT:-}"
if [[ -z "$CONDA_ROOT" && -f "$HERE/.conda_root" ]]; then
    CONDA_ROOT="$(<"$HERE/.conda_root")"
fi
CONDA_ROOT="${CONDA_ROOT:-/data/Public/AIQ/tmp-ben-aiq-dry-run-data/miniconda3}"
export CONDA_SHIM_ROOT="$CONDA_ROOT"

SHIM_DIR="${VST_SHIM_DIR:-$HERE/.shim}"
mkdir -p "$SHIM_DIR"
if [[ ! -x "$SHIM_DIR/conda" ]]; then
    cat > "$SHIM_DIR/conda" <<'SHIM'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "run" ]]; then
    echo "[conda-shim] only 'conda run' is emulated; got: $*" >&2; exit 127
fi
shift
ENV_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)  ENV_NAME="$2"; shift 2 ;;
        -p|--prefix) ENV_NAME=""; PREFIX="$2"; shift 2 ;;
        --no-capture-output|--live-stream) shift ;;
        --cwd)      cd "$2"; shift 2 ;;
        *) break ;;
    esac
done
PREFIX="${PREFIX:-$CONDA_SHIM_ROOT/envs/$ENV_NAME}"
[[ -d "$PREFIX" ]] || { echo "[conda-shim] no such env prefix: $PREFIX" >&2; exit 127; }
export PATH="$PREFIX/bin:$PATH"
export CONDA_PREFIX="$PREFIX"
CMD="$1"; shift
TARGET="$PREFIX/bin/$CMD"
if [[ -x "$TARGET" ]] && head -1 "$TARGET" | grep -q '^#!.*python'; then
    exec "$PREFIX/bin/python" "$TARGET" "$@"
fi
[[ -x "$TARGET" ]] && exec "$TARGET" "$@"
exec "$CMD" "$@"
SHIM
    chmod +x "$SHIM_DIR/conda"
fi
export PATH="$SHIM_DIR:$PATH"

export ABCROWN_VNNCOMP2024_DIR="${ABCROWN_VNNCOMP2024_DIR:-$SUB/src/VeriStressGT/verifiers/alpha-beta-CROWN}"
export ABCROWN_CONDA_ENV="${ABCROWN_CONDA_ENV:-alpha-beta-crown}"

VST_PY="${VST_PYTHON:-$CONDA_ROOT/envs/VeriStressGT/bin/python}"
[[ -x "$VST_PY" ]] || { echo "[wrapper] no VeriStressGT interpreter at $VST_PY" >&2; exit 127; }

# Fail here rather than 50 instances later with every verifier "skipped".
"$VST_PY" -c "import VeriStressGT" 2>/dev/null || {
    echo "[wrapper] VeriStressGT not importable under $VST_PY" >&2
    echo "[wrapper] PYTHONPATH=$PYTHONPATH" >&2
    exit 127
}

exec "$VST_PY" -u "$SUB/aiq/mini_sweep_runner.py" "$@"
