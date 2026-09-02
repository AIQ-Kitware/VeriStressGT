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
# Roots to search, in order. VST_CONDA_ROOTS lets a caller add the
# interpreter's own prefix, which is where an image may keep verifiers
# that have no conda env of their own.
export CONDA_SHIM_ROOTS="${VST_CONDA_ROOTS:-$CONDA_ROOT}"

SHIM_DIR="${VST_SHIM_DIR:-$HERE/.shim}"
mkdir -p "$SHIM_DIR"
# ALWAYS rewritten. Writing it only when absent kept a stale single-root shim
# from 2026-08-31 alive in the checkout, which the container then mounted and
# used: abcrown 49/50, pyrat and nnenum 0/50, "no such env prefix" -- the
# exact defect the multi-root search below had already fixed in this file.
# The shim is derived from this script, so it must never outlive an edit.
_shim_tmp="$(mktemp "$SHIM_DIR/conda.XXXXXX")"
{
    cat > "$_shim_tmp" <<'SHIM'
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
# Search every root, not one. The verifiers do NOT all live in the same conda
# tree: in the container image alpha-beta-crown has its own prefix
# (/opt/abcrown-conda) while pyrat and nnenum are pip-installed into the base
# interpreter (/opt/conda). Resolving them all under one root scored both of
# them 0/50 with zero timeouts -- every instance errored, which reads like a
# verifier failure and is really a path.
if [[ -z "${PREFIX:-}" && -n "$ENV_NAME" ]]; then
    for _root in ${CONDA_SHIM_ROOTS//:/ }; do
        if [[ -d "$_root/envs/$ENV_NAME" ]]; then
            PREFIX="$_root/envs/$ENV_NAME"; break
        fi
    done
fi

if [[ -n "${PREFIX:-}" && -d "$PREFIX" ]]; then
    export PATH="$PREFIX/bin:$PATH"
    export CONDA_PREFIX="$PREFIX"
else
    # No env of that name in any root. The package may still be installed into
    # the interpreter already on PATH, which is how the image carries pyrat and
    # nnenum. Fall through ONLY if the command actually resolves -- a genuinely
    # missing verifier must still fail loudly rather than score zero. This is
    # the SMELL-VST-05 failure mode and it must stay noisy.
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[conda-shim] no env '$ENV_NAME' under {${CONDA_SHIM_ROOTS}}," >&2
        echo "[conda-shim] and '$1' is not on PATH either." >&2
        exit 127
    fi
    echo "[conda-shim] no env '$ENV_NAME'; using '$1' from PATH" >&2
fi
CMD="$1"; shift
# `${PREFIX:-}` because the fall-through above leaves it unset, and `set -u`
# would abort here rather than run the command that was found on PATH.
TARGET="${PREFIX:-}/bin/$CMD"
if [[ -n "${PREFIX:-}" && -x "$TARGET" ]]; then
    if head -1 "$TARGET" | grep -q '^#!.*python'; then
        exec "$PREFIX/bin/python" "$TARGET" "$@"
    fi
    exec "$TARGET" "$@"
fi
exec "$CMD" "$@"
SHIM
    chmod 755 "$_shim_tmp"
    mv -f "$_shim_tmp" "$SHIM_DIR/conda"
}
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
