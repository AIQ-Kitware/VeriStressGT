"""
VeriStressGT doctor — one-command health check.

Reports the status of every dependency (pip packages, verifier repos,
env vars, MATLAB) so a new user can see exactly what's ready and what
needs fixing.

Usage:
    VeriStressGT-doctor          # from pip entry point
    python -m VeriStressGT.cli.doctor   # module invocation
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ── Formatting helpers ───────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _ok(msg: str) -> str:
    return f"  {_GREEN}✓{_RESET} {msg}"

def _warn(msg: str) -> str:
    return f"  {_YELLOW}⚠{_RESET} {msg}"

def _fail(msg: str) -> str:
    return f"  {_RED}✗{_RESET} {msg}"

def _section(title: str) -> str:
    return f"\n{_BOLD}{title}{_RESET}"


# ── Check helpers ────────────────────────────────────────────────────────────

def _check_module(name: str, extra: Optional[str] = None) -> tuple[bool, str]:
    """Try to import *name*; return (ok, message)."""
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        return True, _ok(f"{name} {ver}")
    except ImportError:
        hint = f"pip install 'VeriStressGT[{extra}]'" if extra else f"pip install {name}"
        return False, _fail(f"{name}  →  {hint}")
    except Exception as e:
        # Catches version conflicts, beartype errors, etc. that blow up on import
        short = type(e).__name__
        return False, _fail(f"{name} — import error: {short} (likely version conflict; use its own conda env)")


def _check_env_var(var: str) -> tuple[bool, str]:
    val = os.environ.get(var)
    if val:
        return True, _ok(f"${var} = {val}")
    return False, _warn(f"${var} not set")


def _check_binary(name: str) -> tuple[bool, str]:
    path = shutil.which(name)
    if path:
        return True, _ok(f"{name} → {path}")
    return False, _fail(f"{name} not found on PATH")


def _check_conda_env(env_name: str) -> tuple[bool, str]:
    """Check if a named conda environment exists."""
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            envs = data.get("envs", [])
            for env_path in envs:
                if Path(env_path).name == env_name:
                    return True, _ok(f"conda env '{env_name}' exists → {env_path}")
            return False, _fail(f"conda env '{env_name}' not found (run: make install-abcrown)")
        return False, _warn(f"conda env list failed")
    except FileNotFoundError:
        return False, _fail("conda not installed")
    except Exception as e:
        return False, _warn(f"conda env check failed: {e}")


def _check_module_in_conda_env(env_name: str, module_name: str) -> tuple[bool, str]:
    """Check if a module is importable inside a specific conda env."""
    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c",
             f"import {module_name}; v=getattr({module_name},'__version__','?'); print(v)"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            ver = result.stdout.strip().split("\n")[-1]
            return True, _ok(f"{module_name} {ver} (in conda env '{env_name}')")
        # Extract last meaningful line from stderr for diagnosis
        err_lines = [l.strip() for l in (result.stderr or result.stdout or "").strip().splitlines() if l.strip()]
        short_err = err_lines[-1] if err_lines else "unknown error"
        # Truncate long errors
        if len(short_err) > 120:
            short_err = short_err[:120] + "..."
        return False, _fail(f"{module_name} not importable in conda env '{env_name}'\n         {short_err}")
    except FileNotFoundError:
        return False, _fail("conda not installed")
    except subprocess.TimeoutExpired:
        return False, _warn(f"{module_name} check timed out in conda env '{env_name}'")
    except Exception as e:
        return False, _warn(f"conda check failed: {e}")


def _check_submodule(rel_path: str, label: str) -> tuple[bool, str]:
    """Check if a verifier submodule directory exists and is populated."""
    # Walk up from this file to find the repo root (look for pyproject.toml)
    here = Path(__file__).resolve().parent
    candidates = [here.parent.parent.parent, here.parent.parent, Path.cwd()]
    root = None
    for c in candidates:
        if (c / "pyproject.toml").exists():
            root = c
            break
    if root is None:
        return False, _warn(f"{label}: cannot locate repo root")

    target = root / rel_path
    if not target.exists():
        return False, _fail(f"{label}: {rel_path} not found (run: git submodule update --init --recursive)")
    # Check it's not empty (common submodule issue)
    children = list(target.iterdir()) if target.is_dir() else []
    if len(children) < 2:
        return False, _fail(f"{label}: {rel_path} exists but looks empty (run: git submodule update --init --recursive)")
    return True, _ok(f"{label}: {rel_path}")


def _check_cuda() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, _ok(f"CUDA available: {name}")
        return False, _warn("CUDA not available (CPU-only torch)")
    except ImportError:
        return False, _warn("torch not installed; cannot check CUDA")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_doctor() -> int:
    """Run all checks and return 0 if everything essential passes."""
    print(f"\n{_BOLD}VeriStressGT doctor{_RESET}")
    print("=" * 50)

    all_ok = True
    warnings = 0

    # ── Prerequisites ────────────────────────────────────────────────────
    print(_section("Prerequisites"))

    # Python version
    py_ver = sys.version_info
    if py_ver >= (3, 9):
        print(_ok(f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}"))
    else:
        print(_fail(f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro} (requires 3.9+)"))
        all_ok = False

    # git
    ok, msg = _check_binary("git")
    print(msg)
    if not ok:
        all_ok = False

    # conda
    ok_conda_bin, msg = _check_binary("conda")
    print(msg)
    if not ok_conda_bin:
        print(f"       (required for running verifiers in isolated envs)")
        warnings += 1

    # ── Core (required) ──────────────────────────────────────────────────
    print(_section("Core dependencies"))
    for mod in ["numpy", "yaml", "packaging"]:
        ok, msg = _check_module(mod)
        print(msg)
        if not ok:
            all_ok = False

    # ── Instance generation (optional) ───────────────────────────────────
    print(_section("Instance generation  [pip install '.[generate]']"))
    for mod in ["torch", "onnx", "onnxruntime", "onnxsim"]:
        ok, msg = _check_module(mod, extra="generate")
        print(msg)
        if not ok:
            warnings += 1

    ok, msg = _check_cuda()
    print(msg)

    # ── MILP constructions ───────────────────────────────────────────────
    print(_section("MILP constructions  [pip install '.[milp]']"))
    ok, msg = _check_module("gurobipy", extra="milp")
    print(msg)

    # ── Verifier readiness (per-verifier deep checks) ──────────────────
    print(_section("Verifier readiness"))

    # --- α-β-CROWN ---
    print(f"\n  {_BOLD}α-β-CROWN{_RESET}  [make install-abcrown]")
    ok_sub, msg_sub = _check_submodule("src/VeriStressGT/verifiers/alpha-beta-CROWN", "repo")
    print(f"  {msg_sub}")
    ok_env, msg_env = _check_env_var("ABCROWN_VNNCOMP2024_DIR")
    print(f"  {msg_env}")
    ok_conda_env, msg_conda_env = _check_env_var("ABCROWN_CONDA_ENV")
    print(f"  {msg_conda_env}")
    # Check if the conda env actually exists
    conda_env_name = os.environ.get("ABCROWN_CONDA_ENV", "alpha-beta-crown")
    ok_conda, msg_conda = _check_conda_env(conda_env_name)
    print(f"  {msg_conda}")
    # Check auto_LiRPA inside the conda env (it can't coexist in the base env)
    if ok_conda:
        ok_lirpa, msg_lirpa = _check_module_in_conda_env(conda_env_name, "auto_LiRPA")
    else:
        ok_lirpa, msg_lirpa = False, _fail("auto_LiRPA (skipped — conda env not found)")
    print(f"  {msg_lirpa}")
    if all([ok_sub, ok_env, ok_conda_env, ok_conda, ok_lirpa]):
        print(f"    {_GREEN}→ Ready{_RESET}")
    else:
        print(f"    {_YELLOW}→ Not ready{_RESET}")
        warnings += 1

    # --- NeuralSAT ---
    print(f"\n  {_BOLD}NeuralSAT{_RESET}  [make install-neuralsat]")
    ok_sub, msg = _check_submodule("src/VeriStressGT/verifiers/neuralsat", "repo")
    print(f"  {msg}")
    ok_env, msg = _check_env_var("NEURALSAT_DIR")
    print(f"  {msg}")
    # Check if key NeuralSAT deps are importable
    ns_deps_ok = True
    for dep in ["onnx2pytorch"]:
        ok_dep, msg = _check_module(dep)
        print(f"  {msg}")
        if not ok_dep:
            ns_deps_ok = False
    if all([ok_sub, ok_env, ns_deps_ok]):
        print(f"    {_GREEN}→ Ready{_RESET}")
    else:
        print(f"    {_YELLOW}→ Not ready{_RESET}")
        warnings += 1

    # --- nnenum ---
    print(f"\n  {_BOLD}nnenum{_RESET}  [make install-nnenum]")
    ok_sub, msg = _check_submodule("src/VeriStressGT/verifiers/nnenum", "repo")
    print(f"  {msg}")
    ok_mod, msg = _check_module("nnenum")
    print(f"  {msg}")
    if ok_sub and ok_mod:
        print(f"    {_GREEN}→ Ready{_RESET}")
    else:
        print(f"    {_YELLOW}→ Not ready{_RESET}")
        warnings += 1

    # --- Marabou ---
    print(f"\n  {_BOLD}Marabou{_RESET}  [see Marabou README for cmake build]")
    ok_sub, msg = _check_submodule("src/VeriStressGT/verifiers/Marabou", "repo")
    print(f"  {msg}")
    ok_env, msg = _check_env_var("MARABOU_DIR")
    print(f"  {msg}")
    ok_mod, msg = _check_module("maraboupy")
    print(f"  {msg}")
    if all([ok_sub, ok_env, ok_mod]):
        print(f"    {_GREEN}→ Ready{_RESET}")
    else:
        print(f"    {_YELLOW}→ Not ready (requires cmake build){_RESET}")
        warnings += 1

    # --- NNV ---
    print(f"\n  {_BOLD}NNV{_RESET}  [requires MATLAB]")
    ok_sub, msg = _check_submodule("src/VeriStressGT/verifiers/nnv", "repo")
    print(f"  {msg}")
    ok_matlab, msg = _check_binary("matlab")
    print(f"  {msg}")
    if ok_sub and ok_matlab:
        print(f"    {_GREEN}→ Ready{_RESET}")
    else:
        print(f"    {_YELLOW}→ Not ready{_RESET}")
        # Don't count as warning — MATLAB is a special case

    # --- PyRAT ---
    print(f"\n  {_BOLD}PyRAT{_RESET}")
    ok_sub, msg = _check_submodule("src/VeriStressGT/verifiers/pyrat", "repo")
    print(f"  {msg}")
    ok_mod, msg = _check_module("pyrat")
    print(f"  {msg}")
    if ok_sub and ok_mod:
        print(f"    {_GREEN}→ Ready{_RESET}")
    else:
        print(f"    {_YELLOW}→ Not ready{_RESET}")
        warnings += 1

    # ── Constructor discovery ────────────────────────────────────────────
    print(_section("Constructor discovery"))
    try:
        from VeriStressGT.registry.constructions import discover_constructions
        cons = discover_constructions()
        if cons:
            print(_ok(f"Found {len(cons)} constructor(s): {', '.join(sorted(cons.keys()))}"))
        else:
            print(_warn("No constructors discovered (torch may not be installed)"))
    except Exception as e:
        print(_fail(f"Discovery failed: {e}"))

    # ── Verifier adapter discovery ───────────────────────────────────────
    print(_section("Verifier adapter discovery"))
    try:
        from VeriStressGT.registry.verifiers import discover_verifiers
        vers = discover_verifiers()
        if vers:
            print(_ok(f"Found {len(vers)} adapter(s): {', '.join(sorted(vers.keys()))}"))
        else:
            print(_warn("No verifier adapters discovered"))
    except Exception as e:
        print(_fail(f"Discovery failed: {e}"))

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if all_ok and warnings == 0:
        print(f"{_GREEN}All checks passed!{_RESET}")
    elif all_ok:
        print(f"{_YELLOW}Core OK, {warnings} optional component(s) need attention.{_RESET}")
    else:
        print(f"{_RED}Some core dependencies are missing.{_RESET}")
    print()
    return 0 if all_ok else 1


def main():
    sys.exit(run_doctor())


if __name__ == "__main__":
    main()