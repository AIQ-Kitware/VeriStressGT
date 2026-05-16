# src/VeriStressGT/registry/constructions.py
from __future__ import annotations
import importlib
import pkgutil
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

import VeriStressGT.robust_constructions as rc_pkg


@dataclass(frozen=True)
class Construction:
    name: str
    module: Any
    add_args: Callable
    run: Callable


def discover_constructions(*, verbose: bool = False) -> Dict[str, Construction]:
    """
    Auto-import all modules under VeriStressGT.robust_constructions.*
    and collect ones that define CONSTRUCTION_NAME, add_args, run.

    Modules that fail to import due to missing optional dependencies
    (e.g. torch) are skipped with a single summary warning rather than
    per-module noise.  Pass verbose=True for detailed error output.
    """
    out: Dict[str, Construction] = {}
    skipped_missing_dep: List[str] = []
    skipped_error: List[tuple[str, str]] = []

    for modinfo in pkgutil.walk_packages(rc_pkg.__path__, rc_pkg.__name__ + "."):
        try:
            mod = importlib.import_module(modinfo.name)
        except ImportError as e:
            err_msg = str(e)
            # Categorize: missing optional dep vs structural issue vs actual bug
            missing_libs = ["torch", "onnx", "onnxruntime", "gurobipy",
                            "auto_LiRPA", "onnxscript", "beartype"]
            is_missing_dep = any(lib in err_msg for lib in missing_libs)
            is_missing_utils = "VeriStressGT.utils" in err_msg
            if is_missing_dep:
                skipped_missing_dep.append(modinfo.name)
            elif is_missing_utils:
                skipped_error.append((modinfo.name, err_msg))
            else:
                skipped_error.append((modinfo.name, err_msg))
            if verbose:
                print(f"[discover_constructions] skip {modinfo.name}: {e}")
            continue
        except Exception as e:
            skipped_error.append((modinfo.name, str(e)))
            if verbose:
                print(f"[discover_constructions] failed {modinfo.name}: {e}")
            continue

        if not hasattr(mod, "CONSTRUCTION_NAME"):
            continue
        if not hasattr(mod, "add_args"):
            continue
        if not hasattr(mod, "run"):
            continue

        name = getattr(mod, "CONSTRUCTION_NAME")
        out[name] = Construction(
            name=name,
            module=mod,
            add_args=getattr(mod, "add_args"),
            run=getattr(mod, "run"),
        )

    # Print a single helpful summary if things were skipped
    if skipped_missing_dep and not out:
        print(
            f"[VeriStressGT] {len(skipped_missing_dep)} constructor module(s) "
            f"skipped due to missing dependencies (torch, onnx, etc.).\n"
            f"  To install:  pip install 'VeriStressGT[generate]'",
            file=sys.stderr,
        )
    if skipped_error:
        # Check if it's the common "missing utils/__init__.py" pattern
        utils_errors = [n for n, e in skipped_error if "VeriStressGT.utils" in e]
        if utils_errors:
            print(
                f"[VeriStressGT] {len(utils_errors)} constructor(s) failed because "
                f"VeriStressGT.utils is not importable.\n"
                f"  Fix:  touch src/VeriStressGT/utils/__init__.py  (then reinstall)",
                file=sys.stderr,
            )
        for name, err in skipped_error:
            if "VeriStressGT.utils" not in err:
                print(f"[VeriStressGT] ERROR importing {name}: {err}", file=sys.stderr)

    return out
