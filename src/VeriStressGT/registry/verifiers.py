# src/VeriStressGT/registry/verifiers.py
from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import VeriStressGT.verifier_adapters as va_pkg


@dataclass(frozen=True)
class Verifier:
    name: str
    module: Any
    add_args: Callable
    build_cmd: Callable
    parse_result: Optional[Callable]
    conda_env_var: Optional[str] = None       # e.g. "ABCROWN_CONDA_ENV"
    conda_env_default: Optional[str] = None   # e.g. "alpha-beta-crown"

    def resolve_conda_env(self) -> Optional[str]:
        """Return the conda env name this verifier should run in, or None."""
        if self.conda_env_var:
            return os.environ.get(self.conda_env_var, self.conda_env_default)
        return self.conda_env_default


def discover_verifiers(*, verbose: bool = False) -> Dict[str, Verifier]:
    out: Dict[str, Verifier] = {}

    for modinfo in pkgutil.walk_packages(va_pkg.__path__, va_pkg.__name__ + "."):
        try:
            mod = importlib.import_module(modinfo.name)
        except Exception as e:
            if verbose:
                print(f"[discover_verifiers] skip {modinfo.name}: {e}", file=sys.stderr)
            continue

        if not hasattr(mod, "VERIFIER_NAME"):
            continue
        if not hasattr(mod, "add_args"):
            continue
        if not hasattr(mod, "build_cmd"):
            continue

        name = getattr(mod, "VERIFIER_NAME")
        out[name] = Verifier(
            name=name,
            module=mod,
            add_args=getattr(mod, "add_args"),
            build_cmd=getattr(mod, "build_cmd"),
            parse_result=getattr(mod, "parse_result", None),
            conda_env_var=getattr(mod, "CONDA_ENV_VAR", None),
            conda_env_default=getattr(mod, "CONDA_ENV_DEFAULT", None),
        )
    return out