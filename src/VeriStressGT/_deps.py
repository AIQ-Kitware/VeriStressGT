"""
Lazy import helpers for optional dependencies.

Usage in any module:
    from VeriStressGT._deps import require
    torch = require("torch", extra="generate")
    # raises a clear error if torch isn't installed, telling the user
    # exactly which pip extra to install.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional


# Maps module names → the pip extra that provides them.
_EXTRA_FOR_MODULE: dict[str, str] = {
    "torch":        "generate",
    "onnx":         "generate",
    "onnxruntime":  "generate",
    "onnxscript":   "generate",
    "onnx2pytorch": "generate",
    "gurobipy":     "milp",
    "onnxsim":      "generate",
}


def require(module_name: str, *, extra: Optional[str] = None) -> ModuleType:
    """Import *module_name* or raise a helpful error.

    Parameters
    ----------
    module_name : str
        Top-level module to import (e.g. ``"torch"``).
    extra : str, optional
        The pip extra that provides this module.  If omitted, looked up
        from ``_EXTRA_FOR_MODULE``.  If still unknown, the error message
        omits the extra hint.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        extra = extra or _EXTRA_FOR_MODULE.get(module_name)
        if extra:
            hint = (
                f"Missing optional dependency '{module_name}'.\n"
                f"Install it with:  pip install 'VeriStressGT[{extra}]'"
            )
        else:
            hint = (
                f"Missing dependency '{module_name}'.\n"
                f"Install it with:  pip install {module_name}"
            )
        raise ImportError(hint) from None


def is_available(module_name: str) -> bool:
    """Check if a module is importable without raising."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False