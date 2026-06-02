"""Verifier adapter for the AlgebraicVerification HC-based wrapper.

Invokes `python <ALGEBRAIC_VERIFIER_DIR>/verify_vnnlib.py <onnx> <vnnlib> ...`
in the conda env named by $ALGEBRAIC_VERIFIER_CONDA_ENV (defaulting to the
caller's current env if unset). The wrapper requires a .pnn.json sidecar
alongside the ONNX when used with polynomial PNN instances.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from VeriStressGT.verifier_adapters.common import normalize_status_from_text

VERIFIER_NAME = "algebraic_pnn"
CONDA_ENV_VAR = "ALGEBRAIC_VERIFIER_CONDA_ENV"
CONDA_ENV_DEFAULT = None


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--algebraic-verifier-dir",
        default=None,
        help=(
            "Path to the AlgebraicVerification repo root (contains "
            "verify_vnnlib.py). Falls back to $ALGEBRAIC_VERIFIER_DIR."
        ),
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Forwarded to the wrapper (HC itself always runs on CPU).",
    )
    p.add_argument(
        "--algebraic-timeout",
        type=float,
        default=300.0,
        help="Wall-clock seconds for the verifier subprocess.",
    )


def build_cmd(
    args: argparse.Namespace,
    onnx_path: str,
    vnnlib_path: str,
    workdir: Path,
) -> List[str]:
    repo_dir = args.algebraic_verifier_dir or os.environ.get("ALGEBRAIC_VERIFIER_DIR")
    if not repo_dir:
        raise SystemExit(
            "algebraic_pnn adapter: set --algebraic-verifier-dir or "
            "$ALGEBRAIC_VERIFIER_DIR to the AlgebraicVerification repo root."
        )

    script = Path(repo_dir).expanduser().resolve() / "verify_vnnlib.py"
    if not script.exists():
        raise SystemExit(f"algebraic_pnn adapter: verify_vnnlib.py not found at {script}")

    result_file = workdir / "result.txt"
    return [
        sys.executable,
        str(script),
        str(onnx_path),
        str(vnnlib_path),
        "--timeout", str(float(args.algebraic_timeout)),
        "--device", args.device,
        "--result-file", str(result_file),
    ]


def parse_result(stdout: str, stderr: str, rc: int) -> str | None:
    parsed = normalize_status_from_text(stdout)
    if parsed is not None:
        return parsed
    return normalize_status_from_text(stderr)
