from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

VERIFIER_NAME = "pyrat"
CONDA_ENV_DEFAULT = "pyrat"

_PYRAT_RESULT_RE = re.compile(r"(?im)\bResult\s*=\s*(True|False)\b")
_PYRAT_TIMEOUT_RE = re.compile(r"(?im)\b(timeout|timed out|time limit)\b")
_PYRAT_UNKNOWN_RE = re.compile(r"(?im)\bResult\s*=\s*Unknown\b")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--pyrat_bin",
        default="pyrat",
        help="Path to the PyRAT executable (default: pyrat).",
    )

    p.add_argument(
        "--pyrat_device",
        default="cpu",
        help="Device: cpu or cuda (default: cuda).",
    )

    p.add_argument(
        "--pyrat_library",
        default="torch",
        help="Computing library: numpy or torch (default: torch). Must be 'torch' for cuda.",
    )

    p.add_argument(
        "--pyrat_domains",
        nargs="+",
        default=["poly"],
        help="PyRAT domains (default: con_z). Use 'con_z' for ReLU BaB support.",
    )

    p.add_argument(
        "--pyrat_timeout",
        type=int,
        default=None,
        help="Timeout for PyRAT in seconds.",
    )

    p.add_argument(
        "--pyrat_split_relu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ReLU branch-and-bound (default: enabled).",
    )

    p.add_argument(
        "--pyrat_split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable input splitting (default: disabled).",
    )

    p.add_argument(
        "--pyrat_split_heuristic",
        default="better",
        help="Split heuristic (default: better).",
    )

    p.add_argument(
        "--pyrat_extra",
        nargs=argparse.REMAINDER,
        help="Extra raw args to pass directly to PyRAT.",
    )


def build_cmd(
    args: argparse.Namespace,
    onnx_path: str,
    vnnlib_path: str,
    workdir: Path,
) -> List[str]:
    cmd = [
        str(Path(args.pyrat_bin).expanduser()) if "/" in args.pyrat_bin else args.pyrat_bin,
        "--model_path",
        onnx_path,
        "--property_path",
        vnnlib_path,
    ]

    if args.pyrat_domains:
        cmd += ["--domains"] + list(args.pyrat_domains)

    if args.pyrat_timeout is not None:
        cmd += ["--timeout", str(args.pyrat_timeout)]

    if args.pyrat_split_relu:
        cmd += ["--split_relu", "True"]

    if args.pyrat_split:
        cmd += ["--split", "True"]

    if args.pyrat_split_heuristic:
        cmd += ["--split_heuristic", str(args.pyrat_split_heuristic)]

    if args.pyrat_device:
        cmd += ["--device", str(args.pyrat_device)]


    if args.pyrat_library:
        cmd += ["--library", args.pyrat_library]


    if args.pyrat_extra:
        cmd += list(args.pyrat_extra)

    return cmd


def parse_result(stdout: str, stderr: str, rc: int) -> str | None:
    text = (stdout or "") + "\n" + (stderr or "")

    m = _PYRAT_RESULT_RE.search(text)
    if m:
        token = m.group(1).lower()
        if token == "true":
            return "UNSAT"
        if token == "false":
            return "SAT"

    if _PYRAT_UNKNOWN_RE.search(text):
        return "UNKNOWN"

    if _PYRAT_TIMEOUT_RE.search(text):
        return "TIMEOUT"

    return None
