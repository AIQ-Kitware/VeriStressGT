from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

from .common import normalize_status_from_text

VERIFIER_NAME = "nnenum"


# ---------------------------------------------------------------------
# nnenum output patterns
# ---------------------------------------------------------------------

# Common success messages.
_NNENUM_SAFE_BEFORE_ENUM_RE = re.compile(
    r"(?im)\bproven\s+safe\s+before\s+enumerate\b"
)

_NNENUM_PROVEN_SAFE_RE = re.compile(
    r"(?im)\b(?:proven\s+safe|network\s+is\s+safe|property\s+holds|verified)\b"
)

_NNENUM_UNSAT_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:unsat|safe|verified)\s*$"
)

# Common counterexample / violation messages.
_NNENUM_UNSAFE_RE = re.compile(
    r"(?im)\b(?:unsafe|counterexample|counter-example|violation|violated|falsified)\b"
)

_NNENUM_SAT_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:sat|unsafe)\s*$"
)

# Timeout / killed process messages.
_NNENUM_TIMEOUT_RE = re.compile(
    r"(?im)\b(?:timeout|timed\s+out|time\s+limit\s+reached|killed|terminated)\b"
)

# Actual errors.
_NNENUM_ERROR_RE = re.compile(
    r"(?im)\b(?:error|exception|traceback|not\s+implemented|unsupported|failed)\b"
)


def parse_result(stdout: str, stderr: str, rc: int) -> str | None:
    stdout = stdout or ""
    stderr = stderr or ""
    text = stdout + "\n" + stderr

    # If nnenum produced literally no usable output, treat it as timeout.
    # This catches silent outer wall-clock kills / hangs.
    if not text.strip():
        return "TIMEOUT"

    # nnenum sometimes prints this instead of a clean UNSAT line.
    if _NNENUM_SAFE_BEFORE_ENUM_RE.search(text):
        return "UNSAT"

    # Other safety/proof phrases.
    if _NNENUM_PROVEN_SAFE_RE.search(text):
        return "UNSAT"

    if _NNENUM_UNSAT_RE.search(text):
        return "UNSAT"

    # Counterexample / unsafe phrases.
    if _NNENUM_UNSAFE_RE.search(text):
        return "SAT"

    if _NNENUM_SAT_RE.search(text):
        return "SAT"

    # Explicit timeout text.
    if _NNENUM_TIMEOUT_RE.search(text):
        return "TIMEOUT"

    # subprocess killed by outer timeout / SIGTERM / SIGKILL.
    if rc in {124, 137, 143, -9, -15}:
        return "TIMEOUT"

    # Errors only after checking all valid result/timeout cases.
    if _NNENUM_ERROR_RE.search(text):
        return "ERROR"

    parsed = normalize_status_from_text(text)
    if parsed is not None:
        return parsed

    return None


# def add_args(p: argparse.ArgumentParser) -> None:
#     p.add_argument(
#         "--nnenum_python",
#         type=str,
#         default="python",
#         help="Python executable to use for nnenum.",
#     )
#     p.add_argument(
#         "--nnenum_module",
#         type=str,
#         default="nnenum.nnenum",
#         help="Python module entry point for nnenum.",
#     )
#     p.add_argument(
#         "--nnenum_timeout",
#         type=int,
#         default=600,
#         help="Timeout passed to nnenum, in seconds.",
#     )


# def build_cmd(
#     args: argparse.Namespace,
#     onnx_path: str,
#     vnnlib_path: str,
#     workdir: Path,
# ) -> List[str]:
#     timeout = getattr(args, "nnenum_timeout", None)
#     if timeout is None:
#         timeout = getattr(args, "timeout", 300)

#     return [
#         args.nnenum_python,
#         "-m",
#         args.nnenum_module,
#         onnx_path,
#         vnnlib_path,
#         str(int(timeout)),
#     ]

def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--nnenum_python",
        type=str,
        default="python",
        help="Python executable to use for nnenum.",
    )
    p.add_argument(
        "--nnenum_module",
        type=str,
        default="nnenum.nnenum",
        help="Python module entry point for nnenum.",
    )
    p.add_argument(
        "--nnenum_timeout",
        type=int,
        default=300,
        help="Timeout passed to nnenum, in seconds.",
    )
    p.add_argument(
        "--nnenum_processes",
        type=int,
        default=1,
        help="Number of nnenum worker processes. Use 1 on macOS to avoid ONNX Runtime pickling errors.",
    )
    p.add_argument(
        "--nnenum_settings",
        type=str,
        default="auto",
        choices=["auto", "control", "image", "exact"],
        help="nnenum settings preset.",
    )


def build_cmd(
    args: argparse.Namespace,
    onnx_path: str,
    vnnlib_path: str,
    workdir: Path,
) -> List[str]:
    # nnenum's classic CLI is positional:
    # python -m nnenum.nnenum ONNX VNNLIB TIMEOUT OUTFILE PROCESSES SETTINGS
    outfile = workdir / "nnenum_result.txt"

    return [
        args.nnenum_python,
        "-m",
        args.nnenum_module,
        onnx_path,
        vnnlib_path,
        str(int(args.nnenum_timeout)),
        str(outfile),
        str(int(args.nnenum_processes)),
        str(args.nnenum_settings),
    ]