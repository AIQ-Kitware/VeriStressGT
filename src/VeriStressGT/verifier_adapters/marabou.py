from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

from .common import normalize_status_from_text

VERIFIER_NAME = "marabou"

_MARABOU_UNSAT_RE = re.compile(r"(?im)(?:^|\n)\s*unsat\s*(?:\n|$)")
_MARABOU_SAT_RE = re.compile(r"(?im)(?:^|\n)\s*sat\s*(?:\n|$)")

# Be more specific than just matching the word "timeout" anywhere.
_MARABOU_TIMEOUT_RE = re.compile(
    r"(?im)"
    r"(?:"
    r"time\s*limit\s*reached|"
    r"timed\s*out|"
    r"timeout\s*reached|"
    r"global\s*timeout|"
    r"engine.*timeout|"
    r"solve.*timeout"
    r")"
)

_MARABOU_ERROR_RE = re.compile(
    r"(?im)\b(?:error|exception|traceback|not currently supported|unsupported)\b"
)


def parse_result(stdout: str, stderr: str, rc: int) -> str | None:
    stdout = stdout or ""
    stderr = stderr or ""
    text = stdout + "\n" + stderr

    # Silent outer kill / no output.
    if not text.strip():
        return "TIMEOUT"

    # Marabou prints clean sat/unsat lines.
    if _MARABOU_UNSAT_RE.search(text):
        return "UNSAT"

    if _MARABOU_SAT_RE.search(text):
        return "SAT"

    # Outer timeout / killed process return codes.
    if rc in {124, 137, 143, -9, -15}:
        return "TIMEOUT"

    # Explicit Marabou timeout language.
    if _MARABOU_TIMEOUT_RE.search(text):
        return "TIMEOUT"

    if _MARABOU_ERROR_RE.search(text):
        return "ERROR"

    parsed = normalize_status_from_text(text)
    if parsed is not None:
        return parsed

    return None


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--marabou_bin",
        type=str,
        default="Marabou",
        help="Path to Marabou binary executable.",
    )
    p.add_argument(
        "--marabou_timeout",
        type=int,
        default=None,
        help=(
            "Timeout passed directly to Marabou in seconds. "
            "Defaults to the outer --timeout value."
        ),
    )
    p.add_argument(
        "--marabou_num_workers",
        type=int,
        default=1,
        help="Number of Marabou workers.",
    )
    p.add_argument(
        "--marabou_blas_threads",
        type=int,
        default=1,
        help="Number of BLAS threads used by Marabou.",
    )


def build_cmd(
    args: argparse.Namespace,
    onnx_path: str,
    vnnlib_path: str,
    workdir: Path,
) -> List[str]:
    marabou_bin = str(Path(args.marabou_bin).expanduser().resolve())

    # Important: if user passed --timeout 600 to verify_benchmark, use that
    # as Marabou's internal timeout unless --marabou_timeout overrides it.
    timeout = getattr(args, "marabou_timeout", None)
    if timeout is None:
        timeout = getattr(args, "timeout", 300)

    return [
        marabou_bin,
        onnx_path,
        vnnlib_path,
        f"--timeout={int(timeout)}",
        f"--num-workers={int(args.marabou_num_workers)}",
        f"--blas-threads={int(args.marabou_blas_threads)}",
        "--verbosity=2",
        "--summary-file",
        str(workdir / "marabou_summary.txt"),
    ]