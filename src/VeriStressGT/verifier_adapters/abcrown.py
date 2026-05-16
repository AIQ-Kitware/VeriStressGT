from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .common import normalize_status_from_text

VERIFIER_NAME = "abcrown"
CONDA_ENV_VAR = "ABCROWN_CONDA_ENV"
CONDA_ENV_DEFAULT = "alpha-beta-crown"

def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--abcrown_config",
        required=True,
        help="Path to ABCROWN yaml config (passed to VeriStressGT-verify --config_path).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for alpha-beta-CROWN (default: cpu).",
    )


def build_cmd(args: argparse.Namespace, onnx_path: str, vnnlib_path: str, workdir: Path) -> List[str]:
    cfg = str(Path(args.abcrown_config).expanduser().resolve())
    return [
        "VeriStressGT-verify",
        "--method", "abcrown-vnncomp2024",
        "--onnx", onnx_path,
        "--vnnlib", vnnlib_path,
        "--config_path", cfg,
        "--device", args.device,
        # Hardcode a large value so abcrown's internal BaB timer never fires.
        # The subprocess wall-clock kill in verify_benchmark (--timeout) is
        # the only control point, and it correctly receives --timeout flag
        "--timeout", "999999",
    ]


def parse_result(stdout: str, stderr: str, rc: int) -> str | None:
    # ABCROWN's own final result line is in stdout and should beat conda stderr.
    parsed = normalize_status_from_text(stdout)
    if parsed is not None:
        return parsed

    # Only inspect stderr if stdout had no parseable result.
    return normalize_status_from_text(stderr)