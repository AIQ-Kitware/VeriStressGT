from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import List, Optional

from .common import normalize_status_from_text

VERIFIER_NAME = "neuralsat"
CONDA_ENV_VAR = "NEURALSAT_CONDA_ENV"
CONDA_ENV_DEFAULT = None 

_NEURALSAT_RESULT_RE = re.compile(
    r"(?im)^\s*(?:\[\!\]\s*)?Result:\s*(sat|unsat|unknown|timeout)\s*$"
)
_NEURALSAT_SAFE_RE = re.compile(r"(?im)\b(?:network is safe|result:\s*safe|safe)\b")
_NEURALSAT_UNSAFE_RE = re.compile(
    r"(?im)\b(?:network is unsafe|result:\s*unsafe|unsafe|counterexample|adversarial)\b"
)


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--neuralsat_root",
        default=None,
        help=(
            "Path to NeuralSAT repo root"
        ),
    )
    p.add_argument(
        "--neuralsat_python",
        default="python",
        help="Python executable to run NeuralSAT (must have deps installed).",
    )
    p.add_argument(
        "--neuralsat_force_split",
        default="auto",
        choices=["auto", "hidden", "input"],
        help="Branching strategy override. 'auto' (default) lets NeuralSAT "
            "pick based on input dimension — input splits for low-dim inputs "
            "(ACAS Xu, control), hidden splits for image classifiers. Only "
            "override if you know what you're doing.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for NeuralSAT (default: cpu). Use 'cuda' to enable GPU acceleration.",
    )


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1] / "verifiers" / "neuralsat"


def build_cmd(
    args: argparse.Namespace,
    onnx_path: str,
    vnnlib_path: str,
    workdir: Path,
) -> List[str]:
    root = Path(args.neuralsat_root).expanduser().resolve() if args.neuralsat_root else _default_root()
    main_py = root / "src" / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"NeuralSAT main.py not found at: {main_py}")

    py = args.neuralsat_python

    force_split = ""
    if args.neuralsat_force_split != "auto":
        force_split = f" --force_split {shlex.quote(args.neuralsat_force_split)}"

    cmd = (
        f"cd {shlex.quote(str(root))} && "
        f"{shlex.quote(py)} {shlex.quote(str(main_py))} "
        f"--net {shlex.quote(onnx_path)} "
        f"--spec {shlex.quote(vnnlib_path)} "
        f"--device {shlex.quote(args.device)}"
        f"{force_split}"
    )

    return ["bash", "-lc", cmd]


def parse_result(stdout: str, stderr: str, rc: int) -> Optional[str]:
    text = (stdout or "") + "\n" + (stderr or "")

    matches = list(_NEURALSAT_RESULT_RE.finditer(text))
    if matches:
        tok = matches[-1].group(1).lower()
        if tok == "unsat":
            return "UNSAT"
        if tok == "sat":
            return "SAT"
        if tok == "timeout":
            return "TIMEOUT"
        return "UNKNOWN"

    if _NEURALSAT_SAFE_RE.search(text):
        return "UNSAT"
    if _NEURALSAT_UNSAFE_RE.search(text):
        return "SAT"

    return normalize_status_from_text(text)
