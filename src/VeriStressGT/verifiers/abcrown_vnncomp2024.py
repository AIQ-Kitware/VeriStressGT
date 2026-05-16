from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ABCrownResult:
    status: str
    raw_output: str

import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

def _clean_output(out: str) -> str:
    out = _ANSI_RE.sub("", out)
    out = out.replace("\r\n", "\n").replace("\r", "\n")  # <-- key
    return out

_RESULT_RE = re.compile(r"(?im)^\s*Result:\s*(unsat|sat|unknown)\s*$")

# Optional fallbacks (ONLY if no Result line exists)
_HARD_TIMEOUT_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:time limit reached|terminated.*timeout|timed out)\b"
)

def _parse_abcrown_status(out: str) -> tuple[str, str]:

    out = _clean_output(out)
    # 1) Authoritative: last "Result:" line
    matches = list(_RESULT_RE.finditer(out))
    if matches:
        token = matches[-1].group(1).lower()
        if token == "unsat":
            return "robust", "matched last Result: unsat"
        if token == "sat":
            return "violated", "matched last Result: sat"
        return "unknown", "matched last Result: unknown"

    # 2) Fallbacks (avoid matching "Remaining timeout" etc.)
    if _HARD_TIMEOUT_RE.search(out):
        return "timeout", "matched hard timeout message"

    return "unknown", "no Result line"

def run_abcrown_vnncomp2024(
    onnx_path: str | Path,
    vnnlib_path: str | Path,
    config_path: str | Path,
    timeout_s: int = 300,
    abcrown_repo_dir: Optional[str | Path] = None,
    conda_env: Optional[str] = None,
) -> ABCrownResult:
    # IMPORTANT: capture the caller's cwd *before* we run subprocess with a different cwd
    caller_cwd = Path.cwd()

    onnx_path = Path(onnx_path).expanduser().resolve()
    vnnlib_path = Path(vnnlib_path).expanduser().resolve()

    # If config_path is relative, interpret it relative to the caller's cwd (repo root in normal usage)
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = (caller_cwd / config_path).resolve()
    else:
        config_path = config_path.resolve()

    repo_dir = Path(
        abcrown_repo_dir or os.environ.get("ABCROWN_VNNCOMP2024_DIR", "")
    ).expanduser().resolve()

    if not repo_dir.exists():
        raise FileNotFoundError(
            "alpha-beta-CROWN_vnncomp2024 repo not found. "
            "Set ABCROWN_VNNCOMP2024_DIR=/path/to/alpha-beta-CROWN_vnncomp2024"
        )

    if not config_path.exists():
        raise FileNotFoundError(f"ABCROWN config not found: {config_path}")

    conda_env = conda_env or os.environ.get("ABCROWN_CONDA_ENV", "alpha-beta-crown")

    complete_verifier_dir = repo_dir / "complete_verifier"
    abcrown_py = complete_verifier_dir / "abcrown.py"
    if not abcrown_py.exists():
        raise FileNotFoundError(f"Expected {abcrown_py} to exist")

    cmd = [
        "conda", "run", "-n", conda_env,
        "python","-u", str(abcrown_py),
        "--config", str(config_path),
        "--onnx_path", str(onnx_path),
        "--vnnlib_path", str(vnnlib_path),
        "--timeout", str(timeout_s),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(complete_verifier_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    out = proc.stdout
    status, reason = _parse_abcrown_status(out)
    print(reason)
    return ABCrownResult(status=status, raw_output=out)