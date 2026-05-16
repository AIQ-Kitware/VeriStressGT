from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import List, Optional

VERIFIER_NAME = "nnv"

_NNV_RESULT_RE = re.compile(r"(?im)^\s*Result:\s*(safe|unsafe|unknown|error)\s*$")
_TIMEOUT_RE = re.compile(r"(?im)\b(timeout|timed out|time limit|killed)\b")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--matlab_bin",
        default="/Applications/MATLAB_R2025b.app/bin/matlab",
        help="Path to MATLAB binary.",
    )
    p.add_argument(
        "--nnv_dir",
        default=None,
        help=(
            "Path to bundled NNV directory. "
            "Defaults to src/VeriStressGT/verifiers/nnv relative to this file."
        ),
    )
    p.add_argument(
        "--nnv_internal_timeout",
        type=float,
        default=None,
        help=(
            "Timeout passed to run_nnv_verifier(onnx, vnnlib, timeout_s). "
            "If omitted, pass the driver timeout if visible to this adapter, else 0."
        ),
    )
    p.add_argument(
        "--nnv_entry",
        default="run_nnv_verifier",
        help="MATLAB function to call: run_nnv_verifier(onnx_path, vnnlib_path, timeout_s).",
    )


def _default_nnv_dir() -> Path:
    # this file is: src/VeriStressGT/verifier_adapters/nnv.py
    # nnv is at:    src/VeriStressGT/verifiers/nnv
    return Path(__file__).resolve().parents[1] / "verifiers" / "nnv"


def _matlab_quote(s: str | Path) -> str:
    """Single-quote a string for MATLAB code."""
    return "'" + str(s).replace("'", "''") + "'"


def build_cmd(args, onnx_path: str, vnnlib_path: str, workdir: Path) -> List[str]:
    matlab = Path(args.matlab_bin).expanduser().resolve()
    nnv_dir = (
        Path(args.nnv_dir).expanduser().resolve()
        if args.nnv_dir
        else _default_nnv_dir()
    )

    if not matlab.exists():
        raise FileNotFoundError(f"MATLAB binary not found: {matlab}")
    if not nnv_dir.exists():
        raise FileNotFoundError(f"NNV dir not found: {nnv_dir}")

    # Expected layout:
    #   src/VeriStressGT/verifiers/run_nnv_verifier.m
    #   src/VeriStressGT/verifiers/nnv/code/nnv/...
    wrapper_dir = nnv_dir.parent

    entry = args.nnv_entry
    if not re.match(r"^[A-Za-z]\w*$", entry):
        raise ValueError(f"Invalid MATLAB function name: {entry!r}")

    entry_file = wrapper_dir / f"{entry}.m"
    if not entry_file.exists():
        raise FileNotFoundError(f"MATLAB entry file not found: {entry_file}")

    internal_timeout = args.nnv_internal_timeout
    if internal_timeout is None:
        internal_timeout = float(getattr(args, "timeout", 0.0) or 0.0)

    # Stale old checkout path from earlier failures.
    old_nnv_dir = "/Desktop/VeriStressGT/nnv"

    matlab_cmd = " ".join(
        [
            "warning('off','MATLAB:dispatcher:nameConflict');",
            "warning('off','MATLAB:rmpath:DirNotFound');",
            "warning('off','MATLAB:addpath:DirNotFound');",

            # Remove stale saved paths pointing at the old repo-level NNV checkout.
            f"old_nnv = {_matlab_quote(old_nnv_dir)};",
            "p = strsplit(path, pathsep);",
            "for i = 1:numel(p), "
            "if contains(p{i}, old_nnv), "
            "try, rmpath(p{i}); catch, end; "
            "end; "
            "end;",

            # Add bundled NNV and wrapper directory.
            f"nnv_dir = {_matlab_quote(nnv_dir)};",
            f"wrapper_dir = {_matlab_quote(wrapper_dir)};",
            "addpath(genpath(nnv_dir));",
            "addpath(wrapper_dir);",
            "rehash toolboxcache;",

            "warning('on','MATLAB:dispatcher:nameConflict');",
            "warning('on','MATLAB:rmpath:DirNotFound');",
            "warning('on','MATLAB:addpath:DirNotFound');",

            "disp('[VeriStressGT] running NNV');",
            "disp(['[VeriStressGT] nnv_dir = ', nnv_dir]);",
            f"{entry}({_matlab_quote(onnx_path)}, {_matlab_quote(vnnlib_path)}, {float(internal_timeout)});",
            "exit;",
        ]
    )

    cmd_str = f"{shlex.quote(str(matlab))} -batch {shlex.quote(matlab_cmd)}"
    return ["bash", "-lc", cmd_str]


def parse_result(stdout: str, stderr: str, rc: int) -> Optional[str]:
    text = (stdout or "") + "\n" + (stderr or "")

    m = list(_NNV_RESULT_RE.finditer(text))
    if m:
        tok = m[-1].group(1).lower()
        if tok == "safe":
            return "UNSAT"
        if tok == "unsafe":
            return "SAT"
        if tok == "unknown":
            return "UNKNOWN"
        if tok == "error":
            return "ERROR"

    if _TIMEOUT_RE.search(text):
        return "TIMEOUT"

    # Common shell timeout / kill return codes.
    if rc in {124, 137, 143}:
        return "TIMEOUT"

    # Empty MATLAB/NNV output usually means the outer driver killed it.
    if not text.strip():
        return "TIMEOUT"

    tlow = text.lower()

    if "result: safe" in tlow:
        return "UNSAT"
    if "result: unsafe" in tlow:
        return "SAT"
    if "result: unknown" in tlow:
        return "UNKNOWN"
    if "result: error" in tlow:
        return "ERROR"

    return None