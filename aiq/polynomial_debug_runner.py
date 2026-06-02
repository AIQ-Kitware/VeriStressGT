"""MAGNET pipeline-node runner for the polynomial algebraic-boundary debug card.

Single one-node pipeline that:
  1. Invokes VeriStressGT's create_benchmark with a build spec (small
     polynomial.algebraic_boundary instances with known ground truth);
  2. Invokes verify_benchmark with --verifier algebraic_pnn;
  3. Grades reported status against expected UNSAT (the construction
     guarantees is_robust=True for every emitted instance).

Writes {"result": {...}} JSON to --results_fpath so MAGNET's
GenericPipelineProcessor lifts each top-level key into a card symbol.

Requires ALGEBRAIC_VERIFIER_DIR to be set (or --algebraic_verifier_dir).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import scriptconfig as scfg
import ubelt as ub


REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_under_repo(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (REPO_ROOT / pp).resolve()


def _run_subprocess(cmd: List[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


class PolynomialDebugRunnerCLI(scfg.DataConfig):
    """Build a small polynomial algebraic-boundary benchmark, run the
    algebraic_pnn verifier on it, and report per-instance correctness.

    All instances produced by polynomial.algebraic_boundary have
    is_robust=True, so the expected verdict is always UNSAT.
    """

    spec_path = scfg.Value(
        "src/VeriStressGT/configs/polynomial_debug.yaml",
        help="Build spec passed to VeriStressGT.cli.create_benchmark.",
        tags=["algo_param"],
    )

    bench_dir = scfg.Value(
        "polynomial_debug_bench",
        help="Where create_benchmark writes ONNX/VNNLIB instances.",
        tags=["algo_param"],
    )

    run_dir = scfg.Value(
        "polynomial_debug_run",
        help="Where verify_benchmark writes results.jsonl, logs, etc.",
        tags=["algo_param"],
    )

    timeout = scfg.Value(
        300.0,
        type=float,
        help="Per-instance verifier wall-clock timeout (seconds).",
        tags=["algo_param"],
    )

    algebraic_verifier_dir = scfg.Value(
        None,
        help=(
            "Path to the AlgebraicVerification repo (containing verify_vnnlib.py). "
            "If unset, falls back to $ALGEBRAIC_VERIFIER_DIR."
        ),
        tags=["algo_param"],
    )

    results_fpath = scfg.Value(
        "results.json",
        help="Output JSON file consumed by MAGNET.",
        tags=["out_path", "primary"],
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        spec_path = _resolve_under_repo(config.spec_path)
        if not spec_path.exists():
            raise FileNotFoundError(f"build spec not found: {spec_path}")

        bench_dir = Path(config.bench_dir).resolve()
        run_dir = Path(config.run_dir).resolve()

        env_av_dir = config.algebraic_verifier_dir or os.environ.get(
            "ALGEBRAIC_VERIFIER_DIR"
        )
        if not env_av_dir:
            raise SystemExit(
                "ALGEBRAIC_VERIFIER_DIR is not set and --algebraic_verifier_dir "
                "was not passed. Point one at the AlgebraicVerification repo root."
            )
        os.environ["ALGEBRAIC_VERIFIER_DIR"] = str(
            Path(env_av_dir).expanduser().resolve()
        )

        # 1) build
        _run_subprocess([
            sys.executable, "-m", "VeriStressGT.cli.create_benchmark",
            "--spec", str(spec_path),
            "--out_dir", str(bench_dir),
            "--overwrite",
        ])

        # 2) verify
        _run_subprocess([
            sys.executable, "-m", "VeriStressGT.cli.verify_benchmark",
            "--benchmark", str(bench_dir),
            "--verifier", "algebraic_pnn",
            "--out_dir", str(run_dir),
            "--timeout", str(float(config.timeout)),
            "--overwrite",
        ])

        # 3) grade — all instances are UNSAT by construction
        manifest = json.loads((bench_dir / "manifest.json").read_text())
        instance_ids = [inst["id"] for inst in manifest["instances"]]

        results_jsonl = run_dir / "results.jsonl"
        verdicts: Dict[str, Dict[str, Any]] = {}
        for line in results_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            verdicts[rec["instance_id"]] = rec

        per_instance: List[Dict[str, Any]] = []
        correct_count = 0
        for inst_id in instance_ids:
            expected = "unsat"
            rec = verdicts.get(inst_id)
            if rec is None:
                actual = "missing"
                is_correct = False
                wall = None
            else:
                actual = str(rec["status"]).lower()
                is_correct = actual == expected
                wall = float(rec.get("wall_time_s") or 0.0)
            if is_correct:
                correct_count += 1
            per_instance.append({
                "instance_id": inst_id,
                "expected": expected,
                "actual": actual,
                "correct": is_correct,
                "wall_time_s": wall,
            })
            print(
                f"  [{inst_id}] expected={expected:7s} actual={actual:7s} "
                f"correct={is_correct} wall={wall}",
                flush=True,
            )

        total = len(per_instance)
        correct_fraction = (correct_count / total) if total else 0.0

        out = {
            "result": {
                "correct_count": correct_count,
                "total": total,
                "correct_fraction": correct_fraction,
                "spec_path": str(spec_path),
                "bench_dir": str(bench_dir),
                "run_dir": str(run_dir),
                "per_instance": per_instance,
            }
        }

        out_fpath = ub.Path(config.results_fpath)
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps(out, indent=2))
        print(f"Wrote results to: {out_fpath}")


__cli__ = PolynomialDebugRunnerCLI

if __name__ == "__main__":
    __cli__.main()

    r"""
    CommandLine:
        export ALGEBRAIC_VERIFIER_DIR=/path/to/AlgebraicVerification
        python aiq/polynomial_debug_runner.py \
            --spec_path src/VeriStressGT/configs/polynomial_debug.yaml \
            --bench_dir ./polynomial_debug_bench \
            --run_dir ./polynomial_debug_run \
            --timeout 300 \
            --results_fpath ./results.json
    """
