"""Combined evaluation runner: mini sweep then polynomial suite.

Calls mini_sweep_runner and polynomial_suite_runner sequentially, passing
the same --results_fpath to both so their merge-write logic produces a
single JSON with all symbols needed by the card claim.

Using a single pipeline node avoids MAGNET's isolated-output-per-node
behaviour, which would otherwise split mini_sweep_per_verifier and
per_verifier into two separate files that the claim cannot access together.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import scriptconfig as scfg
import ubelt as ub

REPO_ROOT = Path(__file__).resolve().parent.parent
_MINI_SWEEP_RUNNER = Path(__file__).resolve().parent / "mini_sweep_runner.py"
_POLY_RUNNER = Path(__file__).resolve().parent / "polynomial_suite_runner.py"


def _resolve(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (REPO_ROOT / pp).resolve()


class EvaluationRunnerCLI(scfg.DataConfig):
    """Run mini-sweep and polynomial-suite back to back into one results.json."""

    mini_sweep_spec = scfg.Value(
        "src/VeriStressGT/configs/mini_sweep.yaml",
        help="Build spec for the mini sweep.",
        tags=["algo_param"],
    )
    mini_sweep_bench_dir = scfg.Value(
        "mini_sweep_bench",
        help="Where mini-sweep benchmark instances are written.",
        tags=["algo_param"],
    )
    mini_sweep_run_dir = scfg.Value(
        "mini_sweep_run",
        help="Root dir for mini-sweep per-verifier run subdirs.",
        tags=["algo_param"],
    )

    poly_spec = scfg.Value(
        "src/VeriStressGT/configs/polynomial_suite.yaml",
        help="Build spec for the polynomial suite.",
        tags=["algo_param"],
    )
    poly_bench_dir = scfg.Value(
        "polynomial_suite_bench",
        help="Where polynomial-suite benchmark instances are written.",
        tags=["algo_param"],
    )
    poly_run_dir = scfg.Value(
        "polynomial_suite_run",
        help="Root dir for polynomial-suite per-verifier run subdirs.",
        tags=["algo_param"],
    )

    timeout = scfg.Value(
        300.0,
        type=float,
        help="Per-instance verifier wall-clock timeout (seconds), applied to both stages.",
        tags=["algo_param"],
    )
    abcrown_config = scfg.Value(
        "src/VeriStressGT/configs/abcrown_basic.yaml",
        help="α-β-CROWN config YAML used by both stages.",
        tags=["algo_param"],
    )
    rebuild = scfg.Value(
        False,
        help="Force benchmark regeneration even if bench dirs already exist.",
        tags=["algo_param"],
    )
    results_fpath = scfg.Value(
        "results.json",
        help="Output JSON consumed by MAGNET (receives merged output from both stages).",
        tags=["out_path", "primary"],
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        results_fpath = str(ub.Path(config.results_fpath).resolve())

        # Force unbuffered output through the whole subprocess chain so
        # per-instance progress prints live even when MAGNET pipes through tee.
        unbuffered_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        def _run(cmd):
            print(f"$ {' '.join(cmd)}", flush=True)
            subprocess.check_call([sys.executable, "-u"] + cmd[1:], env=unbuffered_env)

        rebuild_flag = ["--rebuild", "True"] if config.rebuild else []

        # ── Stage 1: mini sweep ───────────────────────────────────────────────
        print("\n=== Stage 1: Mini Sweep ===", flush=True)
        _run([
            sys.executable, str(_MINI_SWEEP_RUNNER),
            "--spec_path",      str(_resolve(config.mini_sweep_spec)),
            "--bench_dir",      str(_resolve(config.mini_sweep_bench_dir)),
            "--run_dir",        str(_resolve(config.mini_sweep_run_dir)),
            "--timeout",        str(config.timeout),
            "--abcrown_config", str(_resolve(config.abcrown_config)),
            "--results_fpath",  results_fpath,
            *rebuild_flag,
        ])

        # ── Stage 2: polynomial suite ─────────────────────────────────────────
        print("\n=== Stage 2: Polynomial Suite ===", flush=True)
        _run([
            sys.executable, str(_POLY_RUNNER),
            "--spec_path",      str(_resolve(config.poly_spec)),
            "--bench_dir",      str(_resolve(config.poly_bench_dir)),
            "--run_dir",        str(_resolve(config.poly_run_dir)),
            "--timeout",        str(config.timeout),
            "--abcrown_config", str(_resolve(config.abcrown_config)),
            "--results_fpath",  results_fpath,
            *rebuild_flag,
        ])

        print(f"\nAll stages complete. Results: {results_fpath}", flush=True)


__cli__ = EvaluationRunnerCLI

if __name__ == "__main__":
    __cli__.main()

    r"""
    CommandLine:
        python aiq/evaluation_runner.py \
            --timeout 300 \
            --results_fpath ./results.json
    """
