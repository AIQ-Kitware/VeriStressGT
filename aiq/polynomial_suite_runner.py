"""MAGNET pipeline-node runner for the polynomial suite card.

Builds a 30-instance polynomial.algebraic_boundary benchmark, then runs
every requested verifier on it sequentially. Verifiers whose binary or
environment is missing are skipped gracefully (logged but not fatal).

All instances are UNSAT by construction (is_robust=True guaranteed by the
nearest-boundary certificate), so the expected verdict is always UNSAT.
A verifier returning SAT is counted as wrong; TIMEOUT/UNKNOWN is not wrong.

Writes {"per_verifier": {...}, "summary": {...}} JSON to --results_fpath so
MAGNET's GenericPipelineProcessor lifts each top-level key into a card symbol.

Requires ALGEBRAIC_VERIFIER_DIR to be set (or --algebraic_verifier_dir) when
algebraic_pnn is in the verifier list.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import scriptconfig as scfg
import ubelt as ub


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VERIFIERS = [
    "algebraic_pnn",
    "abcrown",
    "neuralsat",
    "nnenum",
    "pyrat",
]


def _resolve_under_repo(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (REPO_ROOT / pp).resolve()


def _run_subprocess(cmd: List[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def _load_results_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    verdicts: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return verdicts
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        verdicts[rec["instance_id"]] = rec
    return verdicts


_UNSUPPORTED_KEYWORDS = (
    "unsupported",
    "not supported",
    "not implement",
    "notimplementederror",
    "unsupportedop",
    "layer type",
    "no support",
)


def _classify(status: str, rec: Optional[Dict[str, Any]]) -> str:
    """Map a raw status + record into one of: correct, wrong, timeout, unsupported, error, missing."""
    s = status.lower()
    if s == "unsat":
        return "correct"
    if s == "sat":
        return "wrong"
    if s == "timeout":
        return "timeout"
    if s == "missing":
        return "missing"
    if s == "error":
        preview = ""
        if rec:
            preview = (rec.get("error_preview") or "").lower()
            if not preview:
                preview = (rec.get("stdout_preview") or "").lower()
        # HC Julia worker killed by its own internal timeout → treat as TIMEOUT
        if "julia worker exceeded timeout" in preview and "was killed" in preview:
            return "timeout"
        if any(kw in preview for kw in _UNSUPPORTED_KEYWORDS):
            return "unsupported"
        return "error"
    # UNKNOWN, etc.
    return "error"


def _grade_verifier(
    verifier: str,
    instance_ids: List[str],
    run_subdir: Path,
) -> Dict[str, Any]:
    """Grade one verifier's results against expected UNSAT for every instance."""
    verdicts = _load_results_jsonl(run_subdir / "results.jsonl")
    per_instance: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {
        "correct": 0, "wrong": 0, "timeout": 0, "unsupported": 0, "error": 0, "missing": 0,
    }

    for inst_id in instance_ids:
        rec = verdicts.get(inst_id)
        raw_status = str(rec["status"]).lower() if rec else "missing"
        wall = float(rec.get("wall_time_s") or 0.0) if rec else None
        category = _classify(raw_status, rec)
        counts[category] += 1
        per_instance.append({
            "instance_id": inst_id,
            "expected": "unsat",
            "actual": raw_status,
            "category": category,
            "wall_time_s": wall,
        })

    total = len(per_instance)
    correct_fraction = counts["correct"] / total if total else 0.0
    print(
        f"  [{verifier}] correct={counts['correct']}/{total} ({correct_fraction:.0%}) "
        f"wrong={counts['wrong']} timeout={counts['timeout']} "
        f"error={counts['error']}",
        flush=True,
    )
    return {
        "correct": counts["correct"],
        "total": total,
        "correct_fraction": correct_fraction,
        "wrong": counts["wrong"],
        "timeout": counts["timeout"],
        "unsupported": counts["unsupported"],
        "error": counts["error"],
        "skipped": False,
        "per_instance": per_instance,
    }


class PolynomialSuiteRunnerCLI(scfg.DataConfig):
    """Build a 30-instance polynomial suite, run all requested verifiers, report results.

    All instances are UNSAT by construction; a verifier returning SAT is a
    soundness error. TIMEOUT/UNKNOWN is reported but not counted as wrong.
    """

    spec_path = scfg.Value(
        "src/VeriStressGT/configs/polynomial_suite.yaml",
        help="Build spec passed to VeriStressGT.cli.create_benchmark.",
        tags=["algo_param"],
    )

    bench_dir = scfg.Value(
        "polynomial_suite_bench",
        help="Where create_benchmark writes ONNX/VNNLIB instances.",
        tags=["algo_param"],
    )

    run_dir = scfg.Value(
        "polynomial_suite_run",
        help="Root dir for per-verifier verify_benchmark output subdirs.",
        tags=["algo_param"],
    )

    timeout = scfg.Value(
        300.0,
        type=float,
        help="Per-instance verifier wall-clock timeout (seconds).",
        tags=["algo_param"],
    )

    verifiers = scfg.Value(
        DEFAULT_VERIFIERS,
        help="Ordered list of verifier keys to run. Missing verifiers are skipped.",
        tags=["algo_param"],
    )

    algebraic_verifier_dir = scfg.Value(
        None,
        help=(
            "Path to the AlgebraicVerification repo. "
            "Falls back to $ALGEBRAIC_VERIFIER_DIR."
        ),
        tags=["algo_param"],
    )

    max_instances = scfg.Value(
        None,
        type=int,
        help=(
            "Cap the number of instances verified (takes the first N from the manifest). "
            "None means run all. Useful for quick smoke tests."
        ),
        tags=["algo_param"],
    )

    rebuild = scfg.Value(
        False,
        help=(
            "Force re-generation of the benchmark even if bench_dir/manifest.json "
            "already exists. By default the existing build is reused. "
            "Pass --rebuild True to enable."
        ),
        tags=["algo_param"],
    )

    abcrown_config = scfg.Value(
        "src/VeriStressGT/configs/abcrown_basic.yaml",
        help="α-β-CROWN config YAML (required by the abcrown adapter).",
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

        # Inject algebraic verifier dir into env if provided
        env_av_dir = config.algebraic_verifier_dir or os.environ.get(
            "ALGEBRAIC_VERIFIER_DIR"
        )
        if env_av_dir:
            os.environ["ALGEBRAIC_VERIFIER_DIR"] = str(
                Path(env_av_dir).expanduser().resolve()
            )

        verifiers: List[str] = (
            config.verifiers
            if isinstance(config.verifiers, list)
            else [v.strip() for v in str(config.verifiers).split(",") if v.strip()]
        )

        # ── 1. Build benchmark (skipped if already built) ─────────────────────
        manifest_path = bench_dir / "manifest.json"
        if manifest_path.exists() and not config.rebuild:
            print(
                f"\n=== Reusing existing benchmark at {bench_dir} "
                "(pass --rebuild to regenerate) ===",
                flush=True,
            )
        else:
            print("\n=== Building benchmark ===", flush=True)
            _run_subprocess([
                sys.executable, "-m", "VeriStressGT.cli.create_benchmark",
                "--spec", str(spec_path),
                "--out_dir", str(bench_dir),
                "--overwrite",
            ])

        manifest = json.loads(manifest_path.read_text())
        all_instance_ids = [inst["id"] for inst in manifest["instances"]]

        max_n: Optional[int] = config.max_instances
        instance_ids = all_instance_ids[:max_n] if max_n is not None else all_instance_ids
        print(
            f"Using {len(instance_ids)}/{len(all_instance_ids)} instances.",
            flush=True,
        )

        # Per-verifier extra args required by specific adapters
        verifier_extra: Dict[str, List[str]] = {
            "abcrown": [
                "--abcrown_config",
                str(_resolve_under_repo(config.abcrown_config)),
            ],
        }

        # ── 2. Run each verifier ──────────────────────────────────────────────
        per_verifier: Dict[str, Dict[str, Any]] = {}
        verifiers_run: List[str] = []
        verifiers_skipped: List[str] = []

        for verifier in verifiers:
            print(f"\n=== Verifier: {verifier} ===", flush=True)
            run_subdir = run_dir / verifier
            cmd = [
                sys.executable, "-m", "VeriStressGT.cli.verify_benchmark",
                "--benchmark", str(bench_dir),
                "--verifier", verifier,
                "--out_dir", str(run_subdir),
                "--timeout", str(float(config.timeout)),
                "--instances", *instance_ids,
                "--overwrite",
                *verifier_extra.get(verifier, []),
            ]
            try:
                _run_subprocess(cmd)
                verifiers_run.append(verifier)
                per_verifier[verifier] = _grade_verifier(
                    verifier, instance_ids, run_subdir
                )
            except subprocess.CalledProcessError as exc:
                print(
                    f"  [SKIP] {verifier} exited with code {exc.returncode} "
                    "(missing binary/env or setup error — skipping)",
                    flush=True,
                )
                verifiers_skipped.append(verifier)
                per_verifier[verifier] = {
                    "correct": 0,
                    "total": len(instance_ids),
                    "correct_fraction": None,
                    "wrong": 0,
                    "timeout": 0,
                    "unsupported": 0,
                    "error": 0,
                    "skipped": True,
                    "per_instance": [],
                }


        # ── 3. Write results ──────────────────────────────────────────────────
        summary: Dict[str, Any] = {
            "total_instances": len(instance_ids),
            "verifiers_run": verifiers_run,
            "verifiers_skipped": verifiers_skipped,
            "per_verifier_fraction": {
                v: per_verifier[v]["correct_fraction"]
                for v in verifiers_run
            },
        }

        # Lift per_verifier values up to the top level for MAGNET symbol access
        out = {
            "per_verifier": per_verifier,
            "summary": summary,
        }

        out_fpath = ub.Path(config.results_fpath)
        out_fpath.parent.ensuredir()
        merged = json.loads(out_fpath.read_text()) if out_fpath.exists() else {}
        merged.update(out)
        out_fpath.write_text(json.dumps(merged, indent=2))
        print(f"\nWrote results to: {out_fpath}", flush=True)

        # Print a quick summary table
        header = f"  {'verifier':<20s} {'ok':>4}  {'wrong':>5}  {'timeout':>7}  {'error':>5}  {'total':>5}"
        print(f"\n── Summary {'─' * (len(header) - 10)}", flush=True)
        print(header, flush=True)
        print(f"  {'─'*20}  {'─'*4}  {'─'*5}  {'─'*7}  {'─'*5}  {'─'*5}", flush=True)
        for v in verifiers:
            info = per_verifier[v]
            if info["skipped"]:
                print(f"  {v:<20s} SKIPPED", flush=True)
            else:
                print(
                    f"  {v:<20s} {info['correct']:>4}  {info['wrong']:>5}  "
                    f"{info['timeout']:>7}  {info['error']:>5}  {info['total']:>5}",
                    flush=True,
                )

        csv_fpath = out_fpath.parent / "polynomial_summary.csv"
        with open(csv_fpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["verifier", "correct", "wrong", "timeout", "unsupported", "error", "skipped", "total", "correct_fraction"])
            for v in verifiers:
                info = per_verifier[v]
                w.writerow([
                    v,
                    info["correct"],
                    info["wrong"],
                    info["timeout"],
                    info["unsupported"],
                    info["error"],
                    info["skipped"],
                    info["total"],
                    info["correct_fraction"],
                ])
        print(f"Summary CSV: {csv_fpath}", flush=True)


__cli__ = PolynomialSuiteRunnerCLI

if __name__ == "__main__":
    __cli__.main()

    r"""
    CommandLine:
        export ALGEBRAIC_VERIFIER_DIR=/path/to/AlgebraicVerification
        python aiq/polynomial_suite_runner.py \
            --spec_path src/VeriStressGT/configs/polynomial_suite.yaml \
            --bench_dir ./polynomial_suite_bench \
            --run_dir ./polynomial_suite_run \
            --timeout 300 \
            --verifiers algebraic_pnn \
            --results_fpath ./results.json
    """
