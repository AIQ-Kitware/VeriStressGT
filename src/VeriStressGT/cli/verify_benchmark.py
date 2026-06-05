# src/VeriStressGT/cli/verify_benchmark.py
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from VeriStressGT.registry.verifiers import discover_verifiers
from VeriStressGT.verifier_adapters.common import finalize_status


def _git_commit() -> Optional[str]:
    import subprocess as sp
    try:
        r = sp.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return None


def _first_nonempty_line(*chunks: str) -> Optional[str]:
    for chunk in chunks:
        if not chunk:
            continue
        for line in chunk.splitlines():
            s = line.strip()
            if s:
                return s
    return None


def _summarize_process_output(stdout: str, stderr: str, rc: int, status: str) -> Optional[str]:
    """
    Build a short human-readable preview from raw process output.

    Preference:
      1) lines containing obvious failure keywords
      2) first non-empty stderr line
      3) first non-empty stdout line
    """
    merged = "\n".join([stderr or "", stdout or ""])
    keywords = (
        "error",
        "exception",
        "traceback",
        "not supported",
        "not currently supported",
        "external data",
        "failed",
        "segmentation fault",
        "abort",
        "killed",
        "illegal",
        "unsupported",
    )

    for line in merged.splitlines():
        s = line.strip()
        if not s:
            continue
        sl = s.lower()
        if any(k in sl for k in keywords):
            return s[:300]

    first = _first_nonempty_line(stderr, stdout)
    if first is not None and (status == "ERROR" or rc != 0):
        return first[:300]

    return None


@dataclass(frozen=True)
class InstanceJob:
    instance_id: str
    seed: Optional[int]
    construction: Optional[str]
    onnx_path: Path
    vnnlib_path: Path
    meta_path: Optional[Path]


def _load_manifest(bench_dir: Path) -> Dict[str, Any]:
    p = bench_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"manifest.json not found: {p}")
    return json.loads(p.read_text())


def _collect_jobs(bench_dir: Path, manifest: Dict[str, Any], only_instances: Optional[List[str]]) -> List[InstanceJob]:
    jobs: List[InstanceJob] = []
    only_set = set(only_instances) if only_instances else None

    for inst in manifest.get("instances", []):
        inst_id = str(inst["id"])
        if only_set is not None and inst_id not in only_set:
            continue

        onnx_rel = Path(inst["paths"]["onnx"])
        vnnlib_rel = Path(inst["paths"]["vnnlib"])
        meta_rel = Path(inst["paths"].get("meta", "")) if "meta" in inst.get("paths", {}) else None

        job = InstanceJob(
            instance_id=inst_id,
            seed=inst.get("seed"),
            construction=inst.get("construction") or manifest.get("construction"),
            onnx_path=(bench_dir / onnx_rel).resolve(),
            vnnlib_path=(bench_dir / vnnlib_rel).resolve(),
            meta_path=(bench_dir / meta_rel).resolve() if meta_rel else None,
        )
        jobs.append(job)

    if not jobs:
        raise ValueError("No instances found to run.")
    return jobs


def _rlimit_as_supported() -> bool:
    """Return True only if RLIMIT_AS can actually be set on this platform."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = soft if soft != resource.RLIM_INFINITY else hard
        if limit == resource.RLIM_INFINITY:
            # Can't probe against infinity; attempt a harmless set/restore.
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        return True
    except Exception:
        return False


def _make_mem_preexec(mem_bytes: int):
    def _fn():
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass
    return _fn


def _run_one(
    job: InstanceJob,
    verifier_name: str,
    verifier_args: argparse.Namespace,
    verifier_build_cmd,
    verifier_parse_result,
    out_dir: Path,
    timeout_s: Optional[float],
    conda_env: Optional[str] = None,
    max_memory_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    workdir = out_dir / "work" / job.instance_id
    workdir.mkdir(parents=True, exist_ok=True)

    stdout_path = logs_dir / f"{job.instance_id}.stdout.txt"
    stderr_path = logs_dir / f"{job.instance_id}.stderr.txt"

    cmd = verifier_build_cmd(verifier_args, str(job.onnx_path), str(job.vnnlib_path), workdir)

    # Auto-wrap in conda env if the verifier declares one.
    if conda_env:
        conda_exe = (
            os.environ.get("CONDA_EXE")
            or shutil.which("conda")
            or "conda"
        )
        cmd = [conda_exe, "run", "-n", conda_env, "--no-capture-output"] + cmd

    t0 = time.time()
    timed_out = False
    rc: Optional[int] = None
    stdout = ""
    stderr = ""

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout_s if timeout_s and timeout_s > 0 else None,
            env=os.environ.copy(),
            preexec_fn=_make_mem_preexec(max_memory_bytes) if max_memory_bytes else None,
        )
        rc = int(proc.returncode)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        timed_out = True
        rc = 124
        stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
        stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
    except Exception as e:
        rc = 1
        stderr = (
            f"{stderr}\n"
            f"[verify_benchmark] failed to launch verifier\n"
            f"  cmd: {cmd}\n"
            f"  cwd: {workdir}\n"
            f"{traceback.format_exc()}"
        )

    wall = time.time() - t0

    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)

    parsed = None
    parse_exception = None
    if verifier_parse_result is not None:
        try:
            parsed = verifier_parse_result(stdout, stderr, rc)
        except Exception as e:
            parse_exception = repr(e)
            stderr_path.write_text(stderr + f"\n[parse_result] exception: {parse_exception}\n")

    status = finalize_status(parsed, rc=rc, timed_out=timed_out)
    error_preview = _summarize_process_output(stdout, stderr, rc=rc, status=status)

    record: Dict[str, Any] = {
        "benchmark": out_dir.name,
        "instance_id": job.instance_id,
        "seed": job.seed,
        "construction": job.construction,
        "paths": {
            "onnx": str(job.onnx_path),
            "vnnlib": str(job.vnnlib_path),
        },
        "verifier": verifier_name,
        "cmd": cmd,
        "status": status,
        "wall_time_s": wall,
        "rc": rc,
        "log_paths": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
        "parse_result_status": parsed,
        "timed_out": timed_out,
    }

    if parse_exception is not None:
        record["parse_result_exception"] = parse_exception
    if error_preview is not None:
        record["error_preview"] = error_preview

    return record


def _write_summary(out_dir: Path, records: List[Dict[str, Any]]) -> None:
    summary: Dict[str, Any] = {
        "total": len(records),
        "by_status": {},
        "avg_wall_time_s": None,
    }
    by_status: Dict[str, int] = {}
    total_time = 0.0
    for r in records:
        s = r["status"]
        by_status[s] = by_status.get(s, 0) + 1
        total_time += float(r.get("wall_time_s", 0.0))
    summary["by_status"] = dict(sorted(by_status.items(), key=lambda kv: kv[0]))
    summary["avg_wall_time_s"] = (total_time / len(records)) if records else None

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    csv_lines = ["instance_id,status,wall_time_s,rc,construction,seed"]
    for r in records:
        csv_lines.append(
            f"{r['instance_id']},{r['status']},{r['wall_time_s']:.6f},{r['rc']},"
            f"{(r.get('construction') or '')},{(r.get('seed') if r.get('seed') is not None else '')}"
        )
    (out_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    verifiers = discover_verifiers()
    if not verifiers:
        raise RuntimeError("No verifiers discovered under VeriStressGT.verifier_adapters")

    ap = argparse.ArgumentParser(
        prog="VeriStressGT-verify-benchmark",
        description="Run a verifier across all instances in a benchmark folder and log results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--benchmark", required=True, help="Benchmark directory containing manifest.json.")
    ap.add_argument("--verifier", required=True, choices=sorted(verifiers.keys()))
    ap.add_argument("--out_dir", required=True, help="Where to write logs/results.")
    ap.add_argument("--timeout", type=float, default=None, help="Per-instance timeout (seconds).")
    ap.add_argument("--jobs", type=int, default=1, help="Parallelism.")
    ap.add_argument("--max_memory_gb", type=float, default=None, help="Per-instance memory cap in GB (RLIMIT_AS). Exceeded processes are killed and recorded as ERROR.")
    ap.add_argument("--instances", type=str, nargs="*", default=None, help="Optional list of instance ids to run.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite out_dir if it exists.")

    args, remaining = ap.parse_known_args(argv)

    bench_dir = Path(args.benchmark).resolve()
    manifest = _load_manifest(bench_dir)

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} exists. Pass --overwrite to replace.")
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v = verifiers[args.verifier]
    vap = argparse.ArgumentParser(add_help=False)
    v.add_args(vap)
    vargs = vap.parse_args(remaining)

    conda_env = v.resolve_conda_env()
    if conda_env:
        print(f"[VeriStressGT] Will run {args.verifier} in conda env: {conda_env}")

    max_memory_bytes: Optional[int] = (
        int(args.max_memory_gb * 1024 ** 3) if args.max_memory_gb else None
    )
    if max_memory_bytes:
        if _rlimit_as_supported():
            print(f"[VeriStressGT] Per-instance memory cap: {args.max_memory_gb:.1f} GB (RLIMIT_AS)")
        else:
            print(
                f"[VeriStressGT] WARNING: RLIMIT_AS is not enforced on this platform "
                f"(macOS does not support it). --max_memory_gb will have no effect."
            )
            max_memory_bytes = None

    run_cfg = {
        "benchmark": str(bench_dir),
        "manifest_construction": manifest.get("construction"),
        "verifier": args.verifier,
        "conda_env": conda_env,
        "timeout": args.timeout,
        "jobs": args.jobs,
        "max_memory_gb": args.max_memory_gb,
        "git_commit": _git_commit(),
        "verifier_args": vars(vargs),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    jobs = _collect_jobs(bench_dir, manifest, args.instances)

    results_jsonl = out_dir / "results.jsonl"
    records: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=int(args.jobs)) as ex:
        futs = []
        for job in jobs:
            futs.append(
                ex.submit(
                    _run_one,
                    job,
                    args.verifier,
                    vargs,
                    v.build_cmd,
                    v.parse_result,
                    out_dir,
                    args.timeout,
                    conda_env,
                    max_memory_bytes,
                )
            )

        for i, fut in enumerate(as_completed(futs), start=1):
            rec = fut.result()
            records.append(rec)
            with results_jsonl.open("a") as f:
                f.write(json.dumps(rec) + "\n")

            base_msg = f"[{i}/{len(jobs)}] {rec['instance_id']} -> {rec['status']} ({rec['wall_time_s']:.2f}s)"

            if rec["status"] in {"ERROR", "UNKNOWN"}:
                preview = rec.get("error_preview")
                stderr_path = rec["log_paths"]["stderr"]
                stdout_path = rec["log_paths"]["stdout"]
                if preview:
                    print(f"{base_msg}\n    preview: {preview}\n    stderr: {stderr_path}\n    stdout: {stdout_path}")
                else:
                    print(f"{base_msg}\n    stderr: {stderr_path}\n    stdout: {stdout_path}")
            else:
                print(base_msg)

    _write_summary(out_dir, records)
    print(f"\nDone. Results: {results_jsonl}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()