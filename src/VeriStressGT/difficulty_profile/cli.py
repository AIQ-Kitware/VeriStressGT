#!/usr/bin/env python3
"""
CLI for estimating Difficulty Profiles of verification instances.

Usage:
  # Single instance
  python -m VeriStressGT.difficulty_profile.cli \
      --onnx model.onnx --vnnlib spec.vnnlib

  # Whole benchmark directory (reads manifest.json, instances.csv, or discovers ONNX/VNNLIB pairs)
  python -m VeriStressGT.difficulty_profile.cli \
      --benchmark-dir ./benchmarks/my_bench --out profiles.json

  # Skip expensive components
  python -m VeriStressGT.difficulty_profile.cli \
      --onnx model.onnx --vnnlib spec.vnnlib --skip-crown --skip-milp
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


def _discover_instances(bench_dir: Path) -> List[Tuple[str, str, Optional[str]]]:
    """
    Discover (onnx_path, vnnlib_path, instance_id) pairs in a benchmark directory.

    Order preference:
      1. manifest.json instance order
      2. instances.csv row order
      3. Matching *.onnx / *.vnnlib files by lexicographic path order
    """
    # 1) Prefer manifest.json, since create_benchmark.py writes it in spec order
    manifest_path = bench_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            pairs: List[Tuple[str, str, Optional[str]]] = []
            for inst in manifest.get("instances", []):
                if not isinstance(inst, dict):
                    continue
                inst_id = inst.get("id")
                paths = inst.get("paths", {}) or {}

                onnx_p = paths.get("onnx") or inst.get("onnx_path") or inst.get("onnx")
                vnnlib_p = paths.get("vnnlib") or inst.get("vnnlib_path") or inst.get("vnnlib")

                if onnx_p and vnnlib_p:
                    onnx_abs = str((bench_dir / onnx_p).resolve())
                    vnnlib_abs = str((bench_dir / vnnlib_p).resolve())
                    pairs.append((onnx_abs, vnnlib_abs, str(inst_id) if inst_id is not None else None))
            if pairs:
                return pairs
        except Exception:
            pass

    # 2) Backward-compatible: instances.csv row order
    instances_csv = bench_dir / "instances.csv"
    if instances_csv.exists():
        pairs: List[Tuple[str, str, Optional[str]]] = []
        with open(instances_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                onnx_p = row.get("onnx_path") or row.get("onnx")
                vnnlib_p = row.get("vnnlib_path") or row.get("vnnlib")
                inst_id = row.get("id")
                if onnx_p and vnnlib_p:
                    onnx_abs = str((bench_dir / onnx_p).resolve())
                    vnnlib_abs = str((bench_dir / vnnlib_p).resolve())
                    pairs.append((onnx_abs, vnnlib_abs, inst_id))
        if pairs:
            return pairs

    # 3) Fallback: lexicographic discovery
    onnx_files = sorted(bench_dir.rglob("*.onnx"))
    pairs: List[Tuple[str, str, Optional[str]]] = []
    for onnx_f in onnx_files:
        stem = onnx_f.stem
        vnnlib_candidates = list(onnx_f.parent.glob(f"{stem}*.vnnlib"))
        if not vnnlib_candidates:
            vnnlib_candidates = list(onnx_f.parent.glob("*.vnnlib"))
        if vnnlib_candidates:
            pairs.append(
                (
                    str(onnx_f.resolve()),
                    str(vnnlib_candidates[0].resolve()),
                    onnx_f.parent.name,
                )
            )
    return pairs


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Estimate Difficulty Profile for verification instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input mode: single instance or benchmark directory
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--onnx", type=str, help="Path to ONNX model (single instance mode).")
    grp.add_argument("--benchmark-dir", type=str, help="Path to benchmark directory.")

    ap.add_argument("--vnnlib", type=str, help="Path to VNNLIB spec (required for single instance).")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path. Default: <onnx_dir>/difficulty_profile.json "
             "or <bench_dir>/difficulty_profiles.json.",
    )

    # Estimation parameters
    ap.add_argument("--n-pgd-restarts", type=int, default=1)
    ap.add_argument("--n-pgd-steps", type=int, default=100)
    ap.add_argument("--n-gradient-samples", type=int, default=300)
    ap.add_argument("--n-beta-pairs", type=int, default=200)
    ap.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Tolerance fraction for A_tau upper bound.",
    )
    ap.add_argument(
        "--component-timeout",
        type=float,
        default=600.0,
        help="Max seconds per component estimation.",
    )

    # Compatibility flags
    ap.add_argument("--skip-crown", action="store_true", help="Accepted for compatibility; currently ignored.")
    ap.add_argument("--skip-milp", action="store_true", help="Accepted for compatibility; currently ignored.")

    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args(argv)

    from .profile import estimate_profile, DifficultyProfile

    verbose = not args.quiet

    # Collect instances
    if args.onnx:
        if not args.vnnlib:
            ap.error("--vnnlib is required when using --onnx.")
        instances: List[Tuple[str, str, Optional[str]]] = [
            (args.onnx, args.vnnlib, Path(args.onnx).parent.name)
        ]
        default_out = str(Path(args.onnx).parent / "difficulty_profile.json")
    else:
        bench_dir = Path(args.benchmark_dir)
        if not bench_dir.is_dir():
            ap.error(f"Benchmark directory not found: {bench_dir}")
        instances = _discover_instances(bench_dir)
        if not instances:
            ap.error(f"No ONNX/VNNLIB pairs found in {bench_dir}")
        default_out = str(bench_dir / "difficulty_profiles.json")
        if verbose:
            source = (
                "manifest.json"
                if (bench_dir / "manifest.json").exists()
                else "instances.csv"
                if (bench_dir / "instances.csv").exists()
                else "alphabetical filesystem discovery"
            )
            print(f"Found {len(instances)} instances in {bench_dir}")
            print(f"Discovery order source: {source}")

    out_path = args.out or default_out

    # Run estimation
    all_profiles = []
    total_start = time.time()

    for i, (onnx_path, vnnlib_path, instance_id) in enumerate(instances):
        display_name = instance_id or Path(onnx_path).parent.name or Path(onnx_path).name
        if verbose:
            print(f"\n{'='*60}")
            print(f"Instance {i+1}/{len(instances)}: {display_name}")
            print(f"{'='*60}")

        try:
            profile = estimate_profile(
                onnx_path=onnx_path,
                vnnlib_path=vnnlib_path,
                n_pgd_restarts=args.n_pgd_restarts,
                n_pgd_steps=args.n_pgd_steps,
                n_gradient_samples=args.n_gradient_samples,
                n_beta_pairs=args.n_beta_pairs,
                tau=args.tau,
                component_timeout=args.component_timeout,
                device=args.device,
                verbose=verbose,
            )

            pd = profile.to_dict()

            # Add identifying metadata without assuming anything else about the schema
            if isinstance(pd, dict):
                pd.setdefault("instance_id", display_name)
                pd.setdefault("onnx_path", onnx_path)
                pd.setdefault("vnnlib_path", vnnlib_path)

            all_profiles.append(pd)

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            all_profiles.append({
                "instance_id": display_name,
                "onnx_path": onnx_path,
                "vnnlib_path": vnnlib_path,
                "error": str(e),
            })

    # Save results
    total_time = time.time() - total_start

    if len(all_profiles) == 1:
        output = all_profiles[0]
    else:
        output = {
            "instances": all_profiles,
            "total_time_s": total_time,
            "n_instances": len(all_profiles),
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(output, indent=2))

    if verbose:
        print(f"\nSaved profiles to {out_path}")
        print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()