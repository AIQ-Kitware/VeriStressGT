#!/usr/bin/env python3
"""
Quick diagnostic for a VNN-COMP benchmark directory.

Run this BEFORE preprocessing to surface any surprises in the ONNX op set,
input shapes, or spec structure. If the output reports an op that's not
handled in difficulty_profile.components._interval_propagate, we want to
know up front rather than silently default-casing the interval through.

Usage
-----
    python3 scripts/inspect_vnncomp_benchmark.py \
        --benchmark-dir ./vnncomp2022_benchmarks/benchmarks/cifar_biasfield

Works for benchmarks with .onnx[.gz] + .vnnlib[.gz] + instances.csv.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---- ONNX ops we know how to propagate intervals through ------------------
# Keep this in sync with _interval_propagate in components.py.
HANDLED_OPS = {
    "Conv", "MatMul", "Gemm", "Add", "Relu", "Sigmoid", "Tanh",
    "Flatten", "Reshape", "Squeeze", "Unsqueeze", "Transpose",
    "Identity", "Dropout", "Concat", "Mul", "Softmax",
    "BatchNormalization", "Constant",
}


def _maybe_decompress(path: Path, tmpdir: Path) -> Path:
    """If path ends in .gz, decompress into tmpdir and return new path."""
    if path.suffix == ".gz":
        out = tmpdir / path.stem  # strips .gz
        with gzip.open(path, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return out
    return path


def _find_instances_csv(root: Path) -> Optional[Path]:
    """
    VNN-COMP typically ships instances.csv at the benchmark root. A few
    benchmarks nest it one level down. Look in both.
    """
    candidates = [root / "instances.csv"]
    for child in root.iterdir():
        if child.is_dir():
            candidates.append(child / "instances.csv")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _read_instances_csv(csv_path: Path) -> List[Tuple[str, str, Optional[float]]]:
    """
    Return [(onnx_rel, vnnlib_rel, timeout_s)] rows.
    instances.csv has no header and 3 columns.
    """
    rows: List[Tuple[str, str, Optional[float]]] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                continue
            onnx = row[0].strip()
            vnnlib = row[1].strip()
            timeout: Optional[float] = None
            if len(row) >= 3 and row[2].strip():
                try:
                    timeout = float(row[2].strip())
                except ValueError:
                    pass
            rows.append((onnx, vnnlib, timeout))
    return rows


def _resolve(root: Path, p: str) -> Path:
    """Instances.csv paths are usually relative to the benchmark root."""
    q = Path(p)
    if q.is_absolute():
        return q
    candidate = root / q
    if candidate.exists():
        return candidate
    # Sometimes paths have an extra leading "./" or a stripped .gz
    for alt in (root / q.name, root / "onnx" / q.name, root / "vnnlib" / q.name):
        if alt.exists():
            return alt
        gz = alt.with_suffix(alt.suffix + ".gz")
        if gz.exists():
            return gz
    # Try with .gz appended
    gz = root / (str(q) + ".gz")
    if gz.exists():
        return gz
    return candidate  # will fail downstream but give a clear path


def _inspect_onnx(onnx_path: Path, tmpdir: Path) -> Dict[str, Any]:
    import onnx

    actual = _maybe_decompress(onnx_path, tmpdir)
    model = onnx.load(str(actual))
    ops = [node.op_type for node in model.graph.node]
    op_counts = collections.Counter(ops)

    # Input shape
    input_shape: List[Any] = []
    try:
        dims = model.graph.input[0].type.tensor_type.shape.dim
        for d in dims:
            if d.dim_value > 0:
                input_shape.append(d.dim_value)
            else:
                input_shape.append("?")
    except Exception:
        pass

    # Output shape
    output_shape: List[Any] = []
    try:
        dims = model.graph.output[0].type.tensor_type.shape.dim
        for d in dims:
            if d.dim_value > 0:
                output_shape.append(d.dim_value)
            else:
                output_shape.append("?")
    except Exception:
        pass

    return {
        "ops": op_counts,
        "n_nodes": len(ops),
        "input_shape": input_shape,
        "output_shape": output_shape,
    }


def _inspect_vnnlib(vnnlib_path: Path, tmpdir: Path) -> Dict[str, Any]:
    """
    Shallow scan: we're not parsing, just looking for coarse signals.
    - does it use l_inf-style box bounds?
    - does it use AND/OR mixed specs?
    - how large is it?
    """
    actual = _maybe_decompress(vnnlib_path, tmpdir)
    text = actual.read_text(errors="ignore")
    return {
        "n_bytes": actual.stat().st_size,
        "n_lines": text.count("\n") + 1,
        "has_assert_and": "(and " in text,
        "has_assert_or": "(or " in text,
        "n_declare_const": text.count("(declare-const "),
        "n_assert": text.count("(assert "),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-dir", required=True,
                    help="Path to one benchmark, e.g. .../benchmarks/cifar_biasfield")
    ap.add_argument("--sample", type=int, default=3,
                    help="How many ONNX files to inspect in detail (default 3)")
    args = ap.parse_args()

    root = Path(args.benchmark_dir).expanduser().resolve()
    print(f"Inspecting benchmark at: {root}")
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    csv_path = _find_instances_csv(root)
    if csv_path is None:
        raise SystemExit(
            "No instances.csv found. Expected it at the benchmark root "
            "or one level down. Aborting."
        )
    print(f"  instances.csv: {csv_path.relative_to(root)}")

    rows = _read_instances_csv(csv_path)
    print(f"  rows in instances.csv: {len(rows)}")
    if not rows:
        raise SystemExit("Empty instances.csv; nothing to inspect.")

    timeouts = [r[2] for r in rows if r[2] is not None]
    if timeouts:
        print(f"  timeouts (from CSV): min={min(timeouts):.0f}s  "
              f"max={max(timeouts):.0f}s  unique={sorted(set(timeouts))}")

    # Unique ONNX files — repeated per instance usually.
    unique_onnx = sorted(set(r[0] for r in rows))
    print(f"  unique ONNX files: {len(unique_onnx)}")

    with tempfile.TemporaryDirectory() as td_str:
        tmpdir = Path(td_str)

        # Inspect up to `sample` unique ONNX files in detail.
        print("\nONNX op-type inventory (aggregated across sampled files):")
        aggregated = collections.Counter()
        inspected = 0
        for onnx_rel in unique_onnx:
            if inspected >= args.sample:
                break
            onnx_abs = _resolve(root, onnx_rel)
            if not onnx_abs.exists():
                print(f"  [missing] {onnx_rel}")
                continue
            try:
                info = _inspect_onnx(onnx_abs, tmpdir)
            except Exception as e:
                print(f"  [unreadable] {onnx_rel}: {e}")
                continue
            print(f"\n  {onnx_rel}")
            print(f"    input_shape  = {info['input_shape']}")
            print(f"    output_shape = {info['output_shape']}")
            print(f"    n_nodes      = {info['n_nodes']}")
            for op, c in info["ops"].most_common():
                marker = "" if op in HANDLED_OPS else "  <-- UNHANDLED"
                print(f"      {op:<24s} {c:>5d}{marker}")
            aggregated.update(info["ops"])
            inspected += 1

        print("\nAggregated op counts across sampled files:")
        unhandled = []
        for op, c in aggregated.most_common():
            marker = "" if op in HANDLED_OPS else "  <-- UNHANDLED"
            if op not in HANDLED_OPS:
                unhandled.append(op)
            print(f"  {op:<24s} {c:>5d}{marker}")

        if unhandled:
            print(f"\nHEADS UP: these ops are NOT in HANDLED_OPS and would be "
                  f"default-cased in _interval_propagate: {sorted(set(unhandled))}")
            print("Consider adding handlers before running profile estimation.")
        else:
            print("\nAll sampled ops are handled by _interval_propagate.")

        # Inspect one vnnlib in detail.
        print("\nvnnlib sample:")
        sample_vnnlib_rel = rows[0][1]
        vnnlib_abs = _resolve(root, sample_vnnlib_rel)
        if vnnlib_abs.exists():
            try:
                vinfo = _inspect_vnnlib(vnnlib_abs, tmpdir)
                print(f"  {sample_vnnlib_rel}")
                for k, v in vinfo.items():
                    print(f"    {k:<20s} = {v}")
            except Exception as e:
                print(f"  [error reading {sample_vnnlib_rel}]: {e}")
        else:
            print(f"  [missing] {sample_vnnlib_rel}")


if __name__ == "__main__":
    main()