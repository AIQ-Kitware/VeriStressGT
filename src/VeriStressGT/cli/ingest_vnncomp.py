# src/VeriStressGT/cli/ingest_vnncomp.py
"""
Ingest a VNN-COMP benchmark folder (e.g. `vnncomp2022_benchmarks/benchmarks/mnist_fc/`)
and emit a VeriStressGT-compatible benchmark directory that verify_benchmark.py
can consume directly.

A VNN-COMP benchmark has the canonical layout:
    <src_benchmark>/
        instances.csv       # rows: <onnx_rel_path>,<vnnlib_rel_path>,<timeout_seconds>
        onnx/*.onnx         # (typical, not required)
        vnnlib/*.vnnlib     # (typical, not required)

The output layout matches what create_benchmark.py writes:
    <out_dir>/
        manifest.json
        instances/
            000001/
                model.onnx
                spec.vnnlib
                meta.json

Ground-truth robustness is intentionally left null: VNN-COMP instances have no
a-priori known labels (majority-vote proxies from verifier outputs are exactly
the gap VeriStressGT is designed to fill). The per-instance meta.json
records `is_robust: null` and `ground_truth_source: "vnncomp_unknown"` so
downstream analysis can filter / join as needed.

Usage:
    python -m VeriStressGT.cli.ingest_vnncomp \\
        --src /path/to/vnncomp2022_benchmarks/benchmarks/mnist_fc \\
        --out_dir ./benchmarks/vnncomp_mnist_fc \\
        --name vnncomp_mnist_fc \\
        --overwrite

    # Subset by max instances or timeout ceiling:
    python -m VeriStressGT.cli.ingest_vnncomp \\
        --src /path/to/vnncomp2022_benchmarks/benchmarks/mnist_fc \\
        --out_dir ./benchmarks/vnncomp_mnist_fc_small \\
        --max_instances 20 \\
        --max_timeout 120 \\
        --overwrite
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers (copied locally rather than imported to keep ingest standalone)
# ---------------------------------------------------------------------------

def _git_commit() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return None


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


# ---------------------------------------------------------------------------
# instances.csv parsing
# ---------------------------------------------------------------------------

def _parse_instances_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """
    VNN-COMP `instances.csv` is headerless; columns are:
        onnx_rel_path, vnnlib_rel_path, timeout_seconds

    Some older releases add a trailing blank line or occasional comment lines
    starting with '#'. We strip both.
    """
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for lineno, raw in enumerate(reader, start=1):
            if not raw:
                continue
            # Trim whitespace on every cell and skip blank / comment lines.
            cells = [c.strip() for c in raw]
            if not any(cells):
                continue
            if cells[0].startswith("#"):
                continue
            if len(cells) < 2:
                raise ValueError(
                    f"{csv_path}:{lineno}: expected at least (onnx, vnnlib) columns, "
                    f"got {cells!r}"
                )
            onnx_rel = cells[0]
            vnnlib_rel = cells[1]
            timeout_s: Optional[float] = None
            if len(cells) >= 3 and cells[2] != "":
                try:
                    timeout_s = float(cells[2])
                except ValueError:
                    raise ValueError(
                        f"{csv_path}:{lineno}: timeout column is not a float: {cells[2]!r}"
                    )
            rows.append({
                "onnx_rel": onnx_rel,
                "vnnlib_rel": vnnlib_rel,
                "timeout_s": timeout_s,
                "source_lineno": lineno,
            })
    if not rows:
        raise ValueError(f"{csv_path}: no instances parsed")
    return rows


# ---------------------------------------------------------------------------
# Ingest core
# ---------------------------------------------------------------------------

def ingest(
    *,
    src_dir: Path,
    out_dir: Path,
    name: Optional[str],
    overwrite: bool,
    max_instances: Optional[int],
    max_timeout: Optional[float],
    onnx_name: str,
    vnnlib_name: str,
    meta_name: str,
) -> Path:
    src_dir = src_dir.resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"--src is not a directory: {src_dir}")

    csv_path = src_dir / "instances.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected {csv_path} — this does not look like a VNN-COMP benchmark folder."
        )

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists. Pass --overwrite to replace.")
        _safe_rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances_dir = out_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    bench_name = name or f"vnncomp_{src_dir.name}"

    rows = _parse_instances_csv(csv_path)

    # Optional filtering
    if max_timeout is not None:
        rows = [r for r in rows if (r["timeout_s"] is None or r["timeout_s"] <= max_timeout)]
    if max_instances is not None:
        rows = rows[: int(max_instances)]
    if not rows:
        raise ValueError("No instances left after filtering.")

    manifest: Dict[str, Any] = {
        "name": bench_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
        "common": {
            "onnx_name": onnx_name,
            "vnnlib_name": vnnlib_name,
            "meta_name": meta_name,
        },
        # Marker so downstream tooling can distinguish these from native constructions.
        "mode": "vnncomp_ingest",
        "source": {
            "kind": "vnncomp",
            "src_dir": str(src_dir),
            "instances_csv": str(csv_path),
            "instances_csv_sha256": _sha256_file(csv_path),
            "num_rows_total": len(_parse_instances_csv(csv_path)),
            "num_rows_ingested": len(rows),
            "max_instances": max_instances,
            "max_timeout": max_timeout,
        },
        "instances": [],
    }

    for idx, row in enumerate(rows, start=1):
        inst_id = f"{idx:06d}"
        inst_dir = instances_dir / inst_id
        inst_dir.mkdir(parents=True, exist_ok=True)

        src_onnx = (src_dir / row["onnx_rel"]).resolve()
        src_vnnlib = (src_dir / row["vnnlib_rel"]).resolve()

        if not src_onnx.is_file():
            raise FileNotFoundError(
                f"Instance {inst_id} ({csv_path.name}:{row['source_lineno']}): "
                f"onnx not found: {src_onnx}"
            )
        if not src_vnnlib.is_file():
            raise FileNotFoundError(
                f"Instance {inst_id} ({csv_path.name}:{row['source_lineno']}): "
                f"vnnlib not found: {src_vnnlib}"
            )

        dst_onnx = inst_dir / onnx_name
        dst_vnnlib = inst_dir / vnnlib_name
        dst_meta = inst_dir / meta_name

        shutil.copy2(src_onnx, dst_onnx)
        shutil.copy2(src_vnnlib, dst_vnnlib)

        inst_meta: Dict[str, Any] = {
            "id": inst_id,
            # Sentinel: this instance did not come from a native construction.
            "construction": "vnncomp.ingested",
            "seed": None,
            "paths": {
                "onnx": str(dst_onnx.relative_to(out_dir)),
                "vnnlib": str(dst_vnnlib.relative_to(out_dir)),
                "meta": str(dst_meta.relative_to(out_dir)),
            },
            "sha256": {
                "onnx": _sha256_file(dst_onnx),
                "vnnlib": _sha256_file(dst_vnnlib),
            },
            # Ground-truth is explicitly unknown — the whole point of
            # VeriStressGT is that VNN-COMP instances lack this.
            "is_robust": None,
            "ground_truth_source": "vnncomp_unknown",
            "vnncomp": {
                "onnx_rel": row["onnx_rel"],
                "vnnlib_rel": row["vnnlib_rel"],
                "timeout_s": row["timeout_s"],
                "source_lineno": row["source_lineno"],
            },
        }
        dst_meta.write_text(json.dumps(inst_meta, indent=2))
        manifest["instances"].append(inst_meta)

        print(f"[{idx}/{len(rows)}] {inst_id}  <-  {row['onnx_rel']}  +  {row['vnnlib_rel']}"
              + (f"  (timeout={row['timeout_s']}s)" if row["timeout_s"] is not None else ""))

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Ingested {len(rows)} instance(s).")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="VeriStressGT-ingest-vnncomp",
        description=(
            "Ingest a VNN-COMP benchmark folder (containing instances.csv) into "
            "VeriStressGT format so verify_benchmark.py can consume it."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", required=True,
                    help="Path to a VNN-COMP benchmark folder (contains instances.csv).")
    ap.add_argument("--out_dir", required=True,
                    help="Output benchmark directory (will be created).")
    ap.add_argument("--name", default=None,
                    help="Benchmark name (default: vnncomp_<src folder name>).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite out_dir if it exists.")
    ap.add_argument("--max_instances", type=int, default=None,
                    help="Cap the number of ingested instances (after timeout filter).")
    ap.add_argument("--max_timeout", type=float, default=None,
                    help="Drop instances whose VNN-COMP suggested timeout exceeds this (seconds).")
    ap.add_argument("--onnx_name", default="model.onnx")
    ap.add_argument("--vnnlib_name", default="spec.vnnlib")
    ap.add_argument("--meta_name", default="meta.json")
    args = ap.parse_args(argv)

    ingest(
        src_dir=Path(args.src),
        out_dir=Path(args.out_dir),
        name=args.name,
        overwrite=args.overwrite,
        max_instances=args.max_instances,
        max_timeout=args.max_timeout,
        onnx_name=args.onnx_name,
        vnnlib_name=args.vnnlib_name,
        meta_name=args.meta_name,
    )


if __name__ == "__main__":
    main()