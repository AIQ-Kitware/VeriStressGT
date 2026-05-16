# #!/usr/bin/env python3
# """
# Convert a VNN-COMP benchmark into the VeriStressGT instances/<id>/ layout.

# Output structure (matches every other benchmark in the repo)
# ------------------------------------------------------------

#     <out-dir>/
#         manifest.json
#         instances/
#             vb_mnist_fc_0000/
#                 meta.json
#                 model.onnx
#                 spec.vnnlib
#             vb_mnist_fc_0001/
#                 meta.json
#                 model.onnx
#                 spec.vnnlib
#             ...

# manifest.json schema (matches the repo convention)
# --------------------------------------------------

#     {
#       "benchmark_name": "vnncomp22_cifar_biasfield",
#       "source": "/.../benchmarks/cifar_biasfield",
#       "n_instances": 72,
#       "instances": [
#         {
#           "id": "vb_cifar_biasfield_0000",
#           "seed": null,
#           "construction": "vnncomp.cifar_biasfield",
#           "paths": {
#             "onnx":   "instances/vb_cifar_biasfield_0000/model.onnx",
#             "vnnlib": "instances/vb_cifar_biasfield_0000/spec.vnnlib",
#             "meta":   "instances/vb_cifar_biasfield_0000/meta.json"
#           },
#           "args": { "epsilon": 0.01, "spec_index": 3, "onnx_basename": "...", ... },
#           "vnncomp_timeout_s": 300
#         },
#         ...
#       ]
#     }

# Usage
# -----

#     python3 scripts/preprocess_vnncomp_benchmark.py \\
#         --benchmark-dir ./vnncomp2022_benchmarks/benchmarks/cifar_biasfield \\
#         --out-dir       ./benchmarks/vnncomp22_cifar_biasfield

#     python3 scripts/preprocess_vnncomp_benchmark.py \\
#         --benchmark-dir ./vnncomp2022_benchmarks/benchmarks/mnist_fc \\
#         --out-dir       ./benchmarks/vnncomp22_mnist_fc

# Notes
# -----
# - Handles .gz-compressed inputs.
# - Paths written to manifest.json are RELATIVE to out_dir, matching the
#   other benchmarks in the repo.
# - Construction label defaults to "vnncomp.<benchmark_name>". Pass
#   --split-by-onnx to use "vnncomp.<benchmark>.<onnx_stem>" so that each
#   underlying network in the benchmark becomes its own construction
#   family for within-construction correlation analysis.
# """
# from __future__ import annotations

# import argparse
# import csv
# import gzip
# import json
# import re
# import shutil
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple


# # ---------------------------------------------------------------------------
# # instances.csv discovery and parsing
# # ---------------------------------------------------------------------------

# def _find_instances_csv(root: Path) -> Optional[Path]:
#     candidates = [root / "instances.csv"]
#     for child in sorted(root.iterdir()):
#         if child.is_dir():
#             candidates.append(child / "instances.csv")
#     for c in candidates:
#         if c.is_file():
#             return c
#     return None


# def _read_instances_csv(csv_path: Path) -> List[Tuple[str, str, Optional[float]]]:
#     rows: List[Tuple[str, str, Optional[float]]] = []
#     with open(csv_path, newline="") as f:
#         reader = csv.reader(f)
#         for row in reader:
#             if not row or row[0].startswith("#"):
#                 continue
#             if len(row) < 2:
#                 continue
#             onnx = row[0].strip()
#             vnnlib = row[1].strip()
#             timeout: Optional[float] = None
#             if len(row) >= 3 and row[2].strip():
#                 try:
#                     timeout = float(row[2].strip())
#                 except ValueError:
#                     pass
#             rows.append((onnx, vnnlib, timeout))
#     return rows


# def _resolve(root: Path, p: str) -> Optional[Path]:
#     q = Path(p)
#     base = q if q.is_absolute() else (root / q)
#     if base.exists():
#         return base
#     gz = base.with_suffix(base.suffix + ".gz")
#     if gz.exists():
#         return gz
#     for alt in (root / "onnx" / q.name, root / "vnnlib" / q.name):
#         if alt.exists():
#             return alt
#         gz = alt.with_suffix(alt.suffix + ".gz")
#         if gz.exists():
#             return gz
#     return None


# # ---------------------------------------------------------------------------
# # Materialize files into the instances/<id>/ layout
# # ---------------------------------------------------------------------------

# def _write_decompressed(src: Path, dst: Path) -> None:
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     if dst.exists():
#         return
#     if src.suffix == ".gz":
#         with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
#             shutil.copyfileobj(fin, fout)
#     else:
#         shutil.copy2(src, dst)


# # ---------------------------------------------------------------------------
# # Filename heuristics (epsilon, spec index)
# # ---------------------------------------------------------------------------

# _EPS_PATTERN = re.compile(
#     r"""(?:^|[_\-./])(?:eps|epsilon|e)[_\-]?(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)""",
#     re.VERBOSE,
# )
# _TRAILING_DECIMAL = re.compile(r"(?:^|[_\-])(\d+\.\d+)(?=\.[a-zA-Z]+$)")


# def _parse_filename_hints(onnx_rel: str, vnnlib_rel: str) -> Dict[str, Any]:
#     hints: Dict[str, Any] = {
#         "onnx_basename": Path(onnx_rel).name,
#         "vnnlib_basename": Path(vnnlib_rel).name,
#     }

#     for name in (vnnlib_rel, onnx_rel):
#         m = _EPS_PATTERN.search(name)
#         if m:
#             try:
#                 hints["epsilon"] = float(m.group(1))
#                 break
#             except ValueError:
#                 pass
#     if "epsilon" not in hints:
#         for name in (vnnlib_rel, onnx_rel):
#             m = _TRAILING_DECIMAL.search(Path(name).name)
#             if m:
#                 try:
#                     hints["epsilon"] = float(m.group(1))
#                     break
#                 except ValueError:
#                     pass

#     idx = re.search(r"(?:idx|img|prop|spec)[_\-]?(\d+)", vnnlib_rel)
#     if idx:
#         try:
#             hints["spec_index"] = int(idx.group(1))
#         except ValueError:
#             pass

#     return hints


# def _onnx_stem_for_label(onnx_rel: str) -> str:
#     """
#     Derive a short, filesystem-safe label from the ONNX filename for use as
#     a construction suffix when --split-by-onnx is passed. Strips common
#     VNN-COMP prefixes and extensions.
#     """
#     stem = Path(onnx_rel).name
#     for suffix in (".onnx.gz", ".onnx"):
#         if stem.endswith(suffix):
#             stem = stem[: -len(suffix)]
#             break
#     stem = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
#     return stem or "net"


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main() -> None:
#     ap = argparse.ArgumentParser(description=__doc__,
#                                  formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--benchmark-dir", required=True,
#                     help="Path to one VNN-COMP benchmark, e.g. .../benchmarks/cifar_biasfield")
#     ap.add_argument("--out-dir", required=True,
#                     help="Destination directory (will contain manifest.json and instances/)")
#     ap.add_argument("--id-prefix", default=None,
#                     help="Per-instance id prefix (default: derived from benchmark name, "
#                          "e.g. 'vb_cifar_biasfield')")
#     ap.add_argument("--construction-override", default=None,
#                     help="Force a specific construction label instead of vnncomp.<n>")
#     ap.add_argument("--split-by-onnx", action="store_true",
#                     help="Use a separate construction label per underlying ONNX file, "
#                          "i.e. vnncomp.<benchmark>.<onnx_stem>. Useful when a benchmark "
#                          "bundles multiple networks.")
#     ap.add_argument("--limit", type=int, default=None,
#                     help="Only preprocess the first N instances (for fast iteration)")
#     ap.add_argument("--overwrite", action="store_true",
#                     help="Overwrite out_dir/instances if they already exist")
#     args = ap.parse_args()

#     root = Path(args.benchmark_dir).expanduser().resolve()
#     out_dir = Path(args.out_dir).expanduser().resolve()

#     if not root.is_dir():
#         raise SystemExit(f"Not a directory: {root}")

#     csv_path = _find_instances_csv(root)
#     if csv_path is None:
#         raise SystemExit(f"No instances.csv under {root}")

#     rows = _read_instances_csv(csv_path)
#     if not rows:
#         raise SystemExit(f"No rows in {csv_path}")
#     if args.limit is not None:
#         rows = rows[: args.limit]

#     instances_root = out_dir / "instances"
#     if instances_root.exists() and not args.overwrite:
#         raise SystemExit(
#             f"{instances_root} already exists; pass --overwrite to replace."
#         )
#     out_dir.mkdir(parents=True, exist_ok=True)
#     instances_root.mkdir(parents=True, exist_ok=True)

#     benchmark_name = root.name
#     id_prefix = args.id_prefix or f"vb_{benchmark_name}"
#     base_construction = args.construction_override or f"vnncomp.{benchmark_name}"

#     manifest_instances: List[Dict[str, Any]] = []
#     missing: List[Tuple[str, str]] = []

#     for i, (onnx_rel, vnnlib_rel, timeout) in enumerate(rows):
#         onnx_src = _resolve(root, onnx_rel)
#         vnnlib_src = _resolve(root, vnnlib_rel)
#         if onnx_src is None or vnnlib_src is None:
#             missing.append((onnx_rel, vnnlib_rel))
#             continue

#         inst_id = f"{id_prefix}_{i:04d}"
#         inst_dir = instances_root / inst_id
#         inst_dir.mkdir(parents=True, exist_ok=True)

#         onnx_dst = inst_dir / "model.onnx"
#         vnnlib_dst = inst_dir / "spec.vnnlib"
#         _write_decompressed(onnx_src, onnx_dst)
#         _write_decompressed(vnnlib_src, vnnlib_dst)

#         hints = _parse_filename_hints(onnx_rel, vnnlib_rel)

#         if args.construction_override is not None:
#             construction = args.construction_override
#         elif args.split_by_onnx:
#             construction = f"{base_construction}.{_onnx_stem_for_label(onnx_rel)}"
#         else:
#             construction = base_construction

#         meta_dst = inst_dir / "meta.json"

#         entry = {
#             "id": inst_id,
#             "seed": None,
#             "construction": construction,
#             "paths": {
#                 "onnx": str(onnx_dst.relative_to(out_dir)),
#                 "vnnlib": str(vnnlib_dst.relative_to(out_dir)),
#                 "meta": str(meta_dst.relative_to(out_dir)),
#             },
#             "args": hints,
#         }
#         if timeout is not None:
#             entry["vnncomp_timeout_s"] = timeout

#         meta = {
#             "id": inst_id,
#             "construction": construction,
#             "source_benchmark": benchmark_name,
#             "source_onnx": onnx_rel,
#             "source_vnnlib": vnnlib_rel,
#             "args": hints,
#         }
#         if timeout is not None:
#             meta["vnncomp_timeout_s"] = timeout
#         meta_dst.write_text(json.dumps(meta, indent=2))

#         manifest_instances.append(entry)

#     manifest = {
#         "benchmark_name": f"vnncomp22_{benchmark_name}",
#         "source": str(root),
#         "construction": base_construction,
#         "n_instances": len(manifest_instances),
#         "instances": manifest_instances,
#     }
#     (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

#     print(f"Wrote {out_dir / 'manifest.json'}")
#     print(f"  instances materialized : {len(manifest_instances)}")
#     print(f"  missing files          : {len(missing)}")
#     if missing:
#         for m in missing[:5]:
#             print(f"    {m[0]} | {m[1]}")
#         if len(missing) > 5:
#             print(f"    ... and {len(missing) - 5} more")

#     eps_vals = sorted({
#         inst["args"].get("epsilon") for inst in manifest_instances
#         if inst["args"].get("epsilon") is not None
#     })
#     if eps_vals:
#         print(f"  epsilons parsed from filenames: {eps_vals}")

#     if args.split_by_onnx:
#         constructions = sorted({inst["construction"] for inst in manifest_instances})
#         print(f"  construction labels ({len(constructions)}):")
#         for c in constructions:
#             n = sum(1 for i in manifest_instances if i["construction"] == c)
#             print(f"    {c:<60s} n={n}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Convert a VNN-COMP benchmark into the VeriStressGT instances/<id>/ layout.

Output structure (matches every other benchmark in the repo)
------------------------------------------------------------

    <out-dir>/
        manifest.json
        instances/
            vb_mnist_fc_0000/
                meta.json
                model.onnx
                spec.vnnlib
            vb_mnist_fc_0001/
                meta.json
                model.onnx
                spec.vnnlib
            ...

manifest.json schema (matches the repo convention)
--------------------------------------------------

    {
      "benchmark_name": "vnncomp22_cifar_biasfield",
      "source": "/.../benchmarks/cifar_biasfield",
      "n_instances": 72,
      "instances": [
        {
          "id": "vb_cifar_biasfield_0000",
          "seed": null,
          "construction": "vnncomp.cifar_biasfield",
          "paths": {
            "onnx":   "instances/vb_cifar_biasfield_0000/model.onnx",
            "vnnlib": "instances/vb_cifar_biasfield_0000/spec.vnnlib",
            "meta":   "instances/vb_cifar_biasfield_0000/meta.json"
          },
          "args": { "epsilon": 0.01, "spec_index": 3, "onnx_basename": "...", ... },
          "vnncomp_timeout_s": 300
        },
        ...
      ]
    }

Usage
-----

    python3 scripts/preprocess_vnncomp_benchmark.py \\
        --benchmark-dir ./vnncomp2022_benchmarks/benchmarks/cifar_biasfield \\
        --out-dir       ./benchmarks/vnncomp22_cifar_biasfield

    python3 scripts/preprocess_vnncomp_benchmark.py \\
        --benchmark-dir ./vnncomp2022_benchmarks/benchmarks/cifar2020 \\
        --out-dir       ./benchmarks/vnncomp22_cifar2020 \\
        --onnx-opset    13 \\
        --overwrite

Notes
-----
- Handles .gz-compressed inputs.
- Optionally converts materialized ONNX models to an older opset for
  verifier compatibility, e.g. --onnx-opset 13 for nnenum.
- Paths written to manifest.json are RELATIVE to out_dir, matching the
  other benchmarks in the repo.
- Construction label defaults to "vnncomp.<benchmark_name>". Pass
  --split-by-onnx to use "vnncomp.<benchmark>.<onnx_stem>" so that each
  underlying network in the benchmark becomes its own construction
  family for within-construction correlation analysis.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import onnx
from onnx import version_converter


# ---------------------------------------------------------------------------
# instances.csv discovery and parsing
# ---------------------------------------------------------------------------

def _find_instances_csv(root: Path) -> Optional[Path]:
    candidates = [root / "instances.csv"]
    for child in sorted(root.iterdir()):
        if child.is_dir():
            candidates.append(child / "instances.csv")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _read_instances_csv(csv_path: Path) -> List[Tuple[str, str, Optional[float]]]:
    rows: List[Tuple[str, str, Optional[float]]] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                continue
            onnx_rel = row[0].strip()
            vnnlib_rel = row[1].strip()
            timeout: Optional[float] = None
            if len(row) >= 3 and row[2].strip():
                try:
                    timeout = float(row[2].strip())
                except ValueError:
                    pass
            rows.append((onnx_rel, vnnlib_rel, timeout))
    return rows


def _resolve(root: Path, p: str) -> Optional[Path]:
    q = Path(p)
    base = q if q.is_absolute() else (root / q)
    if base.exists():
        return base
    gz = base.with_suffix(base.suffix + ".gz")
    if gz.exists():
        return gz
    for alt in (root / "onnx" / q.name, root / "vnnlib" / q.name):
        if alt.exists():
            return alt
        gz = alt.with_suffix(alt.suffix + ".gz")
        if gz.exists():
            return gz
    return None


# ---------------------------------------------------------------------------
# Materialize files into the instances/<id>/ layout
# ---------------------------------------------------------------------------

def _write_decompressed(src: Path, dst: Path, *, overwrite: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        return

    if src.suffix == ".gz":
        with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    else:
        shutil.copy2(src, dst)


def _get_ai_onnx_opset(model: onnx.ModelProto) -> Optional[int]:
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return int(imp.version)
    return None


def _convert_onnx_opset_inplace(path: Path, target_opset: int) -> Tuple[Optional[int], Optional[int], bool]:
    """
    Convert an ONNX model to target ai.onnx opset in-place.

    Returns:
        (old_opset, new_opset, converted)

    Notes:
    - ONNX's version_converter is not perfect. If it fails, the better fix is
      to re-export the original model with torch.onnx.export(..., opset_version=13).
    - We only down-convert. If model opset is already <= target_opset, we leave
      it alone.
    """
    model = onnx.load(str(path))
    old_opset = _get_ai_onnx_opset(model)

    if old_opset is None:
        raise RuntimeError(f"Could not find ai.onnx opset in {path}")

    if old_opset == target_opset:
        return old_opset, old_opset, False

    if old_opset < target_opset:
        print(
            f"  warning: {path.name} has ai.onnx opset {old_opset} < target "
            f"{target_opset}; leaving unchanged"
        )
        return old_opset, old_opset, False

    try:
        converted = version_converter.convert_version(model, target_opset)
        onnx.checker.check_model(converted)
        onnx.save(converted, str(path))
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert {path} from ai.onnx opset {old_opset} "
            f"to {target_opset}. Try re-exporting the original model with "
            f"torch.onnx.export(..., opset_version={target_opset}). "
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    converted_model = onnx.load(str(path))
    new_opset = _get_ai_onnx_opset(converted_model)

    print(f"  converted {path} ai.onnx opset {old_opset} -> {new_opset}")
    return old_opset, new_opset, True


# ---------------------------------------------------------------------------
# Filename heuristics (epsilon, spec index)
# ---------------------------------------------------------------------------

_EPS_PATTERN = re.compile(
    r"""(?:^|[_\-./])(?:eps|epsilon|e)[_\-]?(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)""",
    re.VERBOSE,
)
_TRAILING_DECIMAL = re.compile(r"(?:^|[_\-])(\d+\.\d+)(?=\.[a-zA-Z]+$)")


def _parse_filename_hints(onnx_rel: str, vnnlib_rel: str) -> Dict[str, Any]:
    hints: Dict[str, Any] = {
        "onnx_basename": Path(onnx_rel).name,
        "vnnlib_basename": Path(vnnlib_rel).name,
    }

    for name in (vnnlib_rel, onnx_rel):
        m = _EPS_PATTERN.search(name)
        if m:
            try:
                hints["epsilon"] = float(m.group(1))
                break
            except ValueError:
                pass

    if "epsilon" not in hints:
        for name in (vnnlib_rel, onnx_rel):
            m = _TRAILING_DECIMAL.search(Path(name).name)
            if m:
                try:
                    hints["epsilon"] = float(m.group(1))
                    break
                except ValueError:
                    pass

    idx = re.search(r"(?:idx|img|prop|spec)[_\-]?(\d+)", vnnlib_rel)
    if idx:
        try:
            hints["spec_index"] = int(idx.group(1))
        except ValueError:
            pass

    return hints


def _onnx_stem_for_label(onnx_rel: str) -> str:
    """
    Derive a short, filesystem-safe label from the ONNX filename for use as
    a construction suffix when --split-by-onnx is passed. Strips common
    VNN-COMP prefixes and extensions.
    """
    stem = Path(onnx_rel).name
    for suffix in (".onnx.gz", ".onnx"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    return stem or "net"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--benchmark-dir",
        required=True,
        help="Path to one VNN-COMP benchmark, e.g. .../benchmarks/cifar_biasfield",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Destination directory; will contain manifest.json and instances/",
    )
    ap.add_argument(
        "--id-prefix",
        default=None,
        help=(
            "Per-instance id prefix. Default: derived from benchmark name, "
            "e.g. 'vb_cifar_biasfield'."
        ),
    )
    ap.add_argument(
        "--construction-override",
        default=None,
        help="Force a specific construction label instead of vnncomp.<benchmark_name>.",
    )
    ap.add_argument(
        "--split-by-onnx",
        action="store_true",
        help=(
            "Use a separate construction label per underlying ONNX file, "
            "i.e. vnncomp.<benchmark>.<onnx_stem>. Useful when a benchmark "
            "bundles multiple networks."
        ),
    )
    ap.add_argument(
        "--onnx-opset",
        type=int,
        default=None,
        help=(
            "Optionally down-convert materialized ONNX models to this ai.onnx "
            "opset, e.g. 13 for nnenum or 18 for newer tools."
        ),
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only preprocess the first N instances, useful for fast iteration.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite out_dir/instances if they already exist.",
    )
    args = ap.parse_args()

    root = Path(args.benchmark_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    csv_path = _find_instances_csv(root)
    if csv_path is None:
        raise SystemExit(f"No instances.csv under {root}")

    rows = _read_instances_csv(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    if args.limit is not None:
        rows = rows[: args.limit]

    instances_root = out_dir / "instances"

    if instances_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"{instances_root} already exists; pass --overwrite to replace."
            )
        shutil.rmtree(instances_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    instances_root.mkdir(parents=True, exist_ok=True)

    benchmark_name = root.name
    id_prefix = args.id_prefix or f"vb_{benchmark_name}"
    base_construction = args.construction_override or f"vnncomp.{benchmark_name}"

    manifest_instances: List[Dict[str, Any]] = []
    missing: List[Tuple[str, str]] = []
    opset_conversions: List[Dict[str, Any]] = []

    for i, (onnx_rel, vnnlib_rel, timeout) in enumerate(rows):
        onnx_src = _resolve(root, onnx_rel)
        vnnlib_src = _resolve(root, vnnlib_rel)

        if onnx_src is None or vnnlib_src is None:
            missing.append((onnx_rel, vnnlib_rel))
            continue

        inst_id = f"{id_prefix}_{i:04d}"
        inst_dir = instances_root / inst_id
        inst_dir.mkdir(parents=True, exist_ok=True)

        onnx_dst = inst_dir / "model.onnx"
        vnnlib_dst = inst_dir / "spec.vnnlib"

        _write_decompressed(onnx_src, onnx_dst, overwrite=True)

        opset_info: Dict[str, Any] = {}
        if args.onnx_opset is not None:
            old_opset, new_opset, converted = _convert_onnx_opset_inplace(
                onnx_dst,
                args.onnx_opset,
            )
            opset_info = {
                "requested_onnx_opset": args.onnx_opset,
                "source_onnx_opset": old_opset,
                "materialized_onnx_opset": new_opset,
                "onnx_opset_converted": converted,
            }
            opset_conversions.append(
                {
                    "id": inst_id,
                    "source_onnx": onnx_rel,
                    **opset_info,
                }
            )
        else:
            try:
                model = onnx.load(str(onnx_dst))
                opset_info = {
                    "materialized_onnx_opset": _get_ai_onnx_opset(model),
                }
            except Exception:
                opset_info = {}

        _write_decompressed(vnnlib_src, vnnlib_dst, overwrite=True)

        hints = _parse_filename_hints(onnx_rel, vnnlib_rel)
        if opset_info:
            hints.update(opset_info)

        if args.construction_override is not None:
            construction = args.construction_override
        elif args.split_by_onnx:
            construction = f"{base_construction}.{_onnx_stem_for_label(onnx_rel)}"
        else:
            construction = base_construction

        meta_dst = inst_dir / "meta.json"

        entry = {
            "id": inst_id,
            "seed": None,
            "construction": construction,
            "paths": {
                "onnx": str(onnx_dst.relative_to(out_dir)),
                "vnnlib": str(vnnlib_dst.relative_to(out_dir)),
                "meta": str(meta_dst.relative_to(out_dir)),
            },
            "args": hints,
        }
        if timeout is not None:
            entry["vnncomp_timeout_s"] = timeout

        meta = {
            "id": inst_id,
            "construction": construction,
            "source_benchmark": benchmark_name,
            "source_onnx": onnx_rel,
            "source_vnnlib": vnnlib_rel,
            "args": hints,
        }
        if timeout is not None:
            meta["vnncomp_timeout_s"] = timeout

        meta_dst.write_text(json.dumps(meta, indent=2))
        manifest_instances.append(entry)

    manifest: Dict[str, Any] = {
        "benchmark_name": f"vnncomp22_{benchmark_name}",
        "source": str(root),
        "construction": base_construction,
        "n_instances": len(manifest_instances),
        "instances": manifest_instances,
    }

    if args.onnx_opset is not None:
        manifest["requested_onnx_opset"] = args.onnx_opset
        manifest["onnx_opset_conversions"] = opset_conversions

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {out_dir / 'manifest.json'}")
    print(f"  instances materialized : {len(manifest_instances)}")
    print(f"  missing files          : {len(missing)}")

    if args.onnx_opset is not None:
        n_converted = sum(1 for x in opset_conversions if x.get("onnx_opset_converted"))
        print(f"  requested ONNX opset   : {args.onnx_opset}")
        print(f"  ONNX files converted   : {n_converted}")

    if missing:
        for m in missing[:5]:
            print(f"    {m[0]} | {m[1]}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    eps_vals = sorted({
        inst["args"].get("epsilon") for inst in manifest_instances
        if inst["args"].get("epsilon") is not None
    })
    if eps_vals:
        print(f"  epsilons parsed from filenames: {eps_vals}")

    if args.split_by_onnx:
        constructions = sorted({inst["construction"] for inst in manifest_instances})
        print(f"  construction labels ({len(constructions)}):")
        for c in constructions:
            n = sum(1 for inst in manifest_instances if inst["construction"] == c)
            print(f"    {c:<60s} n={n}")


if __name__ == "__main__":
    main()