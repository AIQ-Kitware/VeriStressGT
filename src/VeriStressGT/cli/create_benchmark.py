# src/VeriStressGT/cli/create_benchmark.py
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from VeriStressGT.registry.constructions import discover_constructions


def _git_commit() -> Optional[str]:
    import subprocess
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
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


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_spec(path: Path) -> Dict[str, Any]:
    """
    Load YAML or JSON benchmark spec.

    YAML requires PyYAML installed. If not installed, only JSON works.
    """
    txt = path.read_text()
    suf = path.suffix.lower()
    if suf in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML spec requested but PyYAML is not installed. "
                "Install with: pip install pyyaml  (or use a .json spec)"
            ) from e
        data = yaml.safe_load(txt)
        if not isinstance(data, dict):
            raise ValueError("Spec root must be a mapping/dict")
        return data
    elif suf == ".json":
        data = json.loads(txt)
        if not isinstance(data, dict):
            raise ValueError("Spec root must be a mapping/dict")
        return data
    else:
        raise ValueError(f"Unsupported spec extension: {suf} (use .yaml/.yml/.json)")


def _parse_constructor_args(cons, argv: List[str]) -> argparse.Namespace:
    """
    Parse constructor-specific CLI args for *single-construction mode*.
    IMPORTANT: we do NOT register --onnx_path/--vnnlib_path here to avoid conflicts.
    Driver will inject those attributes into the parsed Namespace later.
    """
    cap = argparse.ArgumentParser(add_help=False)
    cons.add_args(cap)
    cargs = cap.parse_args(argv)
    return cargs


def _resolve_seeds(args) -> List[int]:
    if args.seeds is not None and args.num is not None:
        raise ValueError("Use either --seeds or --num, not both.")
    if args.seeds is None and args.num is None:
        raise ValueError("Must specify either --seeds or --num.")
    if args.seeds is not None:
        return list(args.seeds)
    return list(range(args.seed0, args.seed0 + int(args.num)))


def _safe_rmtree(path: Path) -> None:
    import shutil
    if path.exists():
        shutil.rmtree(path)


def _write_manifest(out_dir: Path, manifest: Dict[str, Any]) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_instance_meta(meta_path: Path, inst_meta: Dict[str, Any]) -> None:
    _ensure_parent(meta_path)
    meta_path.write_text(json.dumps(inst_meta, indent=2))


def _mk_instance_paths(out_dir: Path, inst_dir: Path, onnx_name: str, vnnlib_name: str, meta_name: str):
    onnx_path = inst_dir / onnx_name
    vnnlib_path = inst_dir / vnnlib_name
    meta_path = inst_dir / meta_name
    return onnx_path, vnnlib_path, meta_path


def _namespace_from_dict(d: Dict[str, Any]) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, v)
    return ns


def _run_one_instance(
    *,
    cons,
    out_dir: Path,
    inst_dir: Path,
    inst_id: str,
    seed: Optional[int],
    run_args: argparse.Namespace,
    onnx_name: str,
    vnnlib_name: str,
    meta_name: str,
) -> Dict[str, Any]:
    onnx_path, vnnlib_path, meta_path = _mk_instance_paths(out_dir, inst_dir, onnx_name, vnnlib_name, meta_name)

    # Inject required driver-owned fields (constructors must NOT define/argparse these)
    setattr(run_args, "onnx_path", str(onnx_path))
    setattr(run_args, "vnnlib_path", str(vnnlib_path))
    if seed is not None:
        setattr(run_args, "seed", seed)

    # Run construction
    cons.run(run_args)

    # Sanity
    if not onnx_path.exists():
        raise RuntimeError(f"Constructor did not write ONNX: {onnx_path}")
    if not vnnlib_path.exists():
        raise RuntimeError(f"Constructor did not write VNNLIB: {vnnlib_path}")

    inst_meta: Dict[str, Any] = {
        "id": inst_id,
        "construction": cons.name,
        "seed": seed,
        "paths": {
            "onnx": str(onnx_path.relative_to(out_dir)),
            "vnnlib": str(vnnlib_path.relative_to(out_dir)),
            "meta": str(meta_path.relative_to(out_dir)),
        },
        "sha256": {
            "onnx": _sha256_file(onnx_path),
            "vnnlib": _sha256_file(vnnlib_path),
        },
        # For reproducibility: store the per-instance args we passed
        "args": vars(deepcopy(run_args)),
    }

    _write_instance_meta(meta_path, inst_meta)
    return inst_meta


def _apply_constructor_defaults(cons, run_args: argparse.Namespace) -> argparse.Namespace:
    """
    For spec-mode, fill missing attributes on run_args using the defaults declared
    in cons.add_args().
    """
    p = argparse.ArgumentParser(add_help=False)
    cons.add_args(p)

    # parse empty argv so argparse returns a Namespace populated with defaults
    defaults = p.parse_args([])

    # Any attribute not provided by spec gets filled from defaults
    for k, v in vars(defaults).items():
        if not hasattr(run_args, k):
            setattr(run_args, k, v)

    return run_args


def main(argv: Optional[List[str]] = None) -> None:
    constructions = discover_constructions()
    if not constructions:
        raise RuntimeError("No constructions discovered under VeriStressGT.robust_constructions")

    ap = argparse.ArgumentParser(
        prog="VeriStressGT-create",
        description="Create a benchmark folder by running robust constructions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Two modes:
    #  (A) legacy CLI mode: one --construction + repeated seeds
    #  (B) spec mode: --spec defines a list of instances (possibly mixed constructions + per-instance args)
    ap.add_argument("--spec", default=None, help="Benchmark spec (.yaml/.yml/.json). Enables mixed + per-instance args.")
    ap.add_argument("--construction", default=None, choices=sorted(constructions.keys()),
                    help="Construction name (CLI mode only; ignored if --spec is used).")

    ap.add_argument("--out_dir", required=True, help="Output benchmark directory (will be created).")
    ap.add_argument("--name", default=None, help="Benchmark name (defaults to folder name).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite out_dir if it exists.")

    # CLI-mode instance generation controls
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="Explicit list of seeds (CLI mode).")
    ap.add_argument("--num", type=int, default=None, help="Number of instances to generate (CLI mode).")
    ap.add_argument("--seed0", type=int, default=0, help="Starting seed if using --num (CLI mode).")

    # common output naming
    ap.add_argument("--onnx_name", default="model.onnx")
    ap.add_argument("--vnnlib_name", default="spec.vnnlib")
    ap.add_argument("--meta_name", default="meta.json")

    # Parse known args first (constructor args remain in remaining)
    args, remaining = ap.parse_known_args(argv)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} exists. Pass --overwrite to replace.")
        _safe_rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_name = args.name or out_dir.name
    instances_dir = out_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "name": bench_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
        "common": {
            "onnx_name": args.onnx_name,
            "vnnlib_name": args.vnnlib_name,
            "meta_name": args.meta_name,
        },
        "mode": "spec" if args.spec else "cli",
        "instances": [],
    }

    # -----------------------
    # Mode B: spec-driven
    # -----------------------
    if args.spec is not None:
        spec_path = Path(args.spec)
        spec = _load_spec(spec_path)

        # Optional: spec-level name overrides
        if manifest["name"] == out_dir.name and isinstance(spec.get("name"), str):
            manifest["name"] = spec["name"]

        defaults = spec.get("defaults", {})
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            raise ValueError("spec.defaults must be a dict if provided")

        inst_list = spec.get("instances", None)
        if not isinstance(inst_list, list) or len(inst_list) == 0:
            raise ValueError("spec.instances must be a non-empty list")

        manifest["spec_path"] = str(spec_path)
        manifest["spec"] = {"defaults": defaults}  # keep manifest smaller

        for idx, inst in enumerate(inst_list, start=1):
            if not isinstance(inst, dict):
                raise ValueError(f"Instance #{idx} must be a dict")

            inst_id = str(inst.get("id") or f"{idx:06d}")
            inst_dir = instances_dir / inst_id
            inst_dir.mkdir(parents=True, exist_ok=True)

            cname = inst.get("construction", None)
            if not isinstance(cname, str):
                raise ValueError(f"Instance {inst_id}: missing 'construction' string")
            if cname not in constructions:
                raise ValueError(f"Instance {inst_id}: unknown construction '{cname}'")

            cons = constructions[cname]

            seed = inst.get("seed", None)
            if seed is not None and not isinstance(seed, int):
                raise ValueError(f"Instance {inst_id}: seed must be int if provided")

            inst_args = inst.get("args", {})
            if inst_args is None:
                inst_args = {}
            if not isinstance(inst_args, dict):
                raise ValueError(f"Instance {inst_id}: args must be a dict if provided")

            # Merge: defaults -> instance.args
            merged: Dict[str, Any] = {}
            merged.update(deepcopy(defaults))
            merged.update(deepcopy(inst_args))

            run_args = _namespace_from_dict(merged)
            run_args = _apply_constructor_defaults(cons, run_args)


            inst_meta = _run_one_instance(
                cons=cons,
                out_dir=out_dir,
                inst_dir=inst_dir,
                inst_id=inst_id,
                seed=seed,
                run_args=run_args,
                onnx_name=args.onnx_name,
                vnnlib_name=args.vnnlib_name,
                meta_name=args.meta_name,
            )

            manifest["instances"].append(inst_meta)
            print(f"[{idx}/{len(inst_list)}] wrote {inst_dir}")

        _write_manifest(out_dir, manifest)
        print(f"\nDone. Manifest: {out_dir / 'manifest.json'}")
        return

    # -----------------------
    # Mode A: legacy CLI
    # -----------------------
    if args.construction is None:
        raise ValueError("In CLI mode (no --spec), you must pass --construction <name>")

    cons = constructions[args.construction]
    cargs = _parse_constructor_args(cons, remaining)

    seeds = _resolve_seeds(args)

    manifest.update(
        {
            "construction": cons.name,
            "constructor_args": vars(cargs),
            "seeds": seeds,
        }
    )

    for idx, seed in enumerate(seeds, start=1):
        inst_id = f"{idx:06d}"
        inst_dir = instances_dir / inst_id
        inst_dir.mkdir(parents=True, exist_ok=True)

        run_args = deepcopy(cargs)
        # Driver injects seed + required paths (constructors must not argparse these)
        inst_meta = _run_one_instance(
            cons=cons,
            out_dir=out_dir,
            inst_dir=inst_dir,
            inst_id=inst_id,
            seed=seed,
            run_args=run_args,
            onnx_name=args.onnx_name,
            vnnlib_name=args.vnnlib_name,
            meta_name=args.meta_name,
        )

        manifest["instances"].append(inst_meta)
        print(f"[{idx}/{len(seeds)}] wrote {inst_dir}")

    _write_manifest(out_dir, manifest)
    print(f"\nDone. Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
