from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from VeriStressGT.utils.make_box_vnnlib import make_box_vnnlib

from .model_and_x0 import MLP, export_onnx, sample_x0
from .exact_radius import solve_from_onnx

CONSTRUCTION_NAME = "mlp_relu.milp.exact_radius"


def add_args(p: argparse.ArgumentParser) -> None:
    # Model architecture
    p.add_argument("--input-dim", type=int, default=50)
    p.add_argument("--h1", type=int, default=5)
    p.add_argument("--h2", type=int, default=10)
    p.add_argument("--num-outputs", type=int, default=5)

    p.add_argument("--epsilon-frac", type=float, default=0.999,
               help="Fraction of r* to use as epsilon (safe mode only). "
                    "E.g. 0.5, 0.9, 0.99, 0.999 to approach the boundary.")
    # MILP solve knobs
    p.add_argument("--rmax", type=float, default=2.0)
    p.add_argument("--time-limit", type=float, default=240.0)
    p.add_argument("--mip-gap", type=float, default=None)

    # How to set epsilon after finding r*
    p.add_argument("--epsilon-mode", choices=["exact", "safe", "unsafe"], default="safe")

    # Optional artifacts
    p.add_argument("--save-x0", action="store_true")
    p.add_argument("--save-json", action="store_true")


def run(args) -> Dict[str, Any]:
    """
    Must write args.onnx_path and args.vnnlib_path.
    The driver will also pass args.seed (in your mixed YAML driver).
    """
    seed = int(getattr(args, "seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    onnx_path = Path(args.onnx_path)
    vnnlib_path = Path(args.vnnlib_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    vnnlib_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Create & export ONNX
    model = MLP(d=int(args.input_dim), h1=int(args.h1), h2=int(args.h2), C=int(args.num_outputs))
    export_onnx(model, d=int(args.input_dim), path=str(onnx_path))

    # 2) Choose x0 (and optionally save)
    x0 = sample_x0(model, d=int(args.input_dim))
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)

    if args.save_x0:
        np.save(onnx_path.parent / "x0.npy", x0.astype(np.float32))

    # 3) Solve exact radius on the ONNX model
    res = solve_from_onnx(
        onnx_path=str(onnx_path),
        x0=x0,
        Rmax=float(args.rmax),
        time_limit_s=float(args.time_limit) if args.time_limit is not None else None,
        mip_gap=float(args.mip_gap) if args.mip_gap is not None else None,
        verbose=True,
    )

    r_star = float(res["r_star"])
    y = int(res["y"])
    if not np.isfinite(r_star):
        raise RuntimeError("MILP returned inf/NaN r_star. Try increasing --rmax or debug the solver.")

    if args.save_json:
        (onnx_path.parent / "radius_result.json").write_text(json.dumps(res, indent=2))

    # 4) Choose epsilon based on mode
    if args.epsilon_mode == "exact":
        eps = r_star
    elif args.epsilon_mode == "safe":
        eps = args.epsilon_frac * r_star
    else:  # unsafe
        eps = 1.01 * r_star

    # 5) Export VNNLIB using your standard helper
    make_box_vnnlib(
        center=x0.astype(np.float32),
        eps=float(eps),
        out=str(vnnlib_path),
        num_outputs=int(args.num_outputs),
        label=y,
    )

    print(f"Wrote ONNX:   {onnx_path}")
    print(f"Wrote VNNLIB: {vnnlib_path}")
    print(f"r*={r_star:.6f}  eps={eps:.6f}  mode={args.epsilon_mode}  y={y}")

    return {
        "onnx_path": str(onnx_path),
        "vnnlib_path": str(vnnlib_path),
        "seed": seed,
        "y": y,
        "r_star": r_star,
        "epsilon": float(eps),
        "epsilon_mode": str(args.epsilon_mode),
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description="MILP exact-radius constructor (standalone)")
    add_args(ap)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--onnx_path", required=True)
    ap.add_argument("--vnnlib_path", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    _main()
