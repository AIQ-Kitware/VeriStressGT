#!/usr/bin/env python3
"""
Exact robustness radius computation via Mixed-Integer Linear Programming.

Given a ReLU MLP and input x0, computes the exact minimum L_inf perturbation
that causes misclassification.
"""
import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import onnx
import gurobipy as gp
from gurobipy import GRB

# ----------------------------
# Utilities: parse ONNX tensors
# ----------------------------
def _get_initializer_map(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    init = {}
    for t in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(t).astype(np.float64)
        init[t.name] = arr
    return init


def _get_attr(node: onnx.NodeProto, name: str, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


@dataclass
class LinearLayer:
    W: np.ndarray  # shape (out, in)
    b: np.ndarray  # shape (out,)


def parse_mlp_gemm_relu(model: onnx.ModelProto) -> List[Any]:
    """
    Parse a simple chain MLP of (Linear, Relu, Linear, Relu, ..., Linear).
    Returns a list like: [LinearLayer, "relu", LinearLayer, "relu", ..., LinearLayer].
    Supports Linear encoded as Gemm OR (MatMul + Add).
    """
    init = _get_initializer_map(model)
    nodes = list(model.graph.node)
    layers: List[Any] = []

    i = 0
    while i < len(nodes):
        n = nodes[i]
        op = n.op_type

        if op == "Relu":
            layers.append("relu")
            i += 1
            continue

        if op == "Gemm":
            transB = int(_get_attr(n, "transB", 0))
            alpha = float(_get_attr(n, "alpha", 1.0))
            beta = float(_get_attr(n, "beta", 1.0))

            B_name = n.input[1]
            C_name = n.input[2] if len(n.input) >= 3 else None

            B = init[B_name].astype(np.float64)

            if transB == 1:
                W = B.copy()
            else:
                W = B.T

            if C_name is None or C_name == "":
                b = np.zeros((W.shape[0],), dtype=np.float64)
            else:
                C = init[C_name].reshape(-1).astype(np.float64)
                b = C.copy()

            W = (alpha * W).astype(np.float64)
            b = (beta * b).astype(np.float64)

            layers.append(LinearLayer(W=W, b=b))
            i += 1
            continue

        if op == "MatMul":
            if i + 1 >= len(nodes) or nodes[i + 1].op_type != "Add":
                raise ValueError("Found MatMul not followed by Add; unsupported for this parser.")
            mm = nodes[i]
            add = nodes[i + 1]

            B_name = mm.input[1]
            B = init[B_name].astype(np.float64)

            if B.ndim != 2:
                raise ValueError("MatMul weight is not 2D.")
            W = B.T

            bias_name = add.input[1] if add.input[0] == mm.output[0] else add.input[0]
            b = init[bias_name].reshape(-1).astype(np.float64)

            layers.append(LinearLayer(W=W, b=b))
            i += 2
            continue

        if op in ("Flatten", "Reshape", "Identity", "Dropout"):
            i += 1
            continue

        raise ValueError(f"Unsupported op_type in chain parser: {op}")

    if not layers or not isinstance(layers[0], LinearLayer) or not isinstance(layers[-1], LinearLayer):
        raise ValueError("Parsed layers do not look like an MLP chain of Linear/Relu/Linear.")

    return layers


def count_hidden_affines(layers: List[Any]) -> int:
    """Count number of hidden affine layers (those followed by ReLU)."""
    count = 0
    for i, layer in enumerate(layers):
        if isinstance(layer, LinearLayer) and i + 1 < len(layers) and layers[i + 1] == "relu":
            count += 1
    return count


# ----------------------------
# Forward + IBP bounds
# ----------------------------
def forward_layers(layers: List[Any], x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64)
    for layer in layers:
        if layer == "relu":
            z = np.maximum(z, 0.0)
        else:
            assert isinstance(layer, LinearLayer)
            z = layer.W @ z + layer.b
    return z


def interval_affine(W: np.ndarray, b: np.ndarray, l_in: np.ndarray, u_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interval arithmetic for affine layer: W @ [l_in, u_in] + b."""
    Wpos = np.maximum(W, 0.0)
    Wneg = np.minimum(W, 0.0)
    l = Wpos @ l_in + Wneg @ u_in + b
    u = Wpos @ u_in + Wneg @ l_in + b
    return l, u


def interval_relu(l: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interval arithmetic for ReLU."""
    return np.maximum(l, 0.0), np.maximum(u, 0.0)


def ibp_preact_bounds(layers: List[Any], xL: np.ndarray, xU: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute IBP bounds for each preactivation of hidden layers.
    Returns list of (l_s, u_s) for each hidden affine in order.
    """
    bounds = []
    lz, uz = xL.astype(np.float64), xU.astype(np.float64)

    for layer in layers:
        if layer == "relu":
            lz, uz = interval_relu(lz, uz)
            continue
        assert isinstance(layer, LinearLayer)
        ls, us = interval_affine(layer.W, layer.b, lz, uz)
        bounds.append((ls, us))
        lz, uz = ls, us

    if len(bounds) < 1:
        raise ValueError("No affine layers found.")

    # Exclude final layer bounds (only need hidden layers)
    return bounds[:-1]


# ----------------------------
# MILP builder
# ----------------------------
def build_distance_milp(
    layers: List[Any],
    x0: np.ndarray,
    y: int,
    k: int,
    Rmax: float,
    time_limit_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
) -> Tuple[gp.Model, gp.MVar, gp.Var, gp.MVar]:
    """
    Build MILP minimizing t s.t. exists delta with ||delta||_inf <= t and f_k(x0+delta) >= f_y(x0+delta).

    The search is bounded by t <= Rmax to ensure IBP bounds remain valid.

    Returns (model, delta_var, t_var, logits_var).
    """
    x0 = x0.reshape(-1).astype(np.float64)
    d = x0.shape[0]

    # Compute IBP bounds on [x0 - Rmax, x0 + Rmax]
    xL = x0 - Rmax
    xU = x0 + Rmax
    hidden_preact_bounds = ibp_preact_bounds(layers, xL, xU)

    # Sanity check: number of IBP bounds should match number of hidden affines
    num_hidden = count_hidden_affines(layers)
    if len(hidden_preact_bounds) != num_hidden:
        raise ValueError(
            f"IBP bounds mismatch: got {len(hidden_preact_bounds)} bounds "
            f"but {num_hidden} hidden affine layers"
        )

    m = gp.Model()
    m.Params.OutputFlag = 0
    if time_limit_s is not None:
        m.Params.TimeLimit = float(time_limit_s)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    # Variables for delta and t (L_inf radius)
    delta = m.addMVar(shape=d, lb=-GRB.INFINITY, name="delta")

    # CRITICAL: Bound t by Rmax to ensure IBP bounds remain valid
    t = m.addVar(lb=0.0, ub=Rmax, name="t")

    # L_inf constraint: -t <= delta <= t
    m.addConstr(delta <= t)
    m.addConstr(delta >= -t)

    # Input x = x0 + delta
    x = m.addMVar(shape=d, lb=-GRB.INFINITY, name="x")
    m.addConstr(x == x0 + delta)

    # Build network layer by layer
    z_expr = x
    hidden_affine_idx = 0
    logits = None

    li = 0
    while li < len(layers):
        layer = layers[li]
        if layer == "relu":
            raise ValueError("Unexpected 'relu' at start of block.")
        assert isinstance(layer, LinearLayer)

        out_dim = layer.b.shape[0]
        s = m.addMVar(shape=out_dim, lb=-GRB.INFINITY, name=f"s{li}")
        m.addConstr(s == layer.W @ z_expr + layer.b)

        # Check if this is a hidden layer (followed by ReLU)
        if li + 1 < len(layers) and layers[li + 1] == "relu":
            ls, us = hidden_preact_bounds[hidden_affine_idx]
            hidden_affine_idx += 1

            # Case 1: ReLU always inactive (u <= 0) -> z = 0
            if np.all(us <= 0):
                z = m.addMVar(shape=out_dim, lb=0.0, ub=0.0, name=f"z{li}")
                z_expr = z
                li += 2
                continue

            # Case 2: ReLU always active (l >= 0) -> z = s
            if np.all(ls >= 0):
                z_expr = s
                li += 2
                continue

            # Case 3: Uncertain sign -> need binary variables
            z = m.addMVar(shape=out_dim, lb=0.0, name=f"z{li}")
            a = m.addMVar(shape=out_dim, vtype=GRB.BINARY, name=f"a{li}")

            # z >= s  (together with z >= 0 gives z >= max(0,s))
            m.addConstr(z >= s)

            # Big-M constraints
            m.addConstr(z <= us * a)
            m.addConstr(z <= s - ls * (1 - a))

            z_expr = z
            li += 2

        else:
            # Final layer (no ReLU)
            logits = s
            z_expr = s
            li += 1

    if logits is None:
        raise ValueError("No final affine logits were built.")

    # Misclassification constraint: f_k(x) >= f_y(x)
    m.addConstr(logits[k] >= logits[y])

    # Objective: minimize perturbation radius
    m.setObjective(t, GRB.MINIMIZE)

    return m, delta, t, logits


def solve_exact_radius(
    layers: List[Any],
    x0: np.ndarray,
    Rmax: float,
    time_limit_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute exact r* = min_{k!=y} t_k* by solving MILP per class.

    Args:
        layers: Parsed network layers
        x0: Input point
        Rmax: Maximum search radius (IBP bounds computed on this box)
        time_limit_s: Per-class MILP time limit
        mip_gap: Gurobi MIP gap tolerance
        verbose: Print progress

    Returns:
        Dictionary with r_star, k_star, delta, and per-class results.
        status is one of: OPTIMAL, TIME_LIMIT, INCOMPLETE, or None.
        INCOMPLETE means at least one timed-out class has a lower bound
        below the current best r*, so the true r* may be smaller.
    """
    x0 = x0.reshape(-1).astype(np.float64)
    logits0 = forward_layers(layers, x0)
    y = int(np.argmax(logits0))
    C = int(logits0.shape[0])

    if verbose:
        print(f"{'='*60}")
        print(f"Solving exact robustness radius")
        print(f"  Input dim:    {x0.shape[0]}")
        print(f"  Num classes:  {C}")
        print(f"  True class:   {y}")
        print(f"  Rmax:         {Rmax}")
        print(f"  Time limit:   {time_limit_s}s per class" if time_limit_s else "  Time limit:   None")
        print(f"  MIP gap:      {mip_gap}" if mip_gap else "  MIP gap:      default")
        print(f"{'='*60}")
        print(f"\nSolving {C-1} MILPs (one per target class k ≠ {y})...\n")

    best = {"r_star": float("inf"), "k_star": None, "delta": None, "status": None}
    total_start = time.time()
    class_results = []

    for idx, k in enumerate([kk for kk in range(C) if kk != y]):
        if verbose:
            print(f"[{idx+1}/{C-1}] Solving for target class k={k}...", end=" ", flush=True)

        class_start = time.time()

        m, delta, t, _ = build_distance_milp(
            layers=layers,
            x0=x0,
            y=y,
            k=k,
            Rmax=Rmax,
            time_limit_s=time_limit_s,
            mip_gap=mip_gap,
        )
        m.optimize()

        class_elapsed = time.time() - class_start

        status_str = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
        }.get(m.Status, f"STATUS_{m.Status}")

        result = {
            "k": k,
            "status": status_str,
            "time": class_elapsed,
            "t_k": None,
            "gap": None,
            "lower_bound": None,
        }

        if m.Status == GRB.OPTIMAL:
            tk = float(t.X)
            result["t_k"] = tk
            result["gap"] = m.MIPGap
            result["lower_bound"] = tk  # optimal => lower bound = objective

            if verbose:
                print(f"t*={tk:.6f}  time={class_elapsed:.1f}s  gap={m.MIPGap:.2e}", end="")

            # Check if solution hit Rmax boundary
            if np.isclose(tk, Rmax, rtol=1e-5):
                if verbose:
                    print(f"  [WARNING: t*=Rmax, consider increasing --rmax]", end="")

            if tk < best["r_star"]:
                best["r_star"] = tk
                best["k_star"] = k
                best["delta"] = delta.X.tolist()
                best["status"] = "OPTIMAL"
                if verbose:
                    print(f"  ← new best!", end="")

            if verbose:
                print()

        elif m.Status == GRB.TIME_LIMIT:
            lower_bound = None
            try:
                lower_bound = float(m.ObjBound)
            except Exception:
                pass
            result["lower_bound"] = lower_bound

            if m.SolCount > 0:
                tk = float(t.X)
                result["t_k"] = tk
                result["gap"] = m.MIPGap

                lb_str = f"  lb={lower_bound:.6f}" if lower_bound is not None else ""
                if verbose:
                    print(f"t={tk:.6f} (incumbent)  time={class_elapsed:.1f}s  gap={m.MIPGap:.2e}{lb_str}", end="")

                if tk < best["r_star"]:
                    best["r_star"] = tk
                    best["k_star"] = k
                    best["delta"] = delta.X.tolist()
                    best["status"] = "TIME_LIMIT"
                    if verbose:
                        print(f"  ← new best (not proven optimal)", end="")

                if verbose:
                    print()
            else:
                lb_str = f"  lb={lower_bound:.6f}" if lower_bound is not None else ""
                if verbose:
                    print(f"TIME_LIMIT (no solution)  time={class_elapsed:.1f}s{lb_str}")
        else:
            if verbose:
                print(f"{status_str}  time={class_elapsed:.1f}s")

        class_results.append(result)

    total_elapsed = time.time() - total_start

    # ── Check completeness: are all timed-out lower bounds above r*? ──
    incomplete_classes = []
    if best["r_star"] < float("inf"):
        for cr in class_results:
            if cr["status"] == "TIME_LIMIT":
                lb = cr.get("lower_bound")
                if lb is None or lb < best["r_star"]:
                    incomplete_classes.append({
                        "k": cr["k"],
                        "lower_bound": lb,
                        "incumbent": cr.get("t_k"),
                    })

    if incomplete_classes:
        best["status"] = "INCOMPLETE"
        best["incomplete_classes"] = incomplete_classes

    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"  Total time:   {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f"  True class:   y = {y}")
        print(f"  Best radius:  r* = {best['r_star']:.6f}")
        print(f"  Target class: k* = {best['k_star']}")
        print(f"  Status:       {best['status']}")

        # Warn if best radius equals Rmax
        if best['r_star'] < float('inf') and np.isclose(best['r_star'], Rmax, rtol=1e-5):
            print(f"  [WARNING: r* = Rmax, true radius may be larger. Increase --rmax]")

        if incomplete_classes:
            print(f"\n  ⚠ INCOMPLETE: {len(incomplete_classes)} class(es) have lower bounds below r*={best['r_star']:.6f}")
            for ic in incomplete_classes:
                lb_str = f"{ic['lower_bound']:.6f}" if ic['lower_bound'] is not None else "unknown"
                inc_str = f"{ic['incumbent']:.6f}" if ic['incumbent'] is not None else "no solution"
                print(f"    k={ic['k']}  lower_bound={lb_str}  incumbent={inc_str}")
            print(f"    True r* may be smaller. Increase --time-limit for these.")

        print(f"{'='*60}")

        # Print summary table
        print(f"\nPer-class results:")
        print(f"  {'k':<4} {'status':<12} {'t_k':<12} {'lower_bd':<12} {'time (s)':<10} {'gap':<10}")
        print(f"  {'-'*4} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        for r in class_results:
            tk_str = f"{r['t_k']:.6f}" if r['t_k'] is not None else "N/A"
            lb_str = f"{r['lower_bound']:.6f}" if r['lower_bound'] is not None else "N/A"
            gap_str = f"{r['gap']:.2e}" if r['gap'] is not None else "N/A"
            marker = " ←" if r['k'] == best['k_star'] else ""
            inc = " ⚠" if any(ic['k'] == r['k'] for ic in incomplete_classes) else ""
            print(f"  {r['k']:<4} {r['status']:<12} {tk_str:<12} {lb_str:<12} {r['time']:<10.1f} {gap_str:<10}{marker}{inc}")

    best["y"] = y
    best["logits_x0"] = logits0.tolist()
    best["class_results"] = class_results
    best["total_time"] = total_elapsed
    best["Rmax"] = Rmax

    return best


# ----------------------------
# VNNLIB Export
# ----------------------------
def export_vnnlib(
    x0: np.ndarray,
    epsilon: float,
    y: int,
    num_outputs: int,
    out_path: str,
):
    """
    Export VNNLIB file for robustness verification.

    The property encodes: exists x in [x0-eps, x0+eps] s.t. some Y_k >= Y_y.
    SAT = adversarial exists (unsafe), UNSAT = robust (safe).
    """
    x0 = x0.reshape(-1)
    d = x0.shape[0]

    with open(out_path, "w") as f:
        # Declare input variables
        for i in range(d):
            f.write(f"(declare-const X_{i} Real)\n")

        # Declare output variables
        for i in range(num_outputs):
            f.write(f"(declare-const Y_{i} Real)\n")

        # Input bounds: x0 - eps <= x <= x0 + eps
        for i in range(d):
            lb = float(x0[i] - epsilon)
            ub = float(x0[i] + epsilon)
            f.write(f"(assert (>= X_{i} {lb}))\n")
            f.write(f"(assert (<= X_{i} {ub}))\n")

        # Output property: adversarial exists if any Y_k >= Y_y for k != y
        f.write("(assert (or\n")
        for k in range(num_outputs):
            if k != y:
                f.write(f"  (and (>= Y_{k} Y_{y}))\n")
        f.write("))\n")

    print(f"Wrote VNNLIB: {out_path}")


def export_instance(
    onnx_path: str,
    x0: np.ndarray,
    epsilon: float,
    y: int,
    num_outputs: int,
    output_dir: str,
    instance_name: str,
):
    """Export ONNX + VNNLIB pair for verification benchmarking."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy ONNX
    onnx_out = output_dir / f"{instance_name}.onnx"
    shutil.copy(onnx_path, onnx_out)

    # Export VNNLIB
    vnnlib_out = output_dir / f"{instance_name}.vnnlib"
    export_vnnlib(x0, epsilon, y, num_outputs, str(vnnlib_out))

    # Save x0
    np.save(output_dir / f"{instance_name}_x0.npy", x0)

    print(f"Exported instance to {output_dir}/")
    print(f"  ONNX:   {onnx_out}")
    print(f"  VNNLIB: {vnnlib_out}")
    print(f"  x0:     {instance_name}_x0.npy")

    return {"onnx": str(onnx_out), "vnnlib": str(vnnlib_out)}


def solve_from_onnx(
    onnx_path: str,
    x0: np.ndarray,
    Rmax: float,
    time_limit_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    model = onnx.load(onnx_path)
    layers = parse_mlp_gemm_relu(model)
    return solve_exact_radius(
        layers=layers,
        x0=x0,
        Rmax=Rmax,
        time_limit_s=time_limit_s,
        mip_gap=mip_gap,
        verbose=verbose,
    )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute exact L_inf robustness radius via MILP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--onnx", required=True, help="Path to ONNX model.")
    ap.add_argument("--x0-npy", required=True, help="Path to numpy .npy file containing x0.")
    ap.add_argument("--rmax", type=float, default=2.0,
                    help="Maximum search radius. IBP bounds computed on [x0-rmax, x0+rmax].")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="Per-class MILP time limit (seconds).")
    ap.add_argument("--mip-gap", type=float, default=None,
                    help="Gurobi MIP gap tolerance.")
    ap.add_argument("--out", default="radius_result.json",
                    help="Output JSON path.")

    # VNNLIB export options
    ap.add_argument("--export-vnnlib", action="store_true",
                    help="Export VNNLIB after computing r*.")
    ap.add_argument("--export-dir", default="./milp_instance",
                    help="Directory for exported instance.")
    ap.add_argument("--instance-name", default="exact_radius_instance",
                    help="Instance name.")
    ap.add_argument("--epsilon-mode", choices=["exact", "safe", "unsafe"], default="safe",
                    help="How to set epsilon: exact=r*, safe=0.999*r*, unsafe=1.001*r*")
    ap.add_argument("--epsilon-override", type=float, default=None,
                    help="Override epsilon (ignore epsilon-mode).")

    args = ap.parse_args()
    # Load model and parse
    model = onnx.load(args.onnx)
    layers = parse_mlp_gemm_relu(model)

    # Print network structure
    print(f"Parsed network:")
    total_neurons = 0
    for i, layer in enumerate(layers):
        if isinstance(layer, LinearLayer):
            print(f"  Layer {i}: Linear {layer.W.shape[1]} -> {layer.W.shape[0]}")
            if i + 1 < len(layers) and layers[i + 1] == "relu":
                total_neurons += layer.W.shape[0]
        else:
            print(f"  Layer {i}: ReLU")
    print(f"  Total hidden neurons: {total_neurons}\n")

    # Load input
    x0 = np.load(args.x0_npy).astype(np.float64).reshape(-1)

    # Solve for exact radius
    res = solve_exact_radius(
        layers=layers,
        x0=x0,
        Rmax=float(args.rmax),
        time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
    )

    # Save results
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote: {args.out}")

    # Export VNNLIB if requested
    if args.export_vnnlib and res["r_star"] < float("inf"):
        r_star = res["r_star"]
        y = res["y"]
        num_outputs = len(res["logits_x0"])

        if args.epsilon_override is not None:
            epsilon = args.epsilon_override
        elif args.epsilon_mode == "exact":
            epsilon = r_star
        elif args.epsilon_mode == "safe":
            epsilon = r_star * 0.999
        elif args.epsilon_mode == "unsafe":
            epsilon = r_star * 1.001

        export_instance(
            onnx_path=args.onnx,
            x0=x0,
            epsilon=epsilon,
            y=y,
            num_outputs=num_outputs,
            output_dir=args.export_dir,
            instance_name=args.instance_name,
        )

        print(f"\nVerification instance:")
        print(f"  r* (exact) = {r_star:.6f}")
        print(f"  epsilon    = {epsilon:.6f}")
        print(f"  mode       = {args.epsilon_mode}")
        if args.epsilon_mode == "safe":
            print(f"  Expected result: UNSAT (robust)")
        elif args.epsilon_mode == "unsafe":
            print(f"  Expected result: SAT (adversarial exists)")
        else:
            print(f"  Expected result: boundary case")

        if res.get("status") == "INCOMPLETE":
            print(f"\n  ⚠ WARNING: r* is INCOMPLETE — epsilon may be based on a wrong r*.")
            print(f"    Ground-truth label for this instance is NOT reliable.")

    elif args.export_vnnlib and res["r_star"] == float("inf"):
        print("\nSkipping VNNLIB export: no finite radius found (all classes INFEASIBLE within Rmax)")


if __name__ == "__main__":
    main()