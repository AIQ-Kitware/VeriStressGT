"""
Polynomial-network robust instance construction via algebraic decision-boundary sampling.

This constructor packages the polynomial-network idea for VeriStressGT:
  1. sample points p on the binary decision boundary g(p)=0 by random line slices;
  2. perturb p in the l_inf normal direction to get q;
  3. set eps = ||q - p||_inf - delta_prime;
  4. accept q only if a multi-start L-BFGS-B nearest-boundary search over
     B_eps(q) does not find |g(z)|^2 below the chosen tolerance;
  5. export the polynomial network to ONNX and the robustness query to VNN-LIB.

No verifier is called here. The benchmark runner handles verifier experiments.

Compared with the raw notebook version, the default parameter initialization is
scaled/normalized for numerical stability. There is intentionally no raw
"colab" initialization option.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn

from VeriStressGT.utils.make_box_vnnlib import make_box_vnnlib
from VeriStressGT.utils.onnx_export import ExportConfig, export_pytorch_to_onnx


CONSTRUCTION_NAME = "polynomial.algebraic_boundary"


def _require_scipy_optimize():
    """
    Import SciPy lazily so construction discovery still works without SciPy.

    This constructor intentionally uses brentq root finding and L-BFGS-B
    nearest-boundary optimization, so SciPy is required at run time.
    """
    try:
        from scipy.optimize import brentq, minimize  # type: ignore
    except Exception as e:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "polynomial.algebraic_boundary requires scipy for brentq and "
            "L-BFGS-B. Install it with `pip install scipy` or add "
            "`scipy>=1.10` to the generate extra in pyproject.toml."
        ) from e
    return brentq, minimize


# ---------------------------------------------------------------------------
# Configuration / records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolynomialBoundaryConfig:
    input_dim: int = 100
    hidden_dim: int = 100
    degree: int = 10
    num_outputs: int = 2

    # Boundary sampling: random affine lines a + t v.
    num_lines: int = 1000
    center_scale: float = 2.0
    line_t_min: float = -10.0
    line_t_max: float = 10.0
    line_grid_points: int = 200
    boundary_tol: float = 1e-6
    boundary_rel_tol: float = 1e-10
    unique_tol: float = 1e-4

    # Candidate generation.
    eps_normal: float = 2.0e-2
    delta_prime: float = 5.0e-3
    max_candidates: int = 5
    candidate_margin_tol: float = 1.0e-8
    grad_tol: float = 1.0e-12

    # Nearest-boundary check (NBC): minimize |g(z)|^2 over the verification box.
    nbc_num_restarts: int = 50
    nbc_max_iter: int = 500
    nbc_boundary_tol: float = 1.0e-10
    nbc_ftol: float = 1.0e-15
    nbc_gtol: float = 1.0e-10

    # Random model initialization.
    #   fanin: W1 ~ N(0, 1/input_dim), W2 ~ N(0, 1/hidden_dim), small biases.
    #   row_l2: sample standard normal rows, normalize each row to L2 norm 1,
    #           then apply the user scales; biases are small.
    # There is intentionally no raw unnormalized notebook/Colab mode.
    init_mode: str = "fanin"
    weight_scale: float = 1.0
    bias_scale: float = 1.0
    output_weight_scale: float = 1.0
    output_bias_scale: float = 1.0
    bias_base_scale: float = 0.1
    output_bias_base_scale: float = 0.1

    seed: int = 0
    opset: int = 13
    quiet: bool = False

    def validate(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if self.degree < 2:
            raise ValueError("degree must be >= 2")
        if self.num_outputs != 2:
            raise ValueError(
                "This algebraic-boundary construction is binary; num_outputs must be 2."
            )
        if self.num_lines < 1:
            raise ValueError("num_lines must be >= 1")
        if self.line_grid_points < 2:
            raise ValueError("line_grid_points must be >= 2")
        if not self.line_t_min < self.line_t_max:
            raise ValueError("line_t_min must be < line_t_max")
        if self.center_scale <= 0:
            raise ValueError("center_scale must be > 0")
        if self.boundary_tol <= 0:
            raise ValueError("boundary_tol must be > 0")
        if self.boundary_rel_tol <= 0:
            raise ValueError("boundary_rel_tol must be > 0")
        if self.unique_tol <= 0:
            raise ValueError("unique_tol must be > 0")
        if self.eps_normal <= 0:
            raise ValueError("eps_normal must be > 0")
        if self.delta_prime <= 0:
            raise ValueError("delta_prime must be > 0")
        if self.delta_prime >= self.eps_normal:
            raise ValueError("delta_prime must be smaller than eps_normal")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.candidate_margin_tol < 0:
            raise ValueError("candidate_margin_tol must be >= 0")
        if self.grad_tol <= 0:
            raise ValueError("grad_tol must be > 0")
        if self.nbc_num_restarts < 1:
            raise ValueError("nbc_num_restarts must be >= 1")
        if self.nbc_max_iter < 1:
            raise ValueError("nbc_max_iter must be >= 1")
        if self.nbc_boundary_tol <= 0:
            raise ValueError("nbc_boundary_tol must be > 0")
        if self.init_mode not in {"fanin", "row_l2"}:
            raise ValueError("init_mode must be one of: fanin, row_l2")
        for name, value in [
            ("weight_scale", self.weight_scale),
            ("bias_scale", self.bias_scale),
            ("output_weight_scale", self.output_weight_scale),
            ("output_bias_scale", self.output_bias_scale),
            ("bias_base_scale", self.bias_base_scale),
            ("output_bias_base_scale", self.output_bias_base_scale),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.opset < 11:
            raise ValueError("opset should be >= 11")


@dataclass
class PolynomialParams:
    W1: np.ndarray  # (hidden_dim, input_dim)
    b1: np.ndarray  # (hidden_dim,)
    W2: np.ndarray  # (2, hidden_dim)
    b2: np.ndarray  # (2,)
    degree: int


@dataclass
class Candidate:
    index: int
    boundary_point: np.ndarray
    x0: np.ndarray
    known_boundary_dist_linf: float
    epsilon: float
    label: int
    margin_at_x0: float
    nbc_min_fsq: float
    nbc_best_z: np.ndarray
    nbc_elapsed_sec: float
    nbc_num_restarts: int


# ---------------------------------------------------------------------------
# Polynomial network oracle and PyTorch module
# ---------------------------------------------------------------------------

class PolynomialOracle:
    """NumPy float64 oracle for logits, margin, and exact margin gradient."""

    def __init__(self, params: PolynomialParams):
        self.params = params

    def logits(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        h = (x @ self.params.W1.T + self.params.b1) ** int(self.params.degree)
        out = h @ self.params.W2.T + self.params.b2
        return out[0] if single else out

    def margin(self, x: np.ndarray) -> Union[float, np.ndarray]:
        out = self.logits(x)
        if out.ndim == 1:
            return float(out[0] - out[1])
        return out[:, 0] - out[:, 1]

    def relative_margin_residual(self, x: np.ndarray) -> float:
        out = self.logits(np.asarray(x, dtype=np.float64).reshape(-1))
        abs_res = abs(float(out[0] - out[1]))
        scale = float(np.max(np.abs(out))) + 1.0e-12
        return abs_res / scale

    def grad_margin(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        a = self.params.W1 @ x + self.params.b1
        alpha = self.params.W2[0] - self.params.W2[1]
        coeff = alpha * int(self.params.degree) * (a ** (int(self.params.degree) - 1))
        return self.params.W1.T @ coeff


class PolynomialNet(nn.Module):
    """PyTorch module exported to ONNX."""

    def __init__(self, params: PolynomialParams, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.degree = int(params.degree)
        self.register_buffer("W1", torch.tensor(params.W1, dtype=dtype))
        self.register_buffer("b1", torch.tensor(params.b1, dtype=dtype))
        self.register_buffer("W2", torch.tensor(params.W2, dtype=dtype))
        self.register_buffer("b2", torch.tensor(params.b2, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        a = x @ self.W1.t() + self.b1
        h = a.pow(self.degree)
        return h @ self.W2.t() + self.b2


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _row_l2_normalize(A: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / (norms + eps)


def make_random_params(
    cfg: PolynomialBoundaryConfig,
    rng: np.random.Generator,
) -> PolynomialParams:
    """
    Sample a fixed polynomial network with numerically stable scaling.

    This is still the same shallow polynomial network
        f(x) = W2 (W1 x + b1)^degree + b2,
    but the default scales keep degree-10 powers from exploding just because
    input_dim is large.
    """
    if cfg.init_mode == "fanin":
        W1 = rng.standard_normal((cfg.hidden_dim, cfg.input_dim)).astype(np.float64)
        W1 *= float(cfg.weight_scale) / math.sqrt(float(cfg.input_dim))

        b1 = rng.standard_normal(cfg.hidden_dim).astype(np.float64)
        b1 *= float(cfg.bias_scale) * float(cfg.bias_base_scale)

        W2 = rng.standard_normal((cfg.num_outputs, cfg.hidden_dim)).astype(np.float64)
        W2 *= float(cfg.output_weight_scale) / math.sqrt(float(cfg.hidden_dim))

        b2 = rng.standard_normal(cfg.num_outputs).astype(np.float64)
        b2 *= float(cfg.output_bias_scale) * float(cfg.output_bias_base_scale)

    elif cfg.init_mode == "row_l2":
        W1 = rng.standard_normal((cfg.hidden_dim, cfg.input_dim)).astype(np.float64)
        W1 = float(cfg.weight_scale) * _row_l2_normalize(W1)

        b1 = rng.standard_normal(cfg.hidden_dim).astype(np.float64)
        b1 *= float(cfg.bias_scale) * float(cfg.bias_base_scale)

        W2 = rng.standard_normal((cfg.num_outputs, cfg.hidden_dim)).astype(np.float64)
        W2 = float(cfg.output_weight_scale) * _row_l2_normalize(W2)

        b2 = rng.standard_normal(cfg.num_outputs).astype(np.float64)
        b2 *= float(cfg.output_bias_scale) * float(cfg.output_bias_base_scale)

    else:  # validate() should prevent this.
        raise ValueError(f"Unknown init_mode: {cfg.init_mode}")

    return PolynomialParams(W1=W1, b1=b1, W2=W2, b2=b2, degree=int(cfg.degree))


def find_brackets(
    f_line,
    *,
    t_min: float,
    t_max: float,
    n_points: int,
) -> List[Tuple[float, float]]:
    ts = np.linspace(float(t_min), float(t_max), int(n_points))
    vals = np.asarray([f_line(float(t)) for t in ts], dtype=np.float64)

    brackets: List[Tuple[float, float]] = []
    for i in range(len(ts) - 1):
        vi = vals[i]
        vj = vals[i + 1]
        if not (np.isfinite(vi) and np.isfinite(vj)):
            continue
        if vi == 0.0:
            lo = ts[max(i - 1, 0)]
            hi = ts[min(i + 1, len(ts) - 1)]
            if lo < hi:
                brackets.append((float(lo), float(hi)))
        elif np.sign(vi) != np.sign(vj):
            brackets.append((float(ts[i]), float(ts[i + 1])))
    return brackets


def deduplicate_points(points: Sequence[np.ndarray], tol: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 0), dtype=np.float64)

    unique: List[np.ndarray] = []
    for p in points:
        p = np.asarray(p, dtype=np.float64)
        if all(np.linalg.norm(p - q) > float(tol) for q in unique):
            unique.append(p)

    return np.stack(unique, axis=0) if unique else np.empty((0, points[0].size), dtype=np.float64)


def sample_boundary_points(
    oracle: PolynomialOracle,
    cfg: PolynomialBoundaryConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    brentq, _ = _require_scipy_optimize()

    points: List[np.ndarray] = []
    for _ in range(int(cfg.num_lines)):
        base = rng.uniform(-float(cfg.center_scale), float(cfg.center_scale), size=int(cfg.input_dim))
        direction = rng.standard_normal(int(cfg.input_dim))
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            continue
        direction = direction / norm

        def f_line(t: float) -> float:
            return float(oracle.margin(base + float(t) * direction))

        for lo, hi in find_brackets(
            f_line,
            t_min=float(cfg.line_t_min),
            t_max=float(cfg.line_t_max),
            n_points=int(cfg.line_grid_points),
        ):
            try:
                t_star = float(brentq(f_line, lo, hi, xtol=1.0e-10, maxiter=200))
            except Exception:
                continue

            p = base + t_star * direction
            abs_res = abs(float(oracle.margin(p)))
            rel_res = float(oracle.relative_margin_residual(p))
            if abs_res <= float(cfg.boundary_tol) or rel_res <= float(cfg.boundary_rel_tol):
                points.append(p.astype(np.float64, copy=False))

    unique = deduplicate_points(points, tol=float(cfg.unique_tol))
    if unique.size == 0:
        raise RuntimeError(
            "No boundary points found. Try increasing num_lines, center_scale, "
            "or the line search interval."
        )
    return unique


def linf_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def perturb_normal_linf(
    p: np.ndarray,
    oracle: PolynomialOracle,
    eps_linf: float,
    grad_tol: float,
) -> Optional[np.ndarray]:
    grad = oracle.grad_margin(p)
    if float(np.linalg.norm(grad)) <= float(grad_tol):
        return None
    return np.asarray(p, dtype=np.float64) + float(eps_linf) * np.sign(grad)


def nearest_boundary_check(
    oracle: PolynomialOracle,
    q: np.ndarray,
    eps_verify: float,
    cfg: PolynomialBoundaryConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Numerical nearest-boundary check.

    It minimizes |g(z)|^2 over the l_inf box B_eps(q) using multi-start L-BFGS-B.
    If the best restart finds |g(z)|^2 below nbc_boundary_tol, the candidate is
    discarded.
    """
    _, minimize = _require_scipy_optimize()

    q = np.asarray(q, dtype=np.float64).reshape(-1)
    lb = q - float(eps_verify)
    ub = q + float(eps_verify)
    bounds = list(zip(lb.tolist(), ub.tolist()))

    def obj_and_grad(z: np.ndarray) -> Tuple[float, np.ndarray]:
        z = np.asarray(z, dtype=np.float64)
        gz = float(oracle.margin(z))
        return gz * gz, 2.0 * gz * oracle.grad_margin(z)

    best_fsq = math.inf
    best_z = q.copy()

    for _ in range(int(cfg.nbc_num_restarts)):
        z0 = rng.uniform(lb, ub)
        res = minimize(
            obj_and_grad,
            z0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": int(cfg.nbc_max_iter),
                "ftol": float(cfg.nbc_ftol),
                "gtol": float(cfg.nbc_gtol),
            },
        )
        fun = float(res.fun)
        if fun < best_fsq:
            best_fsq = fun
            best_z = np.asarray(res.x, dtype=np.float64).copy()

    return {
        "passes": bool(best_fsq >= float(cfg.nbc_boundary_tol)),
        "min_fsq": float(best_fsq),
        "best_z": best_z,
        "n_restarts": int(cfg.nbc_num_restarts),
    }


def select_candidate(
    oracle: PolynomialOracle,
    boundary_points: np.ndarray,
    cfg: PolynomialBoundaryConfig,
    rng: np.random.Generator,
    *,
    verbose: bool,
) -> Candidate:
    checked = 0

    for p in boundary_points:
        q = perturb_normal_linf(
            p,
            oracle=oracle,
            eps_linf=float(cfg.eps_normal),
            grad_tol=float(cfg.grad_tol),
        )
        if q is None:
            continue

        margin_q = float(oracle.margin(q))
        if abs(margin_q) <= float(cfg.candidate_margin_tol):
            continue

        known_dist = linf_dist(q, p)
        eps_verify = known_dist - float(cfg.delta_prime)
        if eps_verify <= 0:
            continue

        checked += 1
        t0 = time.time()
        nbc = nearest_boundary_check(
            oracle=oracle,
            q=q,
            eps_verify=eps_verify,
            cfg=cfg,
            rng=rng,
        )
        elapsed = time.time() - t0

        if verbose:
            status = "PASS" if nbc["passes"] else "FAIL"
            print(
                f"Candidate {checked}: eps={eps_verify:.6e}, "
                f"|g(q)|={abs(margin_q):.3e}, "
                f"min|g(z)|^2={nbc['min_fsq']:.3e} -> {status} "
                f"({elapsed:.1f}s)"
            )

        if nbc["passes"]:
            logits = oracle.logits(q)
            label = int(np.argmax(logits))
            return Candidate(
                index=checked - 1,
                boundary_point=np.asarray(p, dtype=np.float64).copy(),
                x0=np.asarray(q, dtype=np.float64).copy(),
                known_boundary_dist_linf=float(known_dist),
                epsilon=float(eps_verify),
                label=label,
                margin_at_x0=margin_q,
                nbc_min_fsq=float(nbc["min_fsq"]),
                nbc_best_z=np.asarray(nbc["best_z"], dtype=np.float64).copy(),
                nbc_elapsed_sec=float(elapsed),
                nbc_num_restarts=int(nbc["n_restarts"]),
            )

        if checked >= int(cfg.max_candidates):
            break

    raise RuntimeError(
        f"No candidate passed the nearest-boundary check after checking {checked} "
        f"candidate(s). Try increasing max_candidates/num_lines or loosening "
        f"nbc_boundary_tol."
    )


def _boundary_residual_summary(
    oracle: PolynomialOracle,
    boundary_points: np.ndarray,
    max_points: int = 10,
) -> Dict[str, List[float]]:
    pts = boundary_points[: min(int(max_points), len(boundary_points))]
    abs_vals = [abs(float(oracle.margin(p))) for p in pts]
    rel_vals = [float(oracle.relative_margin_residual(p)) for p in pts]
    return {"abs": abs_vals, "rel": rel_vals}


def create_instance(
    cfg: PolynomialBoundaryConfig,
    *,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], PolynomialNet]:
    cfg.validate()

    rng = np.random.default_rng(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    params = make_random_params(cfg, rng)
    oracle = PolynomialOracle(params)

    if verbose:
        print(
            f"Polynomial boundary construction: n={cfg.input_dim}, "
            f"h={cfg.hidden_dim}, degree={cfg.degree}, seed={cfg.seed}, "
            f"init={cfg.init_mode}"
        )
        print("Sampling algebraic decision-boundary points...")

    t0 = time.time()
    boundary_points = sample_boundary_points(oracle, cfg, rng)
    sampling_elapsed = time.time() - t0
    residual_summary = _boundary_residual_summary(oracle, boundary_points)

    if verbose:
        print(
            f"Found {len(boundary_points)} unique boundary point(s) "
            f"in {sampling_elapsed:.1f}s."
        )
        print(
            "Boundary abs residuals, first few: "
            f"{np.asarray(residual_summary['abs'], dtype=np.float64)}"
        )
        print(
            "Boundary rel residuals, first few: "
            f"{np.asarray(residual_summary['rel'], dtype=np.float64)}"
        )

    if verbose:
        print("Perturbing along l_inf normal direction and running NBC...")

    candidate = select_candidate(
        oracle=oracle,
        boundary_points=boundary_points,
        cfg=cfg,
        rng=rng,
        verbose=verbose,
    )

    # Export verifier-facing model as float32.
    model = PolynomialNet(params, dtype=torch.float32).eval()

    candidate_boundary_abs_res = abs(float(oracle.margin(candidate.boundary_point)))
    candidate_boundary_rel_res = float(oracle.relative_margin_residual(candidate.boundary_point))

    meta: Dict[str, Any] = {
        "label": int(candidate.label),
        "epsilon": float(candidate.epsilon),
        "x0": candidate.x0.astype(float).tolist(),
        "is_robust": True,
        "certificate_type": "numerical_multistart_lbfgsb_nearest_boundary_check",
        "construction_note": (
            "Accepted because the multi-start L-BFGS-B nearest-boundary check did "
            "not find |g(z)|^2 below nbc_boundary_tol inside the verification box. "
            "No verifier was run by this constructor."
        ),
        "candidate": {
            "candidate_index_checked": int(candidate.index),
            "known_boundary_dist_linf": float(candidate.known_boundary_dist_linf),
            "delta_prime": float(cfg.delta_prime),
            "margin_at_x0": float(candidate.margin_at_x0),
            "boundary_abs_residual_at_p": float(candidate_boundary_abs_res),
            "boundary_rel_residual_at_p": float(candidate_boundary_rel_res),
            "nbc_min_fsq": float(candidate.nbc_min_fsq),
            "nbc_elapsed_sec": float(candidate.nbc_elapsed_sec),
            "nbc_num_restarts": int(candidate.nbc_num_restarts),
            "nbc_boundary_tol": float(cfg.nbc_boundary_tol),
        },
        "model": {
            "input_dim": int(cfg.input_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "degree": int(cfg.degree),
            "num_outputs": int(cfg.num_outputs),
            "init_mode": str(cfg.init_mode),
            "weight_scale": float(cfg.weight_scale),
            "bias_scale": float(cfg.bias_scale),
            "output_weight_scale": float(cfg.output_weight_scale),
            "output_bias_scale": float(cfg.output_bias_scale),
            "bias_base_scale": float(cfg.bias_base_scale),
            "output_bias_base_scale": float(cfg.output_bias_base_scale),
        },
        "sampling": {
            "num_boundary_points": int(len(boundary_points)),
            "sampling_elapsed_sec": float(sampling_elapsed),
            "num_lines": int(cfg.num_lines),
            "center_scale": float(cfg.center_scale),
            "line_t_min": float(cfg.line_t_min),
            "line_t_max": float(cfg.line_t_max),
            "line_grid_points": int(cfg.line_grid_points),
            "boundary_tol": float(cfg.boundary_tol),
            "boundary_rel_tol": float(cfg.boundary_rel_tol),
            "residual_summary_first_points": residual_summary,
        },
    }

    return meta, model


def create_and_export_instance(
    cfg: PolynomialBoundaryConfig,
    onnx_path: Union[str, Path],
    vnnlib_path: Union[str, Path],
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    onnx_path = Path(onnx_path)
    vnnlib_path = Path(vnnlib_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    vnnlib_path.parent.mkdir(parents=True, exist_ok=True)

    meta, model = create_instance(cfg, verbose=verbose)

    example_input = np.asarray(meta["x0"], dtype=np.float32).reshape(1, int(cfg.input_dim))
    export_pytorch_to_onnx(
        model,
        str(onnx_path),
        (1, int(cfg.input_dim)),
        config=ExportConfig(
            opset=int(cfg.opset),
            do_constant_folding=False,
            dynamic_batch=False,
            use_legacy_exporter=True,
        ),
        example_input=example_input,
    )

    make_box_vnnlib(
        center=np.asarray(meta["x0"], dtype=np.float64),
        eps=float(meta["epsilon"]),
        out=str(vnnlib_path),
        num_outputs=int(cfg.num_outputs),
        label=int(meta["label"]),
    )

    # Write a .pnn.json sidecar so verify_vnnlib.py (AlgebraicVerification) can
    # reconstruct the network without parsing the ONNX graph. The sidecar maps
    # PolynomialNet buffers to the PolynomialNeuralNetwork state_dict layout.
    def _arr(t) -> Dict[str, Any]:
        a = t.detach().cpu().numpy().astype(np.float64)
        return {"shape": list(a.shape), "data": a.flatten().tolist()}

    sidecar = {
        "arch": {
            "input_dim": int(cfg.input_dim),
            "output_dim": int(cfg.num_outputs),
            "hidden_dims": [int(cfg.hidden_dim)],
            "act_degree": int(cfg.degree),
            "homogeneous": True,
            "bias": True,
            "trainable": True,
            "s": 1.0,
        },
        "state_dict": {
            "layers.0.weight": _arr(model.W1),
            "layers.0.bias": _arr(model.b1),
            "layers.1.weight": _arr(model.W2),
            "layers.1.bias": _arr(model.b2),
            "activations.0.coeffs": {"shape": [1], "data": [1.0]},
        },
    }
    sidecar_path = onnx_path.with_suffix(onnx_path.suffix + ".pnn.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    result: Dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "vnnlib_path": str(vnnlib_path),
        "sidecar_path": str(sidecar_path),
        "epsilon": float(meta["epsilon"]),
        "label": int(meta["label"]),
        "x0": meta["x0"],
        "is_robust": True,
        "meta": meta,
        "cfg": {
            "input_dim": int(cfg.input_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "degree": int(cfg.degree),
            "num_outputs": int(cfg.num_outputs),
            "num_lines": int(cfg.num_lines),
            "center_scale": float(cfg.center_scale),
            "line_t_min": float(cfg.line_t_min),
            "line_t_max": float(cfg.line_t_max),
            "line_grid_points": int(cfg.line_grid_points),
            "boundary_tol": float(cfg.boundary_tol),
            "boundary_rel_tol": float(cfg.boundary_rel_tol),
            "unique_tol": float(cfg.unique_tol),
            "eps_normal": float(cfg.eps_normal),
            "delta_prime": float(cfg.delta_prime),
            "max_candidates": int(cfg.max_candidates),
            "candidate_margin_tol": float(cfg.candidate_margin_tol),
            "grad_tol": float(cfg.grad_tol),
            "nbc_num_restarts": int(cfg.nbc_num_restarts),
            "nbc_max_iter": int(cfg.nbc_max_iter),
            "nbc_boundary_tol": float(cfg.nbc_boundary_tol),
            "nbc_ftol": float(cfg.nbc_ftol),
            "nbc_gtol": float(cfg.nbc_gtol),
            "init_mode": str(cfg.init_mode),
            "weight_scale": float(cfg.weight_scale),
            "bias_scale": float(cfg.bias_scale),
            "output_weight_scale": float(cfg.output_weight_scale),
            "output_bias_scale": float(cfg.output_bias_scale),
            "bias_base_scale": float(cfg.bias_base_scale),
            "output_bias_base_scale": float(cfg.output_bias_base_scale),
            "seed": int(cfg.seed),
            "opset": int(cfg.opset),
        },
    }
    return result


# ---------------------------------------------------------------------------
# Constructor API
# ---------------------------------------------------------------------------

def add_args(p: argparse.ArgumentParser) -> None:
    # Network.
    p.add_argument("--input-dim", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=100)
    p.add_argument("--degree", type=int, default=10)
    p.add_argument("--num-outputs", type=int, default=2)

    # Boundary sampling.
    p.add_argument("--num-lines", type=int, default=1000)
    p.add_argument("--center-scale", type=float, default=2.0)
    p.add_argument("--line-t-min", type=float, default=-10.0)
    p.add_argument("--line-t-max", type=float, default=10.0)
    p.add_argument("--line-grid-points", type=int, default=200)
    p.add_argument("--boundary-tol", type=float, default=1e-6)
    p.add_argument("--boundary-rel-tol", type=float, default=1e-10)
    p.add_argument("--unique-tol", type=float, default=1e-4)

    # Candidate generation / robustness radius.
    p.add_argument("--eps-normal", type=float, default=2e-2)
    p.add_argument("--delta-prime", type=float, default=5e-3)
    p.add_argument("--max-candidates", type=int, default=5)
    p.add_argument("--candidate-margin-tol", type=float, default=1e-8)
    p.add_argument("--grad-tol", type=float, default=1e-12)

    # Numerical nearest-boundary check.
    p.add_argument("--nbc-num-restarts", type=int, default=50)
    p.add_argument("--nbc-max-iter", type=int, default=500)
    p.add_argument("--nbc-boundary-tol", type=float, default=1e-10)
    p.add_argument("--nbc-ftol", type=float, default=1e-15)
    p.add_argument("--nbc-gtol", type=float, default=1e-10)

    # Initialization / fixed parameter scales.
    p.add_argument(
        "--init-mode",
        choices=["fanin", "row_l2"],
        default="fanin",
        help=(
            "Fixed random initialization scaling. 'fanin' uses 1/sqrt(fan-in) "
            "weight scaling; 'row_l2' L2-normalizes each sampled row. There is "
            "intentionally no raw unnormalized Colab/notebook mode."
        ),
    )
    p.add_argument("--weight-scale", type=float, default=1.0)
    p.add_argument("--bias-scale", type=float, default=1.0)
    p.add_argument("--output-weight-scale", type=float, default=1.0)
    p.add_argument("--output-bias-scale", type=float, default=1.0)
    p.add_argument("--bias-base-scale", type=float, default=0.1)
    p.add_argument("--output-bias-base-scale", type=float, default=0.1)

    # Export / misc.
    p.add_argument("--opset", type=int, default=13)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")


def run(args) -> Dict[str, Any]:
    cfg = PolynomialBoundaryConfig(
        input_dim=int(args.input_dim),
        hidden_dim=int(args.hidden_dim),
        degree=int(args.degree),
        num_outputs=int(args.num_outputs),
        num_lines=int(args.num_lines),
        center_scale=float(args.center_scale),
        line_t_min=float(args.line_t_min),
        line_t_max=float(args.line_t_max),
        line_grid_points=int(args.line_grid_points),
        boundary_tol=float(args.boundary_tol),
        boundary_rel_tol=float(args.boundary_rel_tol),
        unique_tol=float(args.unique_tol),
        eps_normal=float(args.eps_normal),
        delta_prime=float(args.delta_prime),
        max_candidates=int(args.max_candidates),
        candidate_margin_tol=float(args.candidate_margin_tol),
        grad_tol=float(args.grad_tol),
        nbc_num_restarts=int(args.nbc_num_restarts),
        nbc_max_iter=int(args.nbc_max_iter),
        nbc_boundary_tol=float(args.nbc_boundary_tol),
        nbc_ftol=float(args.nbc_ftol),
        nbc_gtol=float(args.nbc_gtol),
        init_mode=str(args.init_mode),
        weight_scale=float(args.weight_scale),
        bias_scale=float(args.bias_scale),
        output_weight_scale=float(args.output_weight_scale),
        output_bias_scale=float(args.output_bias_scale),
        bias_base_scale=float(args.bias_base_scale),
        output_bias_base_scale=float(args.output_bias_base_scale),
        seed=int(args.seed),
        opset=int(args.opset),
        quiet=bool(args.quiet),
    )

    result = create_and_export_instance(
        cfg=cfg,
        onnx_path=args.onnx_path,
        vnnlib_path=args.vnnlib_path,
        verbose=not bool(args.quiet),
    )

    if not bool(args.quiet):
        print(f"Wrote ONNX:   {result['onnx_path']}")
        print(f"Wrote VNNLIB: {result['vnnlib_path']}")
        print(
            f"Selected label={result['label']}, "
            f"epsilon={result['epsilon']:.6e}, "
            f"certificate={result['meta']['certificate_type']}"
        )

    return result


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a polynomial algebraic-boundary robustness instance."
    )
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--vnnlib_path", required=True)
    add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
