from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from VeriStressGT.utils.onnx_export import ExportConfig, export_pytorch_to_onnx
from VeriStressGT.utils.make_box_vnnlib import make_box_vnnlib

CONSTRUCTION_NAME = "attention.linear_dominance"

# Model

class LinearDominanceAttention(nn.Module):
    """
    All Reshape ops go (1, k) -> (1, a, b), keeping batch=1 as dim 0.
    This matches the pattern in fixed_pattern.py that abcrown handles.
    K is stored pre-transposed so view(1, d_k, n) gives K^T without
    emitting a Transpose node.
    """

    def __init__(
        self,
        n: int,
        d_k: int,
        d_v: int,
        W_Q_bd:     np.ndarray,   
        W_K_bd_perm: np.ndarray,  # (n*d, n*d_k) cols permuted so view(1,d_k,n) = K^T
        W_V_bd:     np.ndarray,   # (n*d, n*d_v)
        W_head:     np.ndarray,   # (n*d_v, C)
        b_head:     np.ndarray,   # (C,)
    ) -> None:
        super().__init__()
        self.n   = n
        self.d_k = d_k
        self.d_v = d_v

        def _linear(W: np.ndarray, bias: np.ndarray | None = None) -> nn.Linear:
            in_f, out_f = W.shape
            layer = nn.Linear(in_f, out_f, bias=(bias is not None))
            layer.weight = nn.Parameter(
                torch.from_numpy(W.T.astype(np.float32)), requires_grad=False
            )
            if bias is not None:
                layer.bias = nn.Parameter(
                    torch.from_numpy(bias.astype(np.float32)), requires_grad=False
                )
            return layer

        self.fc_Q   = _linear(W_Q_bd)
        self.fc_K   = _linear(W_K_bd_perm)
        self.fc_V   = _linear(W_V_bd)
        self.fc_out = _linear(W_head, b_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, d_k, d_v = self.n, self.d_k, self.d_v

        Q_flat = F.relu(self.fc_Q(x))          # (1, n*d_k)
        K_flat = F.relu(self.fc_K(x))          # (1, n*d_k)  pre-transposed layout
        V_flat =        self.fc_V(x)           # (1, n*d_v)

        # Batch dim stays 0 throughout -- same pattern as fixed_pattern.py
        Q_3d   = Q_flat.view(1, n, d_k)       
        K_3d_T = K_flat.view(1, d_k, n)
        S      = torch.bmm(Q_3d, K_3d_T)       # (1, n, n)   S[0,i,j] = <phi(q_i), phi(k_j)>

        V_3d   = V_flat.view(1, n, d_v)        # (1, n, d_v)
        Z      = torch.bmm(S, V_3d)  
        z_flat = Z.view(1, n * d_v)            # (1, n*d_v)

        return self.fc_out(z_flat)             # (1, C)



# Weight construction

def _build_block_query_key(n: int, d: int, d_k: int) -> np.ndarray:

    assert d_k <= d
    W = np.zeros((n * d, n * d_k), dtype=np.float64)
    for i in range(n):
        W[i*d : i*d + d_k, i*d_k : (i+1)*d_k] = np.eye(d_k)
    return W


def _permute_k_for_transpose(W_K_bd: np.ndarray, n: int, d_k: int) -> np.ndarray:
    """
    Permute columns of W_K_bd so that:
      ReLU(x @ W_K_bd_perm).view(1, d_k, n)[0, g, i] == K[i, g]
    """
    W_perm = np.zeros_like(W_K_bd)
    for i in range(n):
        for g in range(d_k):
            W_perm[:, g * n + i] = W_K_bd[:, i * d_k + g]
    return W_perm


def _build_block_value(n: int, d: int, d_v: int, rng: np.random.RandomState):
    """
    W_V_bd  (n*d, n*d_v)  block-diagonal semi-orthogonal per-token projection.
    Returns (W_V_bd, sv_V) where sv_V = max per-block spectral norm = 1.
    """
    W_V_bd = np.zeros((n * d, n * d_v), dtype=np.float64)
    sv_V = 0.0
    for i in range(n):
        W_raw = rng.randn(d, d_v)
        U, _, Vt = np.linalg.svd(W_raw, full_matrices=False)
        W_block = U[:, :d_v] @ Vt[:d_v, :]
        W_V_bd[i*d : (i+1)*d, i*d_v : (i+1)*d_v] = W_block
        sv_V = max(sv_V, float(np.linalg.norm(W_block, ord=2)))
    return W_V_bd, sv_V


def _build_instance(
    n: int,
    d: int,
    d_k: int,
    d_v: int,
    num_classes: int,
    gate_scale: float,
    noise_scale: float,
    epsilon: float,
    margin_factor: float,
    seed: int,
) -> Dict[str, Any]:
    if d_k != n:
        raise ValueError(f"Gate structure requires d_k == n, got d_k={d_k}, n={n}.")
    if d < n:
        raise ValueError(f"d={d} must be >= n={n}.")
    if gate_scale <= epsilon:
        raise ValueError(f"gate_scale={gate_scale} must strictly exceed epsilon={epsilon}.")
    if noise_scale <= epsilon:
        raise ValueError(f"noise_scale={noise_scale} must strictly exceed epsilon={epsilon}.")
    if margin_factor <= 1.0:
        raise ValueError(f"margin_factor={margin_factor} must exceed 1.0.")

    rng = np.random.RandomState(seed)

    # X_0
    X0 = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        X0[i, i] = gate_scale
        for j in range(n):
            if j != i:
                X0[i, j] = -noise_scale
    if d > n:
        X0[:, n:] = rng.randn(n, d - n) * 0.1

    # Projection matrices 
    W_Q_bd    = _build_block_query_key(n, d, d_k)
    W_K_bd    = _build_block_query_key(n, d, d_k)
    W_K_bd_perm = _permute_k_for_transpose(W_K_bd, n, d_k)
    W_V_bd, sv_V = _build_block_value(n, d, d_v, rng)

    # Nominal forward pass (numpy) 
    x0_flat = X0.reshape(1, -1)                         # (1, n*d)

    Q0_flat = np.maximum(0.0, x0_flat @ W_Q_bd)         # (1, n*d_k)
    K0_flat = np.maximum(0.0, x0_flat @ W_K_bd_perm)    # (1, n*d_k) permuted
    V0_flat = x0_flat @ W_V_bd                           # (1, n*d_v)

    Q0_3d   = Q0_flat.reshape(1, n, d_k)               # (1, n, d_k)
    K0_3d_T = K0_flat.reshape(1, d_k, n)               # (1, d_k, n) = K^T
    S0      = Q0_3d @ K0_3d_T                           # (1, n, n)
    V0_3d   = V0_flat.reshape(1, n, d_v)                # (1, n, d_v)
    Z0      = S0 @ V0_3d                                # (1, n, d_v)
    z0_flat = Z0.reshape(1, n * d_v)                    # (1, n*d_v)

    # Verify gate structure: S0[0, i, j] should be 0 for i != j
    for i in range(n):
        for j in range(n):
            val = float(S0[0, i, j])
            if i == j:
                assert val > 0, f"Diagonal gate S[{i},{i}] = {val} not positive"
                assert abs(val - gate_scale**2) < 1e-9, (
                    f"S[{i},{i}] = {val} != gate_scale^2 = {gate_scale**2}"
                )
            else:
                assert abs(val) < 1e-12, f"Off-diagonal S[{i},{j}] = {val} != 0"

    # Certificate bound (product rule on Z[i] = w_{ii} * V_i) 
    dw   = epsilon * 2.0 * (gate_scale + epsilon)   # |w_{ii}(x) - w_{ii}(x_0)|
    dV   = epsilon * math.sqrt(d) * sv_V             # ||V_i(x) - V_i(x_0)||_2

    B_max = 0.0
    for i in range(n):
        Vi0_norm = float(np.linalg.norm(V0_flat[0, i*d_v:(i+1)*d_v]))
        wi0      = gate_scale ** 2
        Bi       = dw * (Vi0_norm + dV) + wi0 * dV
        B_max    = max(B_max, Bi)

    # W_head: unit spectral norm 
    W_head_raw = rng.randn(n * d_v, num_classes)
    U2, _, Vt2 = np.linalg.svd(W_head_raw, full_matrices=False)
    W_head = U2[:, :num_classes] @ Vt2[:num_classes, :]
    L_h = float(np.linalg.norm(W_head, ord=2))
    assert abs(L_h - 1.0) < 1e-9, f"||W_head||_op = {L_h} != 1"

    # Certificate RHS and target margin 
    cert_rhs      = 2.0 * L_h * math.sqrt(n) * B_max
    target_margin = margin_factor * cert_rhs

    # b_head: analytical solution for desired margin 
    logits_raw = (z0_flat @ W_head)[0]     # (num_classes,)
    b_head = -logits_raw.copy()
    b_head[0] += target_margin
    # => f(x_0)[0] = target_margin, f(x_0)[k!=0] = 0

    # Verify 
    logits0 = logits_raw + b_head
    m_X0 = float(min(logits0[0] - logits0[k] for k in range(1, num_classes)))
    assert abs(m_X0 - target_margin) < 1e-6, f"Margin mismatch: {m_X0} vs {target_margin}"
    assert m_X0 > cert_rhs, (
        f"Certificate FAILS: m(X_0)={m_X0:.6f} <= cert_rhs={cert_rhs:.6f}"
    )
    assert int(np.argmax(logits0)) == 0, "Class 0 is not argmax at x_0"

    return dict(
        x0_flat=x0_flat.astype(np.float32),
        W_Q_bd=W_Q_bd.astype(np.float32),
        W_K_bd_perm=W_K_bd_perm.astype(np.float32),
        W_V_bd=W_V_bd.astype(np.float32),
        W_head=W_head.astype(np.float32),
        b_head=b_head.astype(np.float32),
        label=0,
        sv_V=float(sv_V),
        L_h=float(L_h),
        dw=float(dw),
        dV=float(dV),
        B_max=float(B_max),
        cert_rhs=float(cert_rhs),
        nominal_margin=float(m_X0),
    )

# Constructor interface
def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n",           type=int,   default=4,
        help="Sequence length. d_k = n is fixed by the gate structure.")
    p.add_argument("--d",           type=int,   default=4,
        help="Token dimension. Must be >= n.")
    p.add_argument("--d_v",         type=int,   default=4,
        help="Value projection dimension.")
    p.add_argument("--num_classes", type=int,   default=5,
        help="Number of output classes.")
    p.add_argument("--gate_scale",  type=float, default=0.5,
        help="Diagonal value in X_0's gate block. Must strictly exceed --epsilon.")
    p.add_argument("--noise_scale", type=float, default=0.2,
        help="Off-diagonal gate value magnitude. Must strictly exceed --epsilon.")
    p.add_argument("--epsilon",     type=float, default=0.05,
        help="Perturbation radius (entrywise l_inf on flat input).")
    p.add_argument("--margin_factor", type=float, default=2.0,
        help="m(X_0) = margin_factor * cert_rhs. Must exceed 1.0.")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    n           = int(args.n)
    d           = int(args.d)
    d_k         = n
    d_v         = int(args.d_v)
    num_classes = int(args.num_classes)
    gate_scale  = float(args.gate_scale)
    noise_scale = float(args.noise_scale)
    epsilon     = float(args.epsilon)
    margin_factor = float(args.margin_factor)
    seed        = int(args.seed)

    onnx_path   = Path(args.onnx_path)
    vnnlib_path = Path(args.vnnlib_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    vnnlib_path.parent.mkdir(parents=True, exist_ok=True)

    w = _build_instance(
        n=n, d=d, d_k=d_k, d_v=d_v, num_classes=num_classes,
        gate_scale=gate_scale, noise_scale=noise_scale,
        epsilon=epsilon, margin_factor=margin_factor, seed=seed,
    )

    model = LinearDominanceAttention(
        n=n, d_k=d_k, d_v=d_v,
        W_Q_bd=w["W_Q_bd"],
        W_K_bd_perm=w["W_K_bd_perm"],
        W_V_bd=w["W_V_bd"],
        W_head=w["W_head"],
        b_head=w["b_head"],
    ).eval()

    # Sanity-check forward pass
    x0_t = torch.from_numpy(w["x0_flat"])
    with torch.no_grad():
        logits = model(x0_t).numpy()[0]
    assert int(np.argmax(logits)) == w["label"], (
        f"Model argmax {np.argmax(logits)} != expected label {w['label']}"
    )

    cfg = ExportConfig(opset=13, do_constant_folding=True, dynamic_batch=False)
    export_pytorch_to_onnx(model, str(onnx_path), (1, n * d), config=cfg)

    make_box_vnnlib(
        center=w["x0_flat"].reshape(-1),
        eps=epsilon,
        out=str(vnnlib_path),
        num_outputs=num_classes,
        label=w["label"],
    )

    meta = {
        "construction":     CONSTRUCTION_NAME,
        "seed":             seed,
        "n":                n, "d": d, "d_k": d_k, "d_v": d_v,
        "num_classes":      num_classes,
        "gate_scale":       gate_scale,
        "noise_scale":      noise_scale,
        "epsilon":          epsilon,
        "margin_factor":    margin_factor,
        "label":            w["label"],
        "is_robust":        True,
        "ground_truth":     "UNSAT",
        "sv_V":             w["sv_V"],
        "L_h":              w["L_h"],
        "dw":               w["dw"],
        "dV":               w["dV"],
        "B_max":            w["B_max"],
        "cert_rhs":         w["cert_rhs"],
        "nominal_margin":   w["nominal_margin"],
        "margin_over_cert": w["nominal_margin"] / w["cert_rhs"],
    }

    print(
        f"[{CONSTRUCTION_NAME}] seed={seed} n={n} d={d} d_k={d_k} d_v={d_v} "
        f"eps={epsilon:.4f}  "
        f"margin={w['nominal_margin']:.6f}  cert_rhs={w['cert_rhs']:.6f}  "
        f"({w['nominal_margin']/w['cert_rhs']:.2f}x headroom)"
    )
    return meta

# Standalone entry point

def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Linear-attention dominance construction (standalone)"
    )
    ap.add_argument("--onnx_path",   required=True)
    ap.add_argument("--vnnlib_path", required=True)
    ap.add_argument("--seed",        type=int, required=True)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    _main()
    