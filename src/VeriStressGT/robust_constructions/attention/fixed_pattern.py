from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from VeriStressGT.utils.make_box_vnnlib import make_box_vnnlib
from VeriStressGT.utils.onnx_export import export_pytorch_to_onnx

CONSTRUCTION_NAME = "attention.fixed_pattern"

class FixedPatternAttentionNet(nn.Module):

    def __init__(
        self,
        n: int, d: int, d_v: int, alpha: float,
        W_V:    torch.Tensor,   # (d, d_v)
        W_head: torch.Tensor,   # (ℓ, n*d_v)
        b_head: torch.Tensor,   # (ℓ,)
    ) -> None:
        super().__init__()
        self.n, self.d, self.d_v = n, d, d_v
        self.register_buffer("alpha",  torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("W_V",    W_V)
        self.register_buffer("W_head", W_head)
        self.register_buffer("b_head", b_head)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        B  = x_flat.shape[0]
        X  = x_flat.view(B, self.n, self.d)
        S  = self.alpha * torch.bmm(X, X.transpose(1, 2))   # (B,n,n)
        A  = torch.softmax(S, dim=-1)                        # (B,n,n)
        V  = torch.matmul(X, self.W_V)                       # (B,n,d_v)
        Z  = torch.bmm(A, V)                                 # (B,n,d_v)
        z  = Z.view(B, self.n * self.d_v)
        return z @ self.W_head.t() + self.b_head

def _spectral_norm(W: np.ndarray) -> float:
    return float(np.linalg.norm(W, ord=2))


def _row_l2_max(Z: np.ndarray) -> float:
    """‖Z‖_{2,∞} = max_i ‖Z[i,:]‖_2"""
    return float(np.max(np.linalg.norm(Z, axis=1)))


def _spectrally_normalised(W: np.ndarray) -> np.ndarray:
    s = _spectral_norm(W)
    return W / s if s > 0.0 else W


def compute_L_attn(
    n: int, d: int,
    alpha: float, epsilon: float,
    V0: np.ndarray,      # (n, d_v)
    W_V: np.ndarray,     # (d, d_v)
) -> tuple[float, float]:

    B_S    = alpha * (2.0 * math.sqrt(d) + epsilon * d)
    sigma  = _spectral_norm(W_V)
    V0_inf = _row_l2_max(V0)
    L_attn = (
        (n / 4.0) * B_S * V0_inf
        + math.sqrt(d) * sigma
        + (n / 4.0) * B_S * epsilon * math.sqrt(d) * sigma
    )
    return B_S, L_attn


def check_certificate(
    X0: np.ndarray,       # (n, d)
    alpha: float, epsilon: float,
    W_V: np.ndarray, W_head: np.ndarray, b_head: np.ndarray,
    verbose: bool = True,
) -> dict:
    n, d = X0.shape

    # ---- (i) Gap condition -----------------------------------------------
    G  = X0 @ X0.T
    np.fill_diagonal(G, 0.0)
    mu       = float(np.max(np.abs(G)))
    lhs_gap  = 1.0 - mu
    rhs_gap  = 4.0 * epsilon * math.sqrt(d) + 2.0 * epsilon**2 * d
    gap_ok   = lhs_gap > rhs_gap

    delta_min = alpha * lhs_gap
    C_max     = alpha * (4.0 * math.sqrt(d) + 2.0 * epsilon * d)
    gap_ok_actual = delta_min > epsilon * C_max

    # ---- (ii) Margin condition -------------------------------------------
    X0t, W_Vt = torch.from_numpy(X0), torch.from_numpy(W_V)
    with torch.no_grad():
        A0    = torch.softmax(alpha * (X0t @ X0t.t()), dim=-1)
        V0    = (A0 @ (X0t @ W_Vt)).numpy()
    # Note: V0 here is Z0 = A0·X0·W_V, not X0·W_V.
    # For L_attn we need V0 = X0·W_V (the value matrix before mixing):
    V0_pre = (X0 @ W_V)                            # (n, d_v)

    B_S, L_attn = compute_L_attn(n, d, alpha, epsilon, V0_pre, W_V)

    L_h    = _spectral_norm(W_head)
    z0     = V0.flatten()
    logits = W_head @ z0 + b_head
    y      = int(np.argmax(logits))
    m_X0   = float(logits[y] - np.delete(logits, y).max())

    rhs_margin = 2.0 * L_h * math.sqrt(n) * L_attn * epsilon
    margin_ok  = m_X0 > rhs_margin

    valid = gap_ok and margin_ok

    result = dict(
        mu=mu, delta_min=delta_min, C_max=C_max,
        lhs_gap=lhs_gap, rhs_gap=rhs_gap,
        gap_ok_simplified=gap_ok, gap_ok_actual=gap_ok_actual,
        B_S=B_S, V0_2inf=_row_l2_max(V0_pre),
        sigma_WV=_spectral_norm(W_V),
        L_attn=L_attn, L_h=L_h,
        logits=logits, label=y, margin=m_X0,
        rhs_margin=rhs_margin, margin_ok=margin_ok,
        valid=valid,
    )
    if verbose:
        _print_cert(result, n, d, epsilon, alpha)
    return result


def _print_cert(r: dict, n, d, eps, alpha) -> None:
    ok = lambda b: "PASS" if b else "FAIL"
    print("=" * 62)
    print("  (i)  Gap condition:")
    print(f"{ok(r['gap_ok_actual'])}")
    print("  (ii) Margin condition:")
    print(f"{ok(r['margin_ok'])}")
    print(f"  Label (y): {r['label']}    Certificate VALID: {r['valid']}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _build_near_orthogonal_tokens(
    n: int, d: int, rng: np.random.RandomState
) -> tuple[np.ndarray, float]:
    """Sample n unit vectors in R^d, minimise coherence over 2000 tries."""
    best_X0, best_mu = None, float("inf")
    for _ in range(2000):
        X0 = rng.randn(n, d).astype(np.float32)
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        G  = X0 @ X0.T
        np.fill_diagonal(G, 0.0)
        mu = float(np.max(np.abs(G)))
        if mu < best_mu:
            best_mu, best_X0 = mu, X0.copy()
    return best_X0, best_mu


def _build_head(
    Z0_flat: np.ndarray,    # (n*d_v,)
    num_classes: int,
    required_margin: float,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build W_head (spectrally normalised, so L_h=1) and b_head such that
    class 0 wins at Z0_flat by exactly required_margin.

    Strategy:
      raw_k = (W_head @ Z0_flat)_k
      b_0 is set so that  (raw_0 + b_0) - (raw_k + b_k)  = required_margin
      for the hardest competitor k.  All other b_k = 0.
    """
    m   = Z0_flat.shape[0]
    W   = _spectrally_normalised(rng.randn(num_classes, m).astype(np.float32))
    raw = W @ Z0_flat                                   # (ℓ,)
    b   = np.zeros(num_classes, dtype=np.float32)
    # required_margin > max_{k≠0} raw_k - raw_0  →  add the shortfall to b[0]
    shortfall = float(np.max(raw[1:]) - raw[0] + required_margin)
    b[0]      = shortfall
    return W, b, 0

# VeriStressGT interface)
def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n",           type=int,   default=4,
                   help="Sequence length (number of tokens)")
    p.add_argument("--d",           type=int,   default=2,
                   help="Token dimension")
    p.add_argument("--d_v",         type=int,   default=4,
                   help="Value projection output dimension")
    p.add_argument("--num_classes", type=int,   default=256,
                   help="Number of output classes")
    p.add_argument("--alpha",       type=float, default=5.0,
                   help="Score scale. Larger means more peaked attention "
                        "and larger L_attn, but gap condition is alpha-independent.")
    p.add_argument("--epsilon",     type=float, default=0.01,
                   help="Perturbation radius ε")
    p.add_argument("--margin_slack",type=float, default=1.0001,
                   help="Multiplier on the required margin. Values just above 1 are tighter (harder to certify)")
    p.add_argument("--seed",        type=int,   default=0)


def run(args: argparse.Namespace) -> dict:
    Path(args.onnx_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.vnnlib_path).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(int(args.seed))
    torch.manual_seed(int(args.seed))

    n, d, d_v = int(args.n), int(args.d), int(args.d_v)
    ell       = int(args.num_classes)
    alpha     = float(args.alpha)
    epsilon   = float(args.epsilon)
    slack     = float(args.margin_slack)

    # 1. Near-orthogonal unit tokens
    X0, mu = _build_near_orthogonal_tokens(n, d, rng)

    lhs = 1.0 - mu
    rhs = 4.0 * epsilon * math.sqrt(d) + 2.0 * epsilon**2 * d
    if lhs <= rhs:
        raise ValueError(
            f"Gap condition cannot be satisfied: 1-μ={lhs:.4f} ≤ {rhs:.4f}.\n"
            f"Increase d or decrease ε.  (α cancels, so changing it won't help.)"
        )

    # 2. Value projection W_V  (spectrally normalised
    W_V = _spectrally_normalised(rng.randn(d, d_v).astype(np.float32))

    # 3. Compute L_attn analytically so we can set the head margin 
    V0_pre = X0 @ W_V                                  # (n, d_v)  = X0·W_V
    B_S, L_attn = compute_L_attn(n, d, alpha, epsilon, V0_pre, W_V)

    # With L_h = 1 (spectrally normalised head), required margin is:
    required_margin = slack * 2.0 * 1.0 * math.sqrt(n) * L_attn * epsilon

    # 4. Attention output at X0  (needed to build the head)            
    X0t, W_Vt = torch.from_numpy(X0), torch.from_numpy(W_V)
    with torch.no_grad():
        A0  = torch.softmax(alpha * (X0t @ X0t.t()), dim=-1)
        Z0  = (A0 @ (X0t @ W_Vt)).numpy()             # (n, d_v)
    Z0_flat = Z0.flatten()

    # 5. Head: spectrally normalised, margin set to required_margin
    W_head, b_head, y = _build_head(Z0_flat, ell, required_margin, rng)

    # 6. Verify full certificate (should always pass, but confirm)  
    cert = check_certificate(X0, alpha, epsilon, W_V, W_head, b_head, verbose=True)
    if not cert["valid"]:
        raise RuntimeError(
            "Certificate unexpectedly invalid after construction. "
            "This is a bug — please report."
        )

    # 7. Export ONNX
    model = FixedPatternAttentionNet(
        n=n, d=d, d_v=d_v, alpha=alpha,
        W_V=W_Vt,
        W_head=torch.from_numpy(W_head),
        b_head=torch.from_numpy(b_head),
    )
    export_pytorch_to_onnx(model, args.onnx_path, input_shape=(1, n * d))

    # 8. Export VNNLIB
    make_box_vnnlib(
        center=X0.flatten().astype(np.float32),
        eps=epsilon,
        out=args.vnnlib_path,
        num_outputs=ell,
        label=y,
    )

    print(f"Wrote ONNX:   {args.onnx_path}")
    print(f"Wrote VNNLIB: {args.vnnlib_path}")

    return {
        "onnx_path":   args.onnx_path,
        "vnnlib_path": args.vnnlib_path,
        "label":       y,
        "epsilon":     epsilon,
        "certificate": cert,
    }


# Standalone entrypoint
def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Fixed-Pattern Attention — provably robust instance constructor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--onnx_path",   required=True)
    ap.add_argument("--vnnlib_path", required=True)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    _main()
