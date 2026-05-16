"""
Convex-corner certified robust instance construction
"""

from __future__ import annotations

import subprocess
import argparse
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from VeriStressGT.utils.make_box_vnnlib import make_box_vnnlib
from VeriStressGT.utils.onnx_export import export_pytorch_to_onnx

CONSTRUCTION_NAME = "mlp_relu.corners"

import sys
from pathlib import Path

def simplify_onnx_inplace(onnx_path: Union[str, Path], log_path: Optional[Union[str, Path]] = None) -> None:
    onnx_path = str(onnx_path)
    cmd = [sys.executable, "-m", "onnxsim", onnx_path, onnx_path]  # in-place: input == output

    if log_path is None:
        # print simplify output to console
        subprocess.run(cmd, check=True)
    else:
        log_path = str(log_path)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
# ---------------------------
# Config
# ---------------------------

@dataclass
class ConvexCornerConfig:
    x0: np.ndarray
    epsilon: float
    num_outputs: int
    label: int = 0

    active_dim: int = 16
    competitors: Optional[List[int]] = None
    num_hinges: int = 256
    hinge_density: float = 1.0
    hinge_l1: float = 1.0
    pattern_density: float = 0.25
    coeff_scale: float = 1.0

    tau: float = 1e-6
    BIG_M: float = 5.0
    seed: int = 0

    def __post_init__(self):
        self.x0 = np.asarray(self.x0, dtype=np.float32).flatten()
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.num_outputs < 2:
            raise ValueError("num_outputs must be >= 2")
        if not (0 <= self.label < self.num_outputs):
            raise ValueError("label out of range")
        if not (1 <= self.active_dim <= self.x0.size):
            raise ValueError("active_dim must be in [1, input_dim]")
        if self.num_hinges < 1:
            raise ValueError("num_hinges must be >= 1")
        if not (0 < self.hinge_density <= 1.0):
            raise ValueError("hinge_density must be in (0,1]")
        if not (0 < self.pattern_density <= 1.0):
            raise ValueError("pattern_density must be in (0,1]")
        if self.hinge_l1 <= 0:
            raise ValueError("hinge_l1 must be > 0")
        if self.coeff_scale <= 0:
            raise ValueError("coeff_scale must be > 0")
        if self.tau < 0:
            raise ValueError("tau must be >= 0")
        if self.BIG_M <= 0:
            raise ValueError("BIG_M must be > 0")


# ---------------------------
# Model
# ---------------------------

class ConvexCornerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        active_dim: int,
        num_classes: int,
        label: int,
        competitors: Sequence[int],
        A: torch.Tensor,
        b: torch.Tensor,
        C: torch.Tensor,
        beta: torch.Tensor,
        BIG_M: float = 5.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.active_dim = int(active_dim)
        self.num_classes = int(num_classes)
        self.label = int(label)
        self.competitors = list(map(int, competitors))
        self.K = len(self.competitors)
        self.m = int(A.shape[0])
        self.BIG_M = float(BIG_M)

        # Selector: take first r coordinates (slice-free)
        self.selector = nn.Linear(self.input_dim, self.active_dim, bias=False)
        with torch.no_grad():
            W = torch.zeros(self.active_dim, self.input_dim, dtype=torch.float32)
            W[torch.arange(self.active_dim), torch.arange(self.active_dim)] = 1.0
            self.selector.weight.copy_(W)
        for p in self.selector.parameters():
            p.requires_grad = False

        # Hinge: t = A x_active + b, then h = relu(t)
        self.hinge = nn.Linear(self.active_dim, self.m, bias=True)
        with torch.no_grad():
            self.hinge.weight.copy_(A)
            self.hinge.bias.copy_(b)
        for p in self.hinge.parameters():
            p.requires_grad = False

        # Competitor layer: (B,m) -> (B,K), with bias = -beta
        self.competitor_layer = nn.Linear(self.m, self.K, bias=True)
        with torch.no_grad():
            self.competitor_layer.weight.copy_(C)       # (K, m)
            self.competitor_layer.bias.copy_(-beta)     # (K,)
        for p in self.competitor_layer.parameters():
            p.requires_grad = False

        # Output scatter + base logits as one affine
        S = torch.zeros(self.K, self.num_classes, dtype=torch.float32)
        for kk, cls in enumerate(self.competitors):
            S[kk, cls] = 1.0

        base = (-self.BIG_M) * torch.ones(self.num_classes, dtype=torch.float32)
        base[self.label] = self.BIG_M

        self.output_layer = nn.Linear(self.K, self.num_classes, bias=True)
        with torch.no_grad():
            self.output_layer.weight.copy_(S.t())  # (num_classes, K)
            self.output_layer.bias.copy_(base)
        for p in self.output_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x_active = self.selector(x)              # (B, r)
        h = F.relu(self.hinge(x_active))         # (B, m)
        z_comp = self.competitor_layer(h)        # (B, K)
        logits = self.output_layer(z_comp)       # (B, C)
        return logits


# ---------------------------
# Helper functions
# ---------------------------

def make_hinges_centered(
    x0_active: np.ndarray,
    m: int,
    r: int,
    hinge_l1: float,
    hinge_density: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    A = np.zeros((m, r), dtype=np.float32)
    b = np.zeros((m,), dtype=np.float32)

    for j in range(m):
        if hinge_density >= 1.0:
            v = rng.standard_normal(size=(r,)).astype(np.float32)
        else:
            nnz = max(1, int(round(hinge_density * r)))
            v = np.zeros((r,), dtype=np.float32)
            idx = rng.choice(r, size=nnz, replace=False)
            v[idx] = rng.standard_normal(size=(nnz,)).astype(np.float32)

        if np.allclose(v, 0):
            v[int(rng.integers(0, r))] = 1.0

        l1 = float(np.sum(np.abs(v)))
        v = (hinge_l1 / (l1 + 1e-12)) * v

        A[j] = v
        b[j] = -float(v @ x0_active)

    return A, b


def make_nonnegative_patterns(
    K: int,
    m: int,
    density: float,
    coeff_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    density = float(density)
    nnz = max(1, int(round(density * m)))
    C = np.zeros((K, m), dtype=np.float32)
    for k in range(K):
        idx = rng.choice(m, size=nnz, replace=False)
        C[k, idx] = float(coeff_scale)
    return C


def enumerate_box_corners(x0_active: np.ndarray, eps: float) -> np.ndarray:
    r = x0_active.size
    corners = np.empty((1 << r, r), dtype=np.float32)
    for i in range(1 << r):
        s = np.empty((r,), dtype=np.float32)
        for bit in range(r):
            s[bit] = 1.0 if ((i >> bit) & 1) else -1.0
        corners[i] = x0_active + eps * s
    return corners


@torch.no_grad()
def compute_beta_via_corner_max(A, b, C, x0, eps, active_dim, tau) -> torch.Tensor:
    x0_active = x0[:active_dim].astype(np.float64)
    corners = enumerate_box_corners(x0_active.astype(np.float32), float(eps))
    X = torch.from_numpy(corners).double()
    A = A.double()
    b = b.double()
    C = C.double()

    T = X @ A.t() + b.unsqueeze(0)
    H = torch.relu(T)
    HK = H @ C.t()
    max_each = HK.max(dim=0).values
    beta = max_each + float(tau)
    return beta.float()


# ---------------------------
# Core creation
# ---------------------------

def create_instance(cfg: ConvexCornerConfig, verbose: bool = True) -> Tuple[Dict[str, Any], ConvexCornerNet]:
    rng = np.random.default_rng(int(cfg.seed))

    x0 = cfg.x0
    d = x0.size
    r = int(cfg.active_dim)

    y = int(cfg.label)
    Ctot = int(cfg.num_outputs)

    if cfg.competitors is None:
        comps = [c for c in range(Ctot) if c != y]
        # keep it small by default
        comps = comps[: min(4, len(comps))]
    else:
        comps = [int(c) for c in cfg.competitors if int(c) != y]

    if len(comps) == 0:
        raise ValueError("Need at least one competitor (class != y).")

    K = len(comps)
    x0_active = x0[:r].astype(np.float32)

    A_np, b_np = make_hinges_centered(
        x0_active=x0_active,
        m=int(cfg.num_hinges),
        r=r,
        hinge_l1=float(cfg.hinge_l1),
        hinge_density=float(cfg.hinge_density),
        rng=rng,
    )
    C_np = make_nonnegative_patterns(
        K=K,
        m=int(cfg.num_hinges),
        density=float(cfg.pattern_density),
        coeff_scale=float(cfg.coeff_scale),
        rng=rng,
    )

    A = torch.from_numpy(A_np)
    b = torch.from_numpy(b_np)
    C = torch.from_numpy(C_np)

    beta = compute_beta_via_corner_max(
        A=A, b=b, C=C,
        x0=x0, eps=float(cfg.epsilon), active_dim=r,
        tau=float(cfg.tau),
    )

    model = ConvexCornerNet(
        input_dim=d,
        active_dim=r,
        num_classes=Ctot,
        label=y,
        competitors=comps,
        A=A,
        b=b,
        C=C,
        beta=beta,
        BIG_M=float(cfg.BIG_M),
    ).eval()

    if verbose:
        print(f"Input dim:       {d}")
        print(f"Active dim r:    {r} (only x[:r] matters)")
        print(f"Classes:         {Ctot}")
        print(f"Label y:         {y}")
        print(f"Competitors:     {comps}")
        print(f"Epsilon:         {cfg.epsilon}")
        print(f"Hinges m:        {cfg.num_hinges}")
        print(f"hinge_l1:        {cfg.hinge_l1}, hinge_density={cfg.hinge_density}")
        print(f"pattern_density: {cfg.pattern_density}, coeff_scale={cfg.coeff_scale}")
        print(f"tau slack:       {cfg.tau}")
        print(f"BIG_M:           {cfg.BIG_M}")

    meta = {
        "label": y,
        "competitors": comps,
        "active_dim": r,
        "num_hinges": int(cfg.num_hinges),
        "tau": float(cfg.tau),
        "BIG_M": float(cfg.BIG_M),
    }
    return meta, model


def create_and_export_instance(
    cfg: ConvexCornerConfig,
    onnx_path: Union[str, Path],
    vnnlib_path: Union[str, Path],
    verbose: bool = True,
) -> Dict[str, Any]:
    onnx_path = Path(onnx_path)
    vnnlib_path = Path(vnnlib_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    vnnlib_path.parent.mkdir(parents=True, exist_ok=True)

    meta, model = create_instance(cfg, verbose=verbose)

    d = int(cfg.x0.size)
    export_pytorch_to_onnx(model, str(onnx_path), (1, d))
    # try:
    #     simplify_onnx_inplace(onnx_path, log_path=onnx_path.with_suffix(".onnxsim.log"))
    # except subprocess.CalledProcessError as e:
    #     print(f"[WARN] onnxsim failed with exit code {e.returncode}; keeping unsimplified ONNX")
    # except Exception as e:
    #     print(f"[WARN] onnxsim failed unexpectedly: {e}; keeping unsimplified ONNX")
    make_box_vnnlib(center=cfg.x0, eps=float(cfg.epsilon), out=str(vnnlib_path), num_outputs=int(cfg.num_outputs), label=int(cfg.label))

    # JSON-serializable result only
    result: Dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "vnnlib_path": str(vnnlib_path),
        "epsilon": float(cfg.epsilon),
        "label": int(cfg.label),
        "x0": cfg.x0.tolist(),
        "is_robust": True,
        "meta": meta,
        "cfg": {
            **{k: v for k, v in asdict(cfg).items() if k != "x0"},
            "x0_shape": list(cfg.x0.shape),
        },
    }
    return result


# ---------------------------
# Construction API
# ---------------------------

def add_args(p: argparse.ArgumentParser):
    # Network / instance
    p.add_argument("--input-dim", type=int, default=100)
    p.add_argument("--num-outputs", type=int, default=10)
    p.add_argument("--label", type=int, default=0)

    # Corner parameters
    p.add_argument("--active-dim", type=int, default=8)
    p.add_argument("--num-hinges", type=int, default=16)
    p.add_argument("--hinge-l1", type=float, default=1.0)
    p.add_argument("--hinge-density", type=float, default=1.0)
    p.add_argument("--pattern-density", type=float, default=0.5)
    p.add_argument("--coeff-scale", type=float, default=1.0)

    # Robustness parameters
    p.add_argument("--epsilon", type=float, default=0.4)
    p.add_argument("--tau", type=float, default=1e-4)
    p.add_argument("--big-m", type=float, default=2.0)

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")


def run(args) -> Dict[str, Any]:
    # Deterministic x0
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))

    x0 = (rng.standard_normal(size=(int(args.input_dim),)).astype(np.float32) * 0.2)

    cfg = ConvexCornerConfig(
        x0=x0,
        epsilon=float(args.epsilon),
        num_outputs=int(args.num_outputs),
        label=int(args.label),
        competitors=None,
        active_dim=int(args.active_dim),
        num_hinges=int(args.num_hinges),
        hinge_density=float(args.hinge_density),
        hinge_l1=float(args.hinge_l1),
        pattern_density=float(args.pattern_density),
        coeff_scale=float(args.coeff_scale),
        tau=float(args.tau),
        BIG_M=float(args.big_m),
        seed=int(args.seed),
    )

    result = create_and_export_instance(
        cfg=cfg,
        onnx_path=args.onnx_path,
        vnnlib_path=args.vnnlib_path,
        verbose=not bool(args.quiet),
    )

    print(f"Wrote ONNX:   {result['onnx_path']}")
    print(f"Wrote VNNLIB: {result['vnnlib_path']}")
    return result


# ---------------------------
# Standalone entrypoint
# ---------------------------

def _main():
    parser = argparse.ArgumentParser(description="Create convex-corner instance")
    # Standalone requires explicit paths (driver will provide these when used via create_benchmark)
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--vnnlib_path", required=True)
    add_args(parser)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    _main()
