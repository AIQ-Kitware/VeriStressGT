from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import argparse
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper, helper

from VeriStressGT.utils.make_box_vnnlib import make_box_vnnlib
from VeriStressGT.utils.onnx_export import ExportConfig, export_pytorch_to_onnx

CONSTRUCTION_NAME = "mlp_relu.meap"

# ======================
# ONNX post-processing
# ======================

def make_ssa_unique(onnx_in: str, onnx_out: str = None) -> None:
    """Ensure each value name is produced at most once by renaming duplicate outputs."""
    if onnx_out is None:
        onnx_out = onnx_in

    m = onnx.load(onnx_in)
    g = m.graph

    used = set()
    rename: Dict[str, str] = {}

    def fresh(name: str, k: int) -> str:
        return f"{name}__ssa{k}"

    # Rename duplicate node outputs
    for node in g.node:
        for idx, out in enumerate(node.output):
            out_cur = rename.get(out, out)
            if out_cur in used:
                k = 1
                new_name = fresh(out_cur, k)
                while new_name in used:
                    k += 1
                    new_name = fresh(out_cur, k)
                rename[out] = new_name
                node.output[idx] = new_name
                used.add(new_name)
            else:
                used.add(out_cur)

    # Update downstream inputs
    if rename:
        for node in g.node:
            for i, inp in enumerate(node.input):
                if inp in rename:
                    node.input[i] = rename[inp]
        for o in g.output:
            if o.name in rename:
                o.name = rename[o.name]

    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, onnx_out)


def rewrite_matmul_add_to_gemm(onnx_in: str, onnx_out: str = None) -> None:
    """
    Convert:
      MatMul(X, W) + b  -> Gemm(X, W_T, b, transB=1)
      MatMul(X, W)      -> Gemm(X, W_T, 0, transB=1)
    where W_T = W^T is stored as an initializer (out,in).
    """
    if onnx_out is None:
        onnx_out = onnx_in

    m = onnx.load(onnx_in)
    g = m.graph

    init_map = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    init_names = set(init_map.keys())

    consumers: Dict[str, int] = {}
    for node in g.node:
        for inp in node.input:
            consumers[inp] = consumers.get(inp, 0) + 1

    nodes = list(g.node)

    def ensure_w_t(W_name: str) -> str:
        WT_name = W_name + "_T_for_gemm"
        if WT_name in init_names:
            return WT_name
        W = init_map[W_name]
        assert W.ndim == 2
        WT = W.T.astype(np.float32, copy=False)  # (out,in)
        g.initializer.append(numpy_helper.from_array(WT, name=WT_name))
        init_names.add(WT_name)
        init_map[WT_name] = WT
        return WT_name

    def ensure_zero_bias(tag: str, out_dim: int) -> str:
        bname = f"{tag}_zero_bias"
        if bname in init_names:
            return bname
        arr = np.zeros((out_dim,), dtype=np.float32)
        g.initializer.append(numpy_helper.from_array(arr, name=bname))
        init_names.add(bname)
        init_map[bname] = arr
        return bname

    removed_idx = set()
    insert_at: Dict[int, list] = {}

    for i, node in enumerate(nodes):
        if node.op_type != "MatMul" or len(node.input) != 2 or len(node.output) != 1:
            continue
        X, W = node.input
        Y = node.output[0]
        if W not in init_names:
            continue

        W_arr = init_map[W]
        if W_arr.ndim != 2:
            continue
        out_dim = int(W_arr.shape[1])
        WT = ensure_w_t(W)

        fused = False
        if consumers.get(Y, 0) == 1:
            add_i = None
            for j, n2 in enumerate(nodes):
                if n2.op_type == "Add" and len(n2.input) == 2 and len(n2.output) == 1 and (n2.input[0] == Y or n2.input[1] == Y):
                    add_i = j
                    break
            if add_i is not None:
                add_node = nodes[add_i]
                b = add_node.input[1] if add_node.input[0] == Y else add_node.input[0]
                if b in init_names:
                    Z = add_node.output[0]
                    gemm = helper.make_node(
                        "Gemm",
                        inputs=[X, WT, b],
                        outputs=[Z],
                        name=f"{node.name}_fused_gemm" if node.name else f"{Y}_fused_gemm",
                        alpha=1.0, beta=1.0, transB=1,
                    )
                    removed_idx.add(i)
                    removed_idx.add(add_i)
                    insert_at[i] = [gemm]
                    fused = True
        if fused:
            continue

        b0 = ensure_zero_bias(Y, out_dim)
        gemm = helper.make_node(
            "Gemm",
            inputs=[X, WT, b0],
            outputs=[Y],
            name=f"{node.name}_as_gemm" if node.name else f"{Y}_as_gemm",
            alpha=1.0, beta=1.0, transB=1,
        )
        removed_idx.add(i)
        insert_at[i] = [gemm]

    final_nodes = []
    for i, node in enumerate(nodes):
        if i in insert_at:
            final_nodes.extend(insert_at[i])
        if i in removed_idx:
            continue
        final_nodes.append(node)

    del g.node[:]
    g.node.extend(final_nodes)

    # Remove initializers no longer referenced by any node (orphaned by Gemm rewrite)
    used_inputs = {inp for node in g.node for inp in node.input}
    live_inits = [init for init in g.initializer if init.name in used_inputs]
    del g.initializer[:]
    g.initializer.extend(live_inits)

    onnx.checker.check_model(m)
    onnx.save(m, onnx_out)


# ======================
# Config
# ======================

@dataclass
class ProvableMEAPConfig:
    x0: np.ndarray
    epsilon: float
    num_classes: int

    num_pairs: int = 128  # power of 2
    gamma: float = 1e-2
    weight_scale: float = 1.0
    seed: int = 0
    label: Optional[int] = None

    def __post_init__(self):
        self.x0 = np.asarray(self.x0, dtype=np.float32).flatten()
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if self.num_pairs < 2 or (self.num_pairs & (self.num_pairs - 1)) != 0:
            raise ValueError("num_pairs must be power of 2 and >=2")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")
        if self.weight_scale <= 0:
            raise ValueError("weight_scale must be > 0")
        if self.label is not None and not (0 <= self.label < self.num_classes):
            raise ValueError("label out of range")


# ======================
# Modeel
# ======================

class ProvableMEAPPairs(nn.Module):
    """
    h(x) = ReLU(xW + b), producing 2P features.
    """
    def __init__(self, x0: torch.Tensor, epsilon: float, num_pairs: int, gamma: float, weight_scale: float, seed: int):
        super().__init__()
        self.in_dim = int(x0.numel())
        self.P = int(num_pairs)
        self.out_dim = 2 * self.P
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.weight_scale = float(weight_scale)

        rng = np.random.RandomState(seed)

        W_out_in = np.zeros((self.out_dim, self.in_dim), dtype=np.float32)
        b_out = np.zeros((self.out_dim,), dtype=np.float32)

        x0_np = x0.detach().cpu().numpy().astype(np.float32).reshape(-1)

        for p in range(self.P):
            w = rng.randn(self.in_dim).astype(np.float32)
            w /= (np.linalg.norm(w) + 1e-12)

            x0_norm2 = float(np.dot(x0_np, x0_np)) + 1e-12
            proj = float(np.dot(w, x0_np)) / x0_norm2
            w = w - proj * x0_np
            w /= (np.linalg.norm(w) + 1e-12)

            k = 10.0
            rad_target = k * self.gamma
            l1 = float(np.sum(np.abs(w))) + 1e-12
            scale = rad_target / (self.epsilon * l1)
            w = (w * scale * self.weight_scale).astype(np.float32)

            i = 2 * p
            W_out_in[i, :] = w
            W_out_in[i + 1, :] = -w
            b_out[i] = self.gamma
            b_out[i + 1] = self.gamma

        # store as constants so ONNX uses MatMul + Add
        W_in_out = torch.from_numpy(W_out_in).t().contiguous()  # (in,2P)
        b = torch.from_numpy(b_out).contiguous()               # (2P,)

        self.register_buffer("W", W_in_out)
        self.register_buffer("b", b)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MatMul + Add + ReLU
        return self.relu(x @ self.W + self.b)


class SelectEvenOddNoNeg(nn.Module):
    """
    Produce:
      he      = h @ W_even
      ho      = h @ W_odd
      ho_neg  = h @ W_odd_neg   (this equals -ho, but computed via MatMul with a constant matrix)
    This avoids runtime Neg/Sub/Mul.
    """
    def __init__(self, P: int):
        super().__init__()
        W_even = torch.zeros((2 * P, P), dtype=torch.float32)
        W_odd  = torch.zeros((2 * P, P), dtype=torch.float32)
        W_odd_neg = torch.zeros((2 * P, P), dtype=torch.float32)
        for p in range(P):
            W_even[2 * p, p] = 1.0
            W_odd[2 * p + 1, p] = 1.0
            W_odd_neg[2 * p + 1, p] = -1.0  # constant -1 baked into weights
        self.register_buffer("W_even", W_even)
        self.register_buffer("W_odd", W_odd)
        self.register_buffer("W_odd_neg", W_odd_neg)

    def forward(self, h: torch.Tensor):
        he = h @ self.W_even
        ho = h @ self.W_odd
        ho_neg = h @ self.W_odd_neg
        return he, ho, ho_neg


class PairwiseMaxNoNeg(nn.Module):
    """
    max(a,b) = ReLU(a - b) + b
             = ReLU(a + (-b)) + b
    We compute (-b) via ho_neg (MatMul with constant matrix), then only use Add/ReLU/Add.
    """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, he: torch.Tensor, ho: torch.Tensor, ho_neg: torch.Tensor) -> torch.Tensor:
        # d = he - ho  implemented as he + ho_neg
        d = he + ho_neg
        t = self.relu(d)
        return t + ho


class NegateByMatMul(nn.Module):
    """Compute -x using MatMul with a constant -I matrix (no Neg/Mul)."""
    def __init__(self, dim: int):
        super().__init__()
        W = (-torch.eye(dim, dtype=torch.float32)).contiguous()
        self.register_buffer("Wneg", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Wneg


class MinTreeNoNeg(nn.Module):
    """
    cur <- min over entries using:
      t = ReLU(a - b)
      min(a,b) = a - t
    but implemented with Add/ReLU/MatMul only:
      b_neg = cur @ O_neg  (equals -b)
      d = a + b_neg
      t = ReLU(d)
      t_neg = t @ (-I)
      cur = a + t_neg
    """
    def __init__(self, P: int):
        super().__init__()
        if P < 2 or (P & (P - 1)) != 0:
            raise ValueError("P must be power of 2")
        self.P = int(P)
        self.relu = nn.ReLU()

        mats_even = []
        mats_odd_neg = []
        negators = []

        W = self.P
        while W > 1:
            half = W // 2
            E = torch.zeros((W, half), dtype=torch.float32)
            Oneg = torch.zeros((W, half), dtype=torch.float32)  # this will produce -b directly
            for i in range(half):
                E[2 * i, i] = 1.0          # a = cur @ E picks even
                Oneg[2 * i + 1, i] = -1.0  # -b = cur @ Oneg picks odd and negates
            mats_even.append(E)
            mats_odd_neg.append(Oneg)
            negators.append(NegateByMatMul(half))
            W = half

        self.num_levels = len(mats_even)
        for lvl, (E, Oneg) in enumerate(zip(mats_even, mats_odd_neg)):
            self.register_buffer(f"E_{lvl}", E)
            self.register_buffer(f"Oneg_{lvl}", Oneg)
            self.add_module(f"neg_{lvl}", negators[lvl])

    def forward(self, cur: torch.Tensor) -> torch.Tensor:
        for lvl in range(self.num_levels):
            E = getattr(self, f"E_{lvl}")
            Oneg = getattr(self, f"Oneg_{lvl}")
            neg = getattr(self, f"neg_{lvl}")

            a = cur @ E
            b_neg = cur @ Oneg     # equals -b
            d = a + b_neg          # equals a - b
            t = self.relu(d)
            t_neg = neg(t)         # equals -t
            cur = a + t_neg        # equals a - t
        return cur  # (B,1)


class OutputEmbed(nn.Module):
    def __init__(self, num_classes: int, label: int):
        super().__init__()
        W = torch.zeros((1, num_classes), dtype=torch.float32)
        W[0, label] = 1.0
        self.register_buffer("W", W)

    def forward(self, y0: torch.Tensor) -> torch.Tensor:
        return y0 @ self.W  # MatMul


class ProvableMEAPNetCoreOps(nn.Module):
    """
    Full construction, but constrained to export as only MatMul/Add/ReLU (and then we optionally rewrite MatMul->Gemm).
    """
    def __init__(self, x0: torch.Tensor, epsilon: float, num_pairs: int, gamma: float, weight_scale: float, seed: int,
                 num_classes: int, label: int):
        super().__init__()
        self.pairs = ProvableMEAPPairs(x0, epsilon, num_pairs, gamma, weight_scale, seed)
        self.sel = SelectEvenOddNoNeg(num_pairs)
        self.pmax = PairwiseMaxNoNeg()
        self.mintree = MinTreeNoNeg(num_pairs)
        self.out = OutputEmbed(num_classes, label)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pairs(x)
        he, ho, ho_neg = self.sel(h)
        m = self.pmax(he, ho, ho_neg)
        y0 = self.mintree(m)
        return self.out(y0)


# ======================
# Driver
# ======================

def create_and_export_instance(cfg: ProvableMEAPConfig, onnx_path: str, vnnlib_path: str) -> Dict[str, Any]:
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    x0 = cfg.x0.astype(np.float32).flatten()
    in_dim = x0.size
    x0_t = torch.from_numpy(x0)

    label = int(cfg.label) if cfg.label is not None else 0

    model = ProvableMEAPNetCoreOps(
        x0=x0_t,
        epsilon=cfg.epsilon,
        num_pairs=cfg.num_pairs,
        gamma=cfg.gamma,
        weight_scale=cfg.weight_scale,
        seed=cfg.seed,
        num_classes=cfg.num_classes,
        label=label,
    ).eval()

    ex_cfg = ExportConfig(
        opset=13,
        do_constant_folding=False,
        dynamic_batch=False,
        #use_external_data_format=False,
        keep_initializers_as_inputs=False,
    )
    export_pytorch_to_onnx(model, str(onnx_path), (1, in_dim), config=ex_cfg)

    # Keep graph in "core ops". Convert MatMul(+Add) to Gemm to help auto_LiRPA treat them as linear layers.
    rewrite_matmul_add_to_gemm(str(onnx_path), str(onnx_path))
    make_ssa_unique(str(onnx_path), str(onnx_path))

    # Spec unchanged
    make_box_vnnlib(center=x0, eps=cfg.epsilon, out=vnnlib_path, num_outputs=cfg.num_classes, label=label)

    return {
        "onnx_path": onnx_path,
        "vnnlib_path": vnnlib_path,
        "x0": x0,
        "epsilon": cfg.epsilon,
        "label": label,
        "num_pairs": cfg.num_pairs,
        "gamma": cfg.gamma,
        "weight_scale": cfg.weight_scale,
        "gt_margin_lower_bound": cfg.gamma,
        "is_robust": True,
    }


def add_args(p):
    p.add_argument("--input-dim", type=int, default=100)
    p.add_argument("--num-classes", type=int, default=8)
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument("--num-pairs", type=int, default=16)
    p.add_argument("--gamma", type=float, default=1e-3)
    p.add_argument("--weight-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--label", type=int, default=0)


def run(args) -> Dict[str, Any]:
    Path(args.onnx_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.vnnlib_path).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(int(args.seed))
    x0 = (rng.randn(int(args.input_dim)).astype(np.float32) * 0.1)

    cfg = ProvableMEAPConfig(
        x0=x0,
        epsilon=float(args.epsilon),
        num_classes=int(args.num_classes),
        num_pairs=int(args.num_pairs),
        gamma=float(args.gamma),
        weight_scale=float(args.weight_scale),
        seed=int(args.seed),
        label=int(args.label) if args.label is not None else None,
    )
    res = create_and_export_instance(cfg, args.onnx_path, args.vnnlib_path)
    print(f"Wrote ONNX:   {res['onnx_path']}")
    print(f"Wrote VNNLIB: {res['vnnlib_path']}")
    return res


def _main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--onnx_path", required=True)
    ap.add_argument("--vnnlib_path", required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    _main()
