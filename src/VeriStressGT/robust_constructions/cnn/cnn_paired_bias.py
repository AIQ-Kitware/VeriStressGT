from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from VeriStressGT.utils.make_box_vnnlib import make_box_vnnlib

CONSTRUCTION_NAME = "cnn.cnn_paired_bias"

#IBP helpers

def _ibp_conv2d(lo, hi, weight, bias, padding):
    w_pos, w_neg = weight.clamp(min=0), weight.clamp(max=0)
    new_lo = F.conv2d(lo, w_pos, padding=padding) + F.conv2d(hi, w_neg, padding=padding)
    new_hi = F.conv2d(hi, w_pos, padding=padding) + F.conv2d(lo, w_neg, padding=padding)
    if bias is not None:
        b = bias.view(1, -1, 1, 1)
        new_lo, new_hi = new_lo + b, new_hi + b
    return new_lo, new_hi


def _ibp_relu(lo, hi):
    return lo.clamp(min=0), hi.clamp(min=0)


def _ibp_linear(lo, hi, weight, bias):
    w_pos, w_neg = weight.clamp(min=0), weight.clamp(max=0)
    new_lo = F.linear(lo, w_pos) + F.linear(hi, w_neg)
    new_hi = F.linear(hi, w_pos) + F.linear(lo, w_neg)
    if bias is not None:
        new_lo, new_hi = new_lo + bias, new_hi + bias
    return new_lo, new_hi


#Network

class FlatWrapper(nn.Module):
    """Reshape flat VNNLIB input (batch, C*H*W) → (batch, C, H, W) and run core."""
    def __init__(self, core: "PairedBiasCNN") -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W = self.core.in_channels, self.core.height, self.core.width
        x = x.view(-1, C, H, W)
        return self.core(x)


class PairedBiasCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        backbone_channels: int,
        num_backbone_layers: int,
        num_pairs: int,
        kernel_size: int,
        num_classes: int,
        label: int,
        margin: float,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for same-padding"
        pad = kernel_size // 2

        self.in_channels   = in_channels
        self.height        = height
        self.width         = width
        self.num_pairs     = num_pairs
        self.num_classes   = num_classes
        self.label         = label
        self.margin        = margin

        # "backbone" layer
        layers: List[nn.Module] = []
        c_in = in_channels
        for _ in range(num_backbone_layers):
            conv = nn.Conv2d(c_in, backbone_channels, kernel_size, padding=pad)
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            conv.bias.data.uniform_(-1e-3, 1e-3)   # small nonzero → stays as initializer
            layers.append(conv)
            layers.append(nn.ReLU())
            c_in = backbone_channels
        self.backbone = nn.Sequential(*layers)

        #  paired conv layer 
        self.paired_conv = nn.Conv2d(c_in, 2 * num_pairs, kernel_size, padding=pad)
        # biases are set later via set_paired_biases(); weights enforced shared too
        nn.init.kaiming_normal_(self.paired_conv.weight, nonlinearity="relu")
        self.paired_conv.bias.data.zero_()

        self.relu    = nn.ReLU()
        self.flatten = nn.Flatten()
        #output klayer
        flat_dim = 2 * num_pairs * height * width
        self.fc = nn.Linear(flat_dim, num_classes, bias=True)
        self._setup_output_layer()

    def _setup_output_layer(self) -> None:
        """Fix the output linear layer to read out the paired-bias sum."""
        P, H, W = self.num_pairs, self.height, self.width
        scale = 1.0 / (P * H * W)          # normalise so f_y ≈ margin + Σ(gaps)·scale

        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.zero_()
            for i in range(P):
                pos_slice = slice(i * H * W,       (i + 1) * H * W)
                neg_slice = slice((i + P) * H * W, (i + P + 1) * H * W)
                self.fc.weight[self.label, pos_slice] = +scale   # +ReLU(s+b_i)
                self.fc.weight[self.label, neg_slice] = -scale   # -ReLU(s+c_i)
            self.fc.bias[self.label] = self.margin                # constant C

    def enforce_weight_sharing(self) -> None:
        """Copy spatial weights so channel i+P is identical to channel i."""
        with torch.no_grad():
            P = self.num_pairs
            self.paired_conv.weight.data[P:] = self.paired_conv.weight.data[:P].clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            x = x.view(-1, self.in_channels, self.height, self.width)
        h = self.backbone(x)
        h = self.paired_conv(h)
        h = self.relu(h)
        h = self.flatten(h)
        return self.fc(h)


#Set bias with IBP (for more instablity)

def set_paired_biases(
    model: PairedBiasCNN,
    x0: torch.Tensor,          # (1, C, H, W)
    eps: float,
    delta: float,
) -> None:

    P = model.num_pairs
    model.eval()
    with torch.no_grad():
        lo, hi = (x0 - eps).clamp(0, 1), (x0 + eps).clamp(0, 1)

        # IBP through backbone
        for m in model.backbone:
            if isinstance(m, nn.Conv2d):
                lo, hi = _ibp_conv2d(lo, hi, m.weight, m.bias, m.padding[0])
            elif isinstance(m, nn.ReLU):
                lo, hi = _ibp_relu(lo, hi)

        # IBP through paired conv WITHOUT bias (weights only, first P channels)
        w   = model.paired_conv.weight[:P]        
        pad = model.paired_conv.padding[0]
        pre_lo_P, _ = _ibp_conv2d(lo, hi, w, bias=None, padding=pad) 
        _, pre_hi_P = _ibp_conv2d(lo, hi, w, bias=None, padding=pad)

        # Per-channel spatial mean as the center t_i
        t = (pre_lo_P[0] + pre_hi_P[0]).mean(dim=(1, 2)) / 2.0   # (P,)

        bias = model.paired_conv.bias.data   # (2P,)
        bias[:P] = -t + delta               # b_i > c_i
        bias[P:] = -t - delta               # c_i

        assert (bias[:P] - bias[P:] > 0).all(), "b_i > c_i violated"


def _sanitize_onnx(model_path: str) -> None: #written via AI to debug export issues
    import onnx
    from onnx import numpy_helper, helper, TensorProto

    m = onnx.load(model_path)

    # --- existing name sanitization ---
    mapping: Dict[str, str] = {}
    used: set = set()

    def safe(name: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_]", "_", name)
        return ("p_" + s) if s and s[0].isdigit() else s

    for init in m.graph.initializer:
        new = base = safe(init.name)
        k = 0
        while new in used:
            k += 1; new = f"{base}_{k}"
        used.add(new)
        mapping[init.name] = new

    for init in m.graph.initializer:
        init.name = mapping[init.name]
    for node in m.graph.node:
        for i, nm in enumerate(node.input):
            if nm in mapping: node.input[i] = mapping[nm]
        for i, nm in enumerate(node.output):
            if nm in mapping: node.output[i] = mapping[nm]
    for v in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        if v.name in mapping: v.name = mapping[v.name]

    init_map = {t.name: t for t in m.graph.initializer}
    for node in m.graph.node:
        if node.op_type != "Conv":
            continue
        attr_names = {a.name for a in node.attribute}
        if "kernel_shape" in attr_names:
            continue
        # infer from weight tensor shape: (out_ch, in_ch, kH, kW)
        w_name = node.input[1]
        if w_name not in init_map:
            continue
        w_shape = list(numpy_helper.to_array(init_map[w_name]).shape)
        kernel_shape = w_shape[2:]   # drop (out_ch, in_ch)
        node.attribute.append(
            helper.make_attribute("kernel_shape", kernel_shape)
        )

    onnx.checker.check_model(m)
    onnx.save(m, model_path)


def export_onnx(model: PairedBiasCNN, onnx_path: str) -> None:
    model.eval()
    C, H, W = model.in_channels, model.height, model.width
    x = torch.randn(1, C, H, W)   # 4D input, no wrapper

    torch.onnx.export(
        model, x, onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=False,
        keep_initializers_as_inputs=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False #required for older envs
    )
    _sanitize_onnx(onnx_path)

    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))


# ---------------------------------------------------------------------------
# Diagnostics

def _ibp_full(model: PairedBiasCNN, x0: torch.Tensor, eps: float):
    """Full IBP pass through the network; returns (lo, hi) on output logits."""
    lo, hi = (x0 - eps).clamp(0, 1), (x0 + eps).clamp(0, 1)
    for m in model.backbone:
        if isinstance(m, nn.Conv2d):
            lo, hi = _ibp_conv2d(lo, hi, m.weight, m.bias, m.padding[0])
        elif isinstance(m, nn.ReLU):
            lo, hi = _ibp_relu(lo, hi)
    lo, hi = _ibp_conv2d(lo, hi,
                         model.paired_conv.weight,
                         model.paired_conv.bias,
                         model.paired_conv.padding[0])
    lo, hi = _ibp_relu(lo, hi)
    lo, hi = lo.flatten(1), hi.flatten(1)
    lo, hi = _ibp_linear(lo, hi, model.fc.weight, model.fc.bias)
    return lo[0], hi[0]


def _count_unstable(model: PairedBiasCNN, x0: torch.Tensor, eps: float) -> dict:
    lo, hi = (x0 - eps).clamp(0, 1), (x0 + eps).clamp(0, 1)
    total = unstable = 0
    for m in model.backbone:
        if isinstance(m, nn.Conv2d):
            lo, hi = _ibp_conv2d(lo, hi, m.weight, m.bias, m.padding[0])
        elif isinstance(m, nn.ReLU):
            total   += lo.numel()
            unstable += int(((lo < 0) & (hi > 0)).sum())
            lo, hi = _ibp_relu(lo, hi)
    lo, hi = _ibp_conv2d(lo, hi,
                         model.paired_conv.weight,
                         model.paired_conv.bias,
                         model.paired_conv.padding[0])
    total   += lo.numel()
    unstable += int(((lo < 0) & (hi > 0)).sum())
    return {"total": total, "unstable": unstable,
            "fraction": unstable / max(total, 1)}

#construction entry point

@dataclass
class PairedBiasCNNConfig:
    in_channels:          int
    height:               int
    width:                int
    backbone_channels:    int
    num_backbone_layers:  int
    num_pairs:            int
    kernel_size:          int
    num_classes:          int
    label:                int
    epsilon:              float
    margin:               float
    delta:                float
    seed:                 int


def create_and_export(
    cfg: PairedBiasCNNConfig,
    onnx_path: str,
    vnnlib_path: str,
) -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = PairedBiasCNN(
        in_channels         = cfg.in_channels,
        height              = cfg.height,
        width               = cfg.width,
        backbone_channels   = cfg.backbone_channels,
        num_backbone_layers = cfg.num_backbone_layers,
        num_pairs           = cfg.num_pairs,
        kernel_size         = cfg.kernel_size,
        num_classes         = cfg.num_classes,
        label               = cfg.label,
        margin              = cfg.margin,
    )

    # x0: uniform in [0.25, 0.75] so the eps-ball stays within [0,1]
    x0 = torch.rand(1, cfg.in_channels, cfg.height, cfg.width) * 0.5 + 0.25
    model.enforce_weight_sharing()
    set_paired_biases(model, x0, cfg.epsilon, cfg.delta)
    model._setup_output_layer()
    model.eval()

    # Sanity check
    with torch.no_grad():
        logits_x0 = model(x0)[0]

    assert logits_x0.argmax().item() == cfg.label, (
        f"Model misclassifies x0: logits={logits_x0.tolist()}"
    )
    actual_margin = float(
        logits_x0[cfg.label] - logits_x0[[k for k in range(cfg.num_classes) if k != cfg.label]].max()
    )
    assert actual_margin > 0, f"Margin at x0 is non-positive: {actual_margin}"

    ibp_lo, ibp_hi = _ibp_full(model, x0, cfg.epsilon)
    ibp_worst = float(
        ibp_lo[cfg.label] - ibp_hi[[k for k in range(cfg.num_classes) if k != cfg.label]].max()
    )
    unstable = _count_unstable(model, x0, cfg.epsilon)

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    Path(vnnlib_path).parent.mkdir(parents=True, exist_ok=True)

    export_onnx(model, onnx_path)

    x0_flat = x0.squeeze(0).numpy().reshape(-1)   # (C*H*W,) for VNNLIB
    make_box_vnnlib(
        center     = x0_flat,
        eps        = cfg.epsilon,
        out        = vnnlib_path,
        num_outputs= cfg.num_classes,
        label      = cfg.label,
    )

    return {
        "onnx_path":          onnx_path,
        "vnnlib_path":        vnnlib_path,
        "is_robust":          True,
        "label":              cfg.label,
        "epsilon":            cfg.epsilon,
        "margin":             cfg.margin,
        "delta":              cfg.delta,
        "actual_margin_x0":   actual_margin,
        "ibp_worst_margin":   ibp_worst,
        "unstable_relus":     unstable,
        "num_pairs":          cfg.num_pairs,
        "spatial_pairs":      cfg.num_pairs * cfg.height * cfg.width,
        "seed":               cfg.seed,
    }


# Interfacce for VeriStressGT

def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--in-channels",          type=int,   default=3)
    p.add_argument("--height",               type=int,   default=4,
                   help="Spatial height H. CROWN relaxation error ∝ P·H·W.")
    p.add_argument("--width",                type=int,   default=4,
                   help="Spatial width W.")
    p.add_argument("--backbone-channels",    type=int,   default=2,
                   help="Output channels of each backbone Conv2d layer.")
    p.add_argument("--num-backbone-layers",  type=int,   default=2,
                   help="Number of Conv2d+ReLU layers before the paired layer.")
    p.add_argument("--num-pairs",            type=int,   default=8,
                   help="Number of paired-bias channel pairs P. "
                        "CROWN's accumulated relaxation error ∝ P.")
    p.add_argument("--kernel-size",          type=int,   default=3,
                   help="Conv kernel size (must be odd).")
    p.add_argument("--num-classes",          type=int,   default=10)
    p.add_argument("--label",                type=int,   default=0)
    p.add_argument("--epsilon",  type=float, default=0.05,
                   help="L_inf perturbation radius.")
    p.add_argument("--margin",   type=float, default=1e-3,
                   help="Provable margin C (bias on label logit). "
                        "Smaller → harder for CROWN to close the gap.")
    p.add_argument("--delta",    type=float, default=None,
                   help="Half-width of bias gap b_i - c_i = 2*delta. "
                        "Defaults to epsilon/2. Controls per-pair relaxation error.")
    p.add_argument("--seed",     type=int,   default=0)


def run(args) -> Dict[str, Any]:
    delta = float(args.delta) if args.delta is not None else float(args.epsilon) / 2.0

    cfg = PairedBiasCNNConfig(
        in_channels         = int(args.in_channels),
        height              = int(args.height),
        width               = int(args.width),
        backbone_channels   = int(args.backbone_channels),
        num_backbone_layers = int(args.num_backbone_layers),
        num_pairs           = int(args.num_pairs),
        kernel_size         = int(args.kernel_size),
        num_classes         = int(args.num_classes),
        label               = int(args.label),
        epsilon             = float(args.epsilon),
        margin              = float(args.margin),
        delta               = delta,
        seed                = int(args.seed),
    )

    result = create_and_export(cfg, args.onnx_path, args.vnnlib_path)
    print(f"  Actual margin at x0:  {result['actual_margin_x0']:.6f}")
    print(f"  IBP worst margin:     {result['ibp_worst_margin']:.6f}"
          + ("  IBP certifies (too easy!)" if result['ibp_worst_margin'] > 0
             else " IBP fails, real work needed ✓"))
    print(f"  Unstable ReLUs:       {result['unstable_relus']['unstable']}"
          f" / {result['unstable_relus']['total']}"
          f"  ({result['unstable_relus']['fraction']:.1%})")
    print()
    print(f"  Ground truth:         UNSAT (f_y ≥ margin > 0 = f_k for all x)")
    print(f"  Wrote ONNX:           {result['onnx_path']}")
    print(f"  Wrote VNNLIB:         {result['vnnlib_path']}")
    print("=" * 60)

    return result

# Standalone entrypoint

def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Paired-bias CNN benchmark constructor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--onnx_path",   required=True)
    ap.add_argument("--vnnlib_path", required=True)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    _main()
