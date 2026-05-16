from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONSTRUCTION_NAME = "cnn.deep_contractive_cnn"


def add_args(parser):
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="L-inf perturbation radius")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--channels", type=int, default=16,
                        help="Number of channels in each contractive layer")
    parser.add_argument("--depth", type=int, default=6,
                        help="Number of contractive Conv+ReLU layers (primary rho knob)")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--contraction-rate", type=float, default=0.9,
                        help="lambda: spectral norm of each conv weight. "
                             "Must be in (0, 1).")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.001,
                        help="Floor on the slack B - cert_bound. "
                             "Smaller -> instance closer to robustness boundary.")
    parser.add_argument("--bias-instability-frac", type=float, default=0.4,
                        help="Target fraction of unstable ReLUs per contractive "
                             "layer under IBP. In (0, 1).")
    parser.add_argument("--n-verify-samples", type=int, default=20000)


# IBP helpers

def _ibp_conv2d(lo, hi, weight, bias, padding):
    w_pos, w_neg = weight.clamp(min=0), weight.clamp(max=0)
    new_lo = F.conv2d(lo, w_pos, padding=padding) + F.conv2d(hi, w_neg, padding=padding)
    new_hi = F.conv2d(hi, w_pos, padding=padding) + F.conv2d(lo, w_neg, padding=padding)
    if bias is not None:
        new_lo = new_lo + bias.view(1, -1, 1, 1)
        new_hi = new_hi + bias.view(1, -1, 1, 1)
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

# Spectral norm utils
def _spectral_norm_power_iter(W: torch.Tensor, n_iter: int = 20) -> float:
    """Estimate largest singular value of W via power iteration.
    W is reshaped to (out, -1) — appropriate for Conv2d (K, C, kH, kW)."""
    W2d = W.reshape(W.shape[0], -1).detach().float()
    u = torch.randn(W2d.shape[0], 1, device=W.device)
    u = u / u.norm()
    for _ in range(n_iter):
        v = W2d.t() @ u;  v = v / v.norm()
        u = W2d @ v;      u = u / u.norm()
    return abs((u.t() @ W2d @ v).item())


def _normalize_to_spectral_norm(conv: nn.Conv2d, target: float) -> None:
    """Scale conv.weight in-place so its spectral norm equals target."""
    with torch.no_grad():
        sigma = _spectral_norm_power_iter(conv.weight)
        if sigma > 1e-8:
            conv.weight.mul_(target / sigma)

class FlatWrapper(nn.Module):
    """Accept flat input (B, C*H*W) as required by VNNLIB/verifiers and
    reshape to (B, C, H, W) before the inner model."""
    def __init__(self, core: "DeepContractiveCNN"):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.core.in_channels, self.core.height, self.core.width)
        return self.core(x)


class DeepContractiveCNN(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        height: int = 8,
        width: int = 8,
        channels: int = 16,
        depth: int = 6,
        kernel_size: int = 3,
        contraction_rate: float = 0.9,
        num_classes: int = 5,
        label: int = 0,
        margin: float = 0.01,
    ):
        super().__init__()
        assert 0 < contraction_rate < 1, "contraction_rate must be in (0, 1)"
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.in_channels      = in_channels
        self.height           = height
        self.width            = width
        self.channels         = channels
        self.depth            = depth
        self.contraction_rate = contraction_rate
        self.num_classes      = num_classes
        self.label            = label
        self.margin           = margin

        pad = kernel_size // 2

        # Input projection: in_channels -> channels (1x1 conv, no spatial mixing)
        self.input_proj = nn.Conv2d(in_channels, channels, kernel_size=1, bias=True)

        # Contractive body
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=True)
            for _ in range(depth)
        ])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(depth)])

        # Output: Flatten then Linear
        self.flatten = nn.Flatten()
        flat_dim = channels * height * width
        self.fc = nn.Linear(flat_dim, num_classes, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_proj.bias)
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, self.in_channels, self.height, self.width)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        h = F.relu(self.input_proj(x))
        for conv, relu in zip(self.convs, self.relus):
            h = relu(conv(h))

        h = self.flatten(h)   # (B, channels*H*W)
        return self.fc(h)

def enforce_contraction(model: DeepContractiveCNN) -> None:
    """Set every contractive conv to spectral norm = contraction_rate."""
    for conv in model.convs:
        _normalize_to_spectral_norm(conv, model.contraction_rate)


def set_biases_for_instability(
    model: DeepContractiveCNN,
    x0: torch.Tensor,
    eps: float,
    instability_frac: float = 0.4,
) -> None:
    model.eval()
    with torch.no_grad():
        lo = (x0 - eps)
        hi = (x0 + eps)
        if lo.dim() == 3:
            lo, hi = lo.unsqueeze(0), hi.unsqueeze(0)

        # Through input projection + ReLU
        lo, hi = _ibp_conv2d(lo, hi,
                              model.input_proj.weight,
                              model.input_proj.bias,
                              padding=0)
        lo, hi = _ibp_relu(lo, hi)

        # Through each contractive layer
        for conv in model.convs:
            pad = conv.padding[0]

            # IBP without bias to get pre-activation range from weights alone
            lo_pre, hi_pre = _ibp_conv2d(lo, hi, conv.weight, bias=None, padding=pad)

            # Per-channel mean over spatial (bias is per-channel, not spatial)
            s_lo = lo_pre[0].mean(dim=(1, 2))   # (channels,)
            s_hi = hi_pre[0].mean(dim=(1, 2))

            # Place crossing at instability_frac of the way from s_lo to s_hi:
            #   bias_c = -crossing_c  =>  pre_lo+bias < 0 < pre_hi+bias
            crossing = s_lo + instability_frac * (s_hi - s_lo)
            conv.bias.data.copy_(-crossing)

            # Propagate with the newly set bias
            lo, hi = _ibp_conv2d(lo, hi, conv.weight, conv.bias, padding=pad)
            lo, hi = _ibp_relu(lo, hi)


def setup_output_layer(
    model: DeepContractiveCNN,
    x0: torch.Tensor,
    eps: float,
) -> float:
    with torch.no_grad():
        C, H, W  = model.channels, model.height, model.width
        flat_dim = C * H * W

        model.fc.weight.zero_()
        model.fc.bias.zero_()

        model.fc.weight[model.label] = 1.0 / flat_dim
        sigma_proj = _spectral_norm_power_iter(model.input_proj.weight)
        w_out_l1   = model.fc.weight[model.label].abs().sum().item()  # = 1.0
        cert_bound = sigma_proj * (model.contraction_rate ** model.depth) * 2 * eps * w_out_l1

        slack = max(model.margin, 0.1 * cert_bound)
        B = cert_bound + slack
        model.fc.bias[model.label] = B

        return float(B)

def compute_true_lipschitz_bound(model: DeepContractiveCNN) -> float:
    """Upper bound on overall Lipschitz constant: sigma_proj * lambda^D * ||w_out||_1."""
    sigma_proj = _spectral_norm_power_iter(model.input_proj.weight)
    w_out_l1   = model.fc.weight[model.label].abs().sum().item()
    return sigma_proj * (model.contraction_rate ** model.depth) * w_out_l1


def count_unstable_relus(
    model: DeepContractiveCNN,
    x0: torch.Tensor,
    eps: float,
) -> dict:
    model.eval()
    with torch.no_grad():
        lo = (x0 - eps)
        hi = (x0 + eps)
        if lo.dim() == 3:
            lo, hi = lo.unsqueeze(0), hi.unsqueeze(0)

        total, unstable = 0, 0

        # Input proj ReLU
        lo, hi = _ibp_conv2d(lo, hi, model.input_proj.weight,
                              model.input_proj.bias, padding=0)
        total    += lo.numel()
        unstable += int(((lo < 0) & (hi > 0)).sum())
        lo, hi    = _ibp_relu(lo, hi)

        # Contractive layers
        for conv in model.convs:
            lo, hi = _ibp_conv2d(lo, hi, conv.weight, conv.bias,
                                  padding=conv.padding[0])
            total    += lo.numel()
            unstable += int(((lo < 0) & (hi > 0)).sum())
            lo, hi    = _ibp_relu(lo, hi)

        return {
            "total_relus":       total,
            "unstable_relus":    unstable,
            "fraction_unstable": unstable / max(total, 1),
        }


def ibp_output_bounds(
    model: DeepContractiveCNN,
    x0: torch.Tensor,
    eps: float,
):
    model.eval()
    with torch.no_grad():
        lo = (x0 - eps)
        hi = (x0 + eps)
        if lo.dim() == 3:
            lo, hi = lo.unsqueeze(0), hi.unsqueeze(0)

        lo, hi = _ibp_conv2d(lo, hi, model.input_proj.weight,
                              model.input_proj.bias, padding=0)
        lo, hi = _ibp_relu(lo, hi)

        for conv in model.convs:
            lo, hi = _ibp_conv2d(lo, hi, conv.weight, conv.bias,
                                  padding=conv.padding[0])
            lo, hi = _ibp_relu(lo, hi)

        # Flatten then Linear (no GAP)
        lo = lo.flatten(1)
        hi = hi.flatten(1)
        lo, hi = _ibp_linear(lo, hi, model.fc.weight, model.fc.bias)

        return lo[0].numpy(), hi[0].numpy()


def verify_certificate_empirically(
    model: DeepContractiveCNN,
    x0: torch.Tensor,
    eps: float,
    n_samples: int = 20000,
) -> dict:
    model.eval()
    with torch.no_grad():
        if x0.dim() == 3:
            x0 = x0.unsqueeze(0)
        deltas  = (torch.rand(n_samples, *x0.shape[1:]) * 2 - 1) * eps
        xs      = x0.expand(n_samples, -1, -1, -1) + deltas
        logits  = model(xs)
        preds   = logits.argmax(1)
        other   = [k for k in range(model.num_classes) if k != model.label]
        margins = logits[:, model.label].unsqueeze(1) - logits[:, other]
        return {
            "n_samples":               n_samples,
            "all_correct":             bool((preds == model.label).all().item()),
            "min_margin_over_samples": float(margins.min().item()),
        }


def _export_onnx(model: DeepContractiveCNN, path: str) -> None:
    import re
    import onnx

    model.eval()
    wrapped = FlatWrapper(model).eval()
    d = model.in_channels * model.height * model.width
    dummy = torch.randn(1, d)

    torch.onnx.export(
        wrapped, dummy, path,
        export_params=True,
        opset_version=17,
        do_constant_folding=False,
        keep_initializers_as_inputs=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False
    )

    m = onnx.load(path)

    # Sanitize initializer names
    used, mapping = set(), {}
    for init in m.graph.initializer:
        s = re.sub(r"[^A-Za-z0-9_]", "_", init.name)
        if s and s[0].isdigit():
            s = "p_" + s
        base, k = s, 0
        while s in used:
            k += 1
            s = f"{base}_{k}"
        used.add(s)
        mapping[init.name] = s

    for init in m.graph.initializer:
        init.name = mapping[init.name]
    for node in m.graph.node:
        for i, nm in enumerate(node.input):
            if nm in mapping: node.input[i] = mapping[nm]
        for i, nm in enumerate(node.output):
            if nm in mapping: node.output[i] = mapping[nm]
    for v in (list(m.graph.input) + list(m.graph.output)
              + list(m.graph.value_info)):
        if v.name in mapping: v.name = mapping[v.name]

    onnx.checker.check_model(m)
    onnx.save(m, path)

def _write_vnnlib(
    x0: np.ndarray,
    eps: float,
    num_outputs: int,
    label: int,
    path: str,
) -> None:
    x0 = x0.astype(float).reshape(-1)
    d  = x0.size
    lo, hi = x0 - eps, x0 + eps

    lines = (
        [f"(declare-const X_{i} Real)" for i in range(d)]
        + [f"(declare-const Y_{j} Real)" for j in range(num_outputs)]
        + ["", "; input bounds"]
        + [f"(assert (>= X_{i} {lo[i]}))\n(assert (<= X_{i} {hi[i]}))"
           for i in range(d)]
        + ["", "; robustness violation: exists j != label with Y_j >= Y_label"]
    )
    disj = [f"(>= Y_{j} Y_{label})" for j in range(num_outputs) if j != label]
    if len(disj) == 1:
        lines.append(f"(assert (and {disj[0]}))")
    else:
        lines.append(
            "(assert (or\n"
            + "\n".join(f"    (and {c})" for c in disj)
            + "\n))"
        )
    Path(path).write_text("\n".join(lines) + "\n")

# completenessbecnh entry point

def run(args):
    import json

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert args.label < args.num_classes
    assert 0 < args.contraction_rate < 1

    model = DeepContractiveCNN(
        in_channels      = args.in_channels,
        height           = args.height,
        width            = args.width,
        channels         = args.channels,
        depth            = args.depth,
        kernel_size      = args.kernel_size,
        contraction_rate = args.contraction_rate,
        num_classes      = args.num_classes,
        label            = args.label,
        margin           = args.margin,
    )

    x0 = torch.rand(1, args.in_channels, args.height, args.width) * 0.5 + 0.25

    enforce_contraction(model)
    set_biases_for_instability(model, x0, args.epsilon, args.bias_instability_frac)
    B = setup_output_layer(model, x0, args.epsilon)

    model.eval()
    with torch.no_grad():
        logits_np     = model(x0)[0].numpy()
        other         = [k for k in range(args.num_classes) if k != args.label]
        actual_margin = min(logits_np[args.label] - logits_np[k] for k in other)

    true_lip   = compute_true_lipschitz_bound(model)
    cert_bound = true_lip * 2 * args.epsilon

    print(f"\n=== Deep Contractive CNN ===")
    print(f"Depth: {args.depth}   lambda: {args.contraction_rate}")
    print(f"True Lipschitz bound:   {true_lip:.6f}")
    print(f"Cert perturbation:      {cert_bound:.6f}")
    print(f"Certified margin B:     {B:.6f}")
    print(f"Actual margin at x0:    {actual_margin:.6f}")
    print(f"Certificate slack B-cb: {B - cert_bound:.6f}")

    relu_stats = count_unstable_relus(model, x0, args.epsilon)
    print(f"\nReLU instability:")
    print(f"  Total:    {relu_stats['total_relus']}")
    print(f"  Unstable: {relu_stats['unstable_relus']}")
    print(f"  Fraction: {relu_stats['fraction_unstable']:.2%}")

    ibp_lo, ibp_hi = ibp_output_bounds(model, x0, args.epsilon)
    ibp_worst = min(ibp_lo[args.label] - ibp_hi[k] for k in other)
    print(f"\nDifficulty diagnostic:")
    print(f"  True slack (B - cert_bound): {B - cert_bound:.6f}")
    print(f"  IBP worst margin:            {ibp_worst:.6f}")
    if ibp_worst > 0:
        print("  -> IBP certifies. Increase --depth or --channels.")
    else:
        print("  -> IBP cannot certify. Hard for Marabou/NeuralSAT.")

    emp = verify_certificate_empirically(model, x0, args.epsilon, args.n_verify_samples)
    print(f"\nEmpirical check ({emp['n_samples']} samples):")
    print(f"  All correct: {emp['all_correct']}")
    print(f"  Min margin:  {emp['min_margin_over_samples']:.6f}")

    try:
        from VeriStressGT.utils import export_pytorch_to_onnx, make_box_vnnlib
        export_pytorch_to_onnx(
            FlatWrapper(model), args.onnx_path,
            input_shape=(args.in_channels * args.height * args.width,),
        )
        make_box_vnnlib(
            center     = x0.squeeze(0).numpy(),
            eps        = args.epsilon,
            out        = args.vnnlib_path,
            num_outputs= args.num_classes,
            label      = args.label,
        )
    except ImportError:
        _export_onnx(model, args.onnx_path)
        _write_vnnlib(x0.squeeze(0).numpy(), args.epsilon,
                      args.num_classes, args.label, args.vnnlib_path)

    print(f"\nExported ONNX:   {args.onnx_path}")
    print(f"Exported VNNLIB: {args.vnnlib_path}")

    # ---- Certificate JSON ----
    cert_path = str(Path(args.onnx_path).parent / "certificate.json")
    Path(cert_path).write_text(json.dumps({
        "method":                   "deep_contractive_cnn",
        "certificate_type":         "contraction mapping (global)",
        "certificate_argument":     "f_y(x) >= B - sigma_proj * lambda^D * 2*eps",
        "certified_margin_B":       B,
        "cert_perturbation_bound":  cert_bound,
        "actual_margin_at_x0":      float(actual_margin),
        "depth":                    args.depth,
        "contraction_rate":         args.contraction_rate,
        "channels":                 args.channels,
        "epsilon":                  args.epsilon,
        "label":                    args.label,
        "num_classes":              args.num_classes,
        "relu_stats":               relu_stats,
        "ibp_worst_margin":         float(ibp_worst),
        "empirical_verification":   emp,
        "seed":                     args.seed,
    }, indent=2))
    print(f"Saved certificate: {cert_path}")
    