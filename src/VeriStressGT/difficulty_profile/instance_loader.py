"""
Instance loader: parse ONNX + VNNLIB into a usable PyTorch object.

Provides:
  - A torch.nn.Module wrapping the ONNX model (via onnx2pytorch or onnxruntime)
  - The margin function g(x) = min_{k != y} [f_y(x) - f_k(x)]
  - Per-class margin functions g_k(x) = f_y(x) - f_k(x)
  - Uniform sampling from B_eps(x0) under the given L_p norm
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class VerificationInstance:
    """Parsed verification instance ready for profile estimation."""

    # Core data
    x0: "torch.Tensor"            # (input_dim,) on device
    epsilon: float
    true_class: int
    input_dim: int
    num_classes: int
    norm_p: str                    # "inf", "2", "1"
    device: str

    # The network as a callable: x -> logits
    # x is (batch, input_dim) float32, returns (batch, num_classes)
    forward_fn: object = None

    # Precomputed
    margin_at_x0: float = 0.0
    logits_at_x0: "torch.Tensor" = None

    # Original paths
    onnx_path: str = ""
    vnnlib_path: str = ""

    # Shape info for CNNs / attention: the ONNX input shape (without batch)
    onnx_input_shape: tuple = ()

    def margin(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Overall margin g(x) = min_{k != y} [f_y(x) - f_k(x)].
        x: (batch, input_dim) or (input_dim,)
        Returns: (batch,) tensor.
        """
        import torch
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Reshape to ONNX input shape if needed
        if self.onnx_input_shape and x.shape[1:] != self.onnx_input_shape:
            x = x.view(-1, *self.onnx_input_shape)
        logits = self.forward_fn(x)  # (batch, C)
        y = self.true_class
        fy = logits[:, y]                          # (batch,)
        # g_k for all k != y, then take min
        mask = torch.ones(self.num_classes, dtype=torch.bool, device=x.device)
        mask[y] = False
        fk = logits[:, mask]                       # (batch, C-1)
        margins = fy.unsqueeze(1) - fk             # (batch, C-1)
        return margins.min(dim=1).values           # (batch,)

    def margin_per_class(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Per-class margins g_k(x) = f_y(x) - f_k(x) for all k != y.
        Returns: (batch, C-1) tensor.
        """
        import torch
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.onnx_input_shape and x.shape[1:] != self.onnx_input_shape:
            x = x.view(-1, *self.onnx_input_shape)
        logits = self.forward_fn(x)
        y = self.true_class
        fy = logits[:, y]
        mask = torch.ones(self.num_classes, dtype=torch.bool, device=x.device)
        mask[y] = False
        fk = logits[:, mask]
        return fy.unsqueeze(1) - fk

    def sample_uniform(self, n: int) -> "torch.Tensor":
        """
        Sample n points uniformly from B_eps(x0) under the instance's norm.
        Returns: (n, input_dim) tensor on self.device.
        """
        import torch
        x0 = self.x0.unsqueeze(0).expand(n, -1)   # (n, d)
        if self.norm_p == "inf":
            # Uniform in L_inf ball: each coord uniform in [-eps, eps]
            delta = (2 * torch.rand_like(x0) - 1) * self.epsilon
        elif self.norm_p == "2":
            # Uniform in L_2 ball: sample on sphere, scale by U^{1/d}
            d = self.input_dim
            z = torch.randn_like(x0)
            z = z / z.norm(dim=1, keepdim=True).clamp(min=1e-12)
            r = torch.rand(n, 1, device=self.device).pow(1.0 / d) * self.epsilon
            delta = z * r
        else:
            raise NotImplementedError(f"Sampling for L_{self.norm_p} not implemented")
        return x0 + delta

    def sample_boundary(self, n: int) -> "torch.Tensor":
        """
        Sample n points on the boundary of B_eps(x0).
        Returns: (n, input_dim) tensor.
        """
        import torch
        x0 = self.x0.unsqueeze(0).expand(n, -1)
        if self.norm_p == "inf":
            # Each coordinate is either x0_i - eps or x0_i + eps with prob 0.5
            signs = 2 * torch.randint(0, 2, x0.shape, device=self.device).float() - 1
            delta = signs * self.epsilon
        elif self.norm_p == "2":
            d = self.input_dim
            z = torch.randn_like(x0)
            z = z / z.norm(dim=1, keepdim=True).clamp(min=1e-12)
            delta = z * self.epsilon
        else:
            raise NotImplementedError(f"Boundary sampling for L_{self.norm_p}")
        return x0 + delta


def load_instance(
    onnx_path: str,
    vnnlib_path: str,
    device: str = "cpu",
) -> VerificationInstance:
    """
    Parse an ONNX model and VNNLIB spec into a VerificationInstance.

    The ONNX model is loaded as a PyTorch module (via onnx2pytorch if
    available, otherwise via an onnxruntime wrapper).  The VNNLIB file
    is parsed to extract x0, epsilon, and the true class.
    """
    import torch

    # ---- Parse VNNLIB ----
    x0_np, epsilon, true_class, num_outputs, norm_p = _parse_vnnlib(vnnlib_path)
    input_dim = x0_np.size
    x0 = torch.from_numpy(x0_np.flatten().astype(np.float32)).to(device)

    # ---- Parse ONNX input shape ----
    onnx_input_shape = _parse_onnx_input_shape(onnx_path)

    # Validate shape vs VNNLIB input count
    if onnx_input_shape:
        shape_prod = 1
        for s in onnx_input_shape:
            shape_prod *= s
        if shape_prod != input_dim:
            onnx_input_shape = (input_dim,)

    # ---- Load model via onnxruntime ----
    # We use ORT exclusively for profile estimation.  onnx2pytorch is
    # unreliable on many ONNX models (broken Reshape backward, Conv op
    # quirks, etc.) and profile estimation is an offline computation
    # where reliability matters more than speed.  Gradients are computed
    # via finite differences through the ORT wrapper's autograd.Function.

    def _make_test_input(x0_flat, shape):
        """Create (1, *shape) test input from flat x0."""
        x = x0_flat.unsqueeze(0)
        if shape and x.shape[1:] != shape:
            x = x.view(-1, *shape)
        return x

    ort_wrapper = _OrtAutogradWrapper(onnx_path, device, onnx_input_shape)

    # Validate shape: try parsed shape, then flat, then infer from session
    model = None
    logits_at_x0 = None
    for shape_candidate in [onnx_input_shape, (input_dim,)]:
        try:
            ort_wrapper.input_shape = shape_candidate
            x_test = _make_test_input(x0, shape_candidate)
            with torch.no_grad():
                logits_at_x0 = ort_wrapper(x_test).squeeze(0)
            model = ort_wrapper
            onnx_input_shape = shape_candidate
            break
        except Exception:
            continue

    if model is None:
        # Last resort: read shape from ORT session directly
        ort_shape = ort_wrapper.session.get_inputs()[0].shape
        fixed_dims = [d for d in ort_shape if isinstance(d, int)]
        if len(fixed_dims) < len(ort_shape):
            inferred = tuple(fixed_dims)
        else:
            inferred = tuple(fixed_dims[1:]) if len(fixed_dims) > 1 else tuple(fixed_dims)
        ort_wrapper.input_shape = inferred
        onnx_input_shape = inferred
        try:
            x_test = _make_test_input(x0, inferred)
            with torch.no_grad():
                logits_at_x0 = ort_wrapper(x_test).squeeze(0)
            model = ort_wrapper
        except Exception as e:
            raise RuntimeError(
                f"Could not load ONNX model via onnxruntime. "
                f"Tried shapes: {[onnx_input_shape, (input_dim,), inferred]}. "
                f"Last error: {e}"
            ) from e

    # ---- Compute metadata ----
    num_classes = int(logits_at_x0.shape[0])
    predicted = int(logits_at_x0.argmax().item())
    if true_class is None:
        true_class = predicted
    if num_outputs is not None:
        num_classes = max(num_classes, num_outputs)

    fy = logits_at_x0[true_class].item()
    other_max = max(logits_at_x0[k].item() for k in range(num_classes) if k != true_class)
    margin_at_x0 = fy - other_max

    inst = VerificationInstance(
        x0=x0,
        epsilon=epsilon,
        true_class=true_class,
        input_dim=input_dim,
        num_classes=num_classes,
        norm_p=norm_p,
        device=device,
        forward_fn=model,
        margin_at_x0=margin_at_x0,
        logits_at_x0=logits_at_x0,
        onnx_path=onnx_path,
        vnnlib_path=vnnlib_path,
        onnx_input_shape=onnx_input_shape,
    )
    return inst




# --------------------------------------------------------------------------
# VNNLIB parser
# --------------------------------------------------------------------------

def _parse_vnnlib(path: str) -> Tuple[np.ndarray, float, Optional[int], Optional[int], str]:
    """
    Minimal VNNLIB parser for box specifications.

    Extracts:
      - x0 (center of the box)
      - epsilon (half-width, assuming symmetric box = L_inf)
      - true_class (from the output constraints)
      - num_outputs (number of output variables declared)
      - norm_p (always "inf" for box specs)

    Returns: (x0, epsilon, true_class, num_outputs, norm_p)
    """
    text = Path(path).read_text()

    # Extract input variable declarations and bounds
    # Pattern: (declare-const X_i Real) ... (assert (<= X_i upper)) (assert (>= X_i lower))
    input_bounds = {}
    output_vars = []

    # Find all declared variables
    for m in re.finditer(r'\(declare-const\s+(X_\d+)\s+Real\)', text):
        var = m.group(1)
        idx = int(var.split('_')[1])
        input_bounds[idx] = [None, None]  # [lower, upper]

    for m in re.finditer(r'\(declare-const\s+(Y_\d+)\s+Real\)', text):
        var = m.group(1)
        idx = int(var.split('_')[1])
        output_vars.append(idx)

    # Parse assertions for input bounds
    # (assert (<= X_i value))  -> upper bound
    # (assert (>= X_i value))  -> lower bound
    for m in re.finditer(r'\(assert\s+\(<=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)\)', text):
        idx, val = int(m.group(1)), float(m.group(2))
        if idx in input_bounds:
            input_bounds[idx][1] = val

    for m in re.finditer(r'\(assert\s+\(>=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)\)', text):
        idx, val = int(m.group(1)), float(m.group(2))
        if idx in input_bounds:
            input_bounds[idx][0] = val

    if not input_bounds:
        raise ValueError(f"No input bounds found in {path}")

    n_inputs = max(input_bounds.keys()) + 1
    lowers = np.zeros(n_inputs, dtype=np.float64)
    uppers = np.zeros(n_inputs, dtype=np.float64)

    for idx in range(n_inputs):
        if idx not in input_bounds or input_bounds[idx][0] is None or input_bounds[idx][1] is None:
            raise ValueError(f"Missing bounds for X_{idx} in {path}")
        lowers[idx] = input_bounds[idx][0]
        uppers[idx] = input_bounds[idx][1]

    # x0 = midpoint, epsilon = half-width (should be constant across dims for L_inf)
    x0 = (lowers + uppers) / 2.0
    half_widths = (uppers - lowers) / 2.0
    epsilon = float(np.median(half_widths))  # robust to floating-point noise

    # Sanity: check box is approximately symmetric
    if np.max(np.abs(half_widths - epsilon)) > 1e-6 * max(epsilon, 1e-12):
        # Non-uniform box — still use median epsilon but warn
        pass

    # Parse output constraints to find true class
    # Pattern: (assert (or (and (<= Y_k Y_j) ...)))
    # The true class y is the one that must be >= all others
    true_class = _extract_true_class(text)
    num_outputs = len(output_vars) if output_vars else None

    return x0.astype(np.float32), epsilon, true_class, num_outputs, "inf"


def _extract_true_class(vnnlib_text: str) -> Optional[int]:
    """
    Extract the true class from VNNLIB output constraints.

    VNNLIB robustness specs encode the UNSAFE condition:
      (assert (or
        (and (<= Y_label Y_1))   ; label loses to class 1
        (and (<= Y_label Y_2))   ; label loses to class 2
        ...
      ))

    The true class appears on the LHS of all <= constraints (it's the
    one whose logit must stay above all others for robustness to hold).
    """
    # Pattern: (<= Y_label Y_k) — true class is on the LHS
    lhs_counts = {}
    for m in re.finditer(r'\(<=\s+Y_(\d+)\s+Y_(\d+)\s*\)', vnnlib_text):
        lhs = int(m.group(1))
        lhs_counts[lhs] = lhs_counts.get(lhs, 0) + 1

    if not lhs_counts:
        # Try >= pattern: (>= Y_k Y_label) — true class is on the RHS
        rhs_counts = {}
        for m in re.finditer(r'\(>=\s+Y_(\d+)\s+Y_(\d+)\s*\)', vnnlib_text):
            rhs = int(m.group(2))
            rhs_counts[rhs] = rhs_counts.get(rhs, 0) + 1
        if rhs_counts:
            return max(rhs_counts, key=rhs_counts.get)

    if lhs_counts:
        return max(lhs_counts, key=lhs_counts.get)

    return None


# --------------------------------------------------------------------------
# ONNX -> PyTorch loader
# --------------------------------------------------------------------------

def _parse_onnx_input_shape(onnx_path: str) -> tuple:
    """
    Parse the non-batch input shape from an ONNX model's shape proto.

    Returns: tuple of ints, e.g. (800,) for flat MLP or (1, 20, 40) for CNN.
    """
    import onnx

    onnx_model = onnx.load(onnx_path)
    inp = onnx_model.graph.input[0]
    shape_proto = inp.type.tensor_type.shape

    all_dims = []  # (value_or_None, is_dynamic)
    for d in shape_proto.dim:
        if d.dim_param:
            all_dims.append((None, True))
        else:
            all_dims.append((d.dim_value, False))

    if not all_dims:
        return ()
    elif all_dims[0][1]:
        # First dim is dynamic (batch) — rest are the input shape
        return tuple(v for v, _ in all_dims[1:])
    elif all_dims[0][0] in (0, 1) and len(all_dims) > 1:
        # First dim is static batch=1 — strip it
        return tuple(v for v, _ in all_dims[1:])
    else:
        return tuple(v for v, _ in all_dims)


class _OrtAutogradWrapper:
    """
    Wraps onnxruntime InferenceSession to behave like a torch callable
    with gradient support via torch.autograd.Function.

    Gradients are computed by finite differences, which is slow but
    universally applicable.  This is the fallback when onnx2pytorch
    cannot parse the model.
    """

    def __init__(self, onnx_path: str, device: str, input_shape: tuple):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.device = device
        self.input_shape = input_shape
        self._fd_eps = 1e-4
        # Detect whether the model supports dynamic batch
        ort_shape = self.session.get_inputs()[0].shape
        self._dynamic_batch = (
            len(ort_shape) > 0
            and (isinstance(ort_shape[0], str) or ort_shape[0] is None)
        )

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        return _OrtForward.apply(x, self)

    def _run_numpy(self, x_np: np.ndarray) -> np.ndarray:
        """Run the ONNX model on a numpy array, handling fixed-batch models."""
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, *self.input_shape) if self.input_shape else x_np.reshape(1, -1)
        elif x_np.ndim == 2 and self.input_shape and len(self.input_shape) > 1:
            x_np = x_np.reshape(-1, *self.input_shape)

        x_np = x_np.astype(np.float32)
        batch = x_np.shape[0]

        # If batch=1 or model supports dynamic batch, run directly
        if batch == 1 or self._dynamic_batch:
            feeds = {self.input_name: x_np}
            return self.session.run(None, feeds)[0]

        # Fixed-batch model: process one sample at a time
        results = []
        for i in range(batch):
            sample = x_np[i:i+1]  # keep batch dim = 1
            feeds = {self.input_name: sample}
            out = self.session.run(None, feeds)[0]
            results.append(out)
        return np.concatenate(results, axis=0)


class _OrtForward(object):
    """
    Custom autograd Function for ORT models using finite differences.

    Implemented as a manual torch.autograd.Function to provide gradients
    through the ONNX runtime session.  The backward pass uses central
    finite differences: df/dx_i ≈ (f(x + e_i h) - f(x - e_i h)) / 2h.

    Note: We avoid subclassing torch.autograd.Function directly because
    ORT sessions are not picklable (required by some autograd internals).
    Instead, we use a closure-based approach that still integrates with
    torch's autograd tape.
    """

    @staticmethod
    def apply(x, wrapper):
        import torch

        x_np = x.detach().cpu().numpy()
        out_np = wrapper._run_numpy(x_np)
        out_tensor = torch.from_numpy(out_np.astype(np.float32)).to(x.device)

        if not x.requires_grad:
            return out_tensor

        # Use a proper autograd.Function via a nested class that captures wrapper
        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x_in):
                ctx.save_for_backward(x_in)
                ctx._wrapper = wrapper
                # Reuse the already-computed output
                return out_tensor.clone()

            @staticmethod
            def backward(ctx, grad_output):
                x_saved, = ctx.saved_tensors
                w = ctx._wrapper
                return _OrtForward._finite_diff_vjp(
                    w, x_saved, grad_output
                )

        return _Fn.apply(x)

    @staticmethod
    def _finite_diff_vjp(wrapper, x, grad_output):
        """
        Compute the vector-Jacobian product grad_output^T @ J via
        central finite differences.

        J_{c,i} = df_c / dx_i ≈ (f_c(x + h e_i) - f_c(x - h e_i)) / 2h
        (VJP)_i = sum_c grad_output_c * J_{c,i}
        """
        import torch

        h = wrapper._fd_eps
        x_np = x.detach().cpu().numpy()
        g_np = grad_output.detach().cpu().numpy()

        batch = x_np.shape[0]
        d = int(np.prod(x_np.shape[1:]))
        x_flat = x_np.reshape(batch, d)
        grad_input = np.zeros_like(x_flat)

        for i in range(d):
            x_plus = x_flat.copy()
            x_plus[:, i] += h
            x_minus = x_flat.copy()
            x_minus[:, i] -= h
            out_plus = wrapper._run_numpy(x_plus)
            out_minus = wrapper._run_numpy(x_minus)
            # J_{:,i} shape (batch, C)
            jac_col = (out_plus - out_minus) / (2.0 * h)
            # VJP_i = sum over output dims
            g_flat = g_np.reshape(batch, -1)
            jac_flat = jac_col.reshape(batch, -1)
            grad_input[:, i] = (jac_flat * g_flat).sum(axis=1)

        result = torch.from_numpy(grad_input.astype(np.float32))
        result = result.view_as(x).to(x.device)
        return result