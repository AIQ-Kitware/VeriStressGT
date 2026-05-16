from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_margin_gradients(
    inst: "VerificationInstance",
    x_batch: "torch.Tensor",
    *,
    margin_fn: str = "overall",
) -> "torch.Tensor":
    """
    Compute gradients of the margin function at a batch of points.

    For piecewise-linear networks, the gradient at a non-differentiable
    point is a Clarke subgradient (any element of the subdifferential
    suffices; autograd returns one such element).

    Parameters
    ----------
    inst : VerificationInstance
    x_batch : (N, d) tensor, requires_grad will be set internally
    margin_fn : "overall" for g(x) = min_k g_k(x), or "worst_class"

    Returns
    -------
    grads : (N, d) tensor of gradients/subgradients
    """
    import torch

    x = x_batch.detach().clone().requires_grad_(True)
    if x.dim() == 1:
        x = x.unsqueeze(0)

    N = x.shape[0]
    d = inst.input_dim

    # Compute margin
    # We reshape for the network if needed
    x_input = x
    if inst.onnx_input_shape and x.shape[1:] != inst.onnx_input_shape:
        x_input = x.view(-1, *inst.onnx_input_shape)

    logits = inst.forward_fn(x_input)  # (N, C)
    y = inst.true_class
    fy = logits[:, y]  # (N,)

    mask = torch.ones(inst.num_classes, dtype=torch.bool, device=x.device)
    mask[y] = False
    fk = logits[:, mask]  # (N, C-1)

    margins = fy.unsqueeze(1) - fk  # (N, C-1)
    # g(x) = min over classes
    g, _ = margins.min(dim=1)  # (N,)

    # Backward pass — sum over batch to get per-example gradients
    # We need individual gradients, so loop or use vmap-like approach
    grads = torch.zeros(N, d, device=x.device, dtype=x.dtype)

    # For efficiency: since g = min_k g_k, the gradient is ∇g_k* where
    # k* = argmin_k g_k(x). We can compute this in one backward pass
    # by using the trick: backward on the sum of selected margins.
    worst_class_idx = margins.argmin(dim=1)  # (N,)
    selected_margins = margins[torch.arange(N, device=x.device), worst_class_idx]  # (N,)

    selected_margins.sum().backward()

    if x.grad is not None:
        grads = x.grad.detach().clone()
    else:
        # Fallback: finite differences
        grads = _finite_diff_margin_grad(inst, x_batch.detach())

    return grads


def _finite_diff_margin_grad(
    inst: "VerificationInstance",
    x_batch: "torch.Tensor",
    eps: float = 1e-4,
) -> "torch.Tensor":
    """Finite-difference gradient of the margin function."""
    import torch

    N, d = x_batch.shape[0], inst.input_dim
    x = x_batch.view(N, d)
    grads = torch.zeros_like(x)

    with torch.no_grad():
        for i in range(d):
            x_plus = x.clone()
            x_plus[:, i] += eps
            x_minus = x.clone()
            x_minus[:, i] -= eps
            g_plus = inst.margin(x_plus)
            g_minus = inst.margin(x_minus)
            grads[:, i] = (g_plus - g_minus) / (2 * eps)

    return grads


def dual_norm(v: "torch.Tensor", p: str) -> "torch.Tensor":
    """
    Compute the dual norm ||v||_{p,*} for each row of v.

    The dual of L_inf is L_1, the dual of L_2 is L_2,
    and the dual of L_1 is L_inf.

    Parameters
    ----------
    v : (N, d) tensor
    p : "inf", "2", or "1"

    Returns
    -------
    norms : (N,) tensor
    """
    if p == "inf":
        return v.abs().sum(dim=1)    # dual of L_inf is L_1
    elif p == "2":
        return v.norm(dim=1)         # L_2 is self-dual
    elif p == "1":
        return v.abs().max(dim=1).values  # dual of L_1 is L_inf
    else:
        raise ValueError(f"Unknown norm p={p}")


def lp_norm(v: "torch.Tensor", p: str) -> "torch.Tensor":
    """Compute ||v||_p for each row of v."""
    if p == "inf":
        return v.abs().max(dim=1).values
    elif p == "2":
        return v.norm(dim=1)
    elif p == "1":
        return v.abs().sum(dim=1)
    else:
        raise ValueError(f"Unknown norm p={p}")
