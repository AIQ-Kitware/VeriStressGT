from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12


# ---------------------------------------------------------------------------
# NA / warning helpers
# ---------------------------------------------------------------------------

def _set_na_reason(result: Dict[str, Any], key: str, reason: str, detail: Optional[str] = None) -> None:
    result[f"{key}__na_reason"] = reason
    if detail is not None:
        result[f"{key}__na_detail"] = str(detail)


def _set_metric(
    result: Dict[str, Any],
    key: str,
    value: Any,
    *,
    reason: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    result[key] = value
    if value is None and reason is not None:
        _set_na_reason(result, key, reason, detail)


def _classify_reason_from_message(msg: str) -> str:
    s = str(msg).lower()
    unsupported_terms = [
        "unsupported", "not supported", "could not load onnx", "unknown op",
        "shape mismatch", "could not identify", "no differentiable",
        "no gradient", "onnxruntime", "no forward",
    ]
    if any(t in s for t in unsupported_terms):
        return "unsupported"
    degenerate_terms = [
        "undefined", "near zero", "tiny", "constant", "fewer than",
        "insufficient", "no valid", "epsilon <= 0", "zero width",
    ]
    if any(t in s for t in degenerate_terms):
        return "math_degenerate"
    return "numerical_error"


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _finite(v: float, default: float = 0.0) -> float:
    try:
        vf = float(v)
        return vf if math.isfinite(vf) else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Sampling, margin, gradient, and geometry helpers
# ---------------------------------------------------------------------------

def _sample_uniform(inst: "VerificationInstance", n: int):
    import torch

    if n <= 0:
        return inst.x0.detach().unsqueeze(0)
    try:
        return inst.sample_uniform(n)
    except Exception:
        # Fallback only for l_inf boxes.
        if getattr(inst, "norm_p", "inf") != "inf":
            raise
        x0 = inst.x0.detach()
        eps = float(inst.epsilon)
        noise = (2.0 * torch.rand((n, *x0.shape), device=x0.device, dtype=x0.dtype) - 1.0) * eps
        return x0.unsqueeze(0) + noise


def _as_flat_batch(X):
    return X.reshape(X.shape[0], -1)


def _linf_random_corners(inst: "VerificationInstance", n: int):
    """Random vertices of the L_inf box."""
    import torch

    x0 = inst.x0.detach()
    eps = float(inst.epsilon)
    if n <= 0:
        return x0.detach().unsqueeze(0)[:0]
    signs = torch.empty((n, *x0.shape), device=x0.device, dtype=x0.dtype).bernoulli_(0.5)
    signs = signs.mul_(2.0).sub_(1.0)
    return x0.unsqueeze(0) + eps * signs


def _linf_random_faces(inst: "VerificationInstance", n: int, *, coords_per_sample: Optional[int] = None):
    """Samples with a small random subset of coordinates clamped to box faces."""
    import math as _math
    import torch

    x0 = inst.x0.detach()
    eps = float(inst.epsilon)
    if n <= 0:
        return x0.detach().unsqueeze(0)[:0]

    X = _sample_uniform(inst, n)
    flat = _as_flat_batch(X - x0.unsqueeze(0))
    d = flat.shape[1]
    if d == 0:
        return X

    if coords_per_sample is None:
        # Clamp enough coordinates to hit faces, but not so many that every
        # sample becomes a full corner in high dimension.
        coords_per_sample = max(1, min(d, int(round(_math.sqrt(d)))))
    k = int(max(1, min(d, coords_per_sample)))

    idx = torch.randint(0, d, (n, k), device=flat.device)
    signs = torch.empty((n, k), device=flat.device, dtype=flat.dtype).bernoulli_(0.5)
    signs = signs.mul_(2.0).sub_(1.0)
    rows = torch.arange(n, device=flat.device).unsqueeze(1).expand_as(idx)
    flat[rows, idx] = eps * signs
    return x0.unsqueeze(0) + flat.reshape_as(X)


def _linf_gradient_face_samples(inst: "VerificationInstance", g0_flat, n: int):
    """Faces biased toward the local first-order margin-decreasing direction."""
    import math as _math
    import torch

    x0 = inst.x0.detach()
    eps = float(inst.epsilon)
    if n <= 0:
        return x0.detach().unsqueeze(0)[:0]

    X = _sample_uniform(inst, n)
    flat = _as_flat_batch(X - x0.unsqueeze(0))
    d = flat.shape[1]
    if d == 0:
        return X

    g = g0_flat.detach().reshape(-1)
    if g.numel() != d or torch.max(torch.abs(g)).item() <= 1e-15:
        return X

    # Clamp the most sensitive coordinates in the sign direction that minimizes
    # the first-order margin: delta = -eps * sign(grad m).
    k = max(1, min(d, int(round(_math.sqrt(d)))))
    top = torch.topk(torch.abs(g), k=k, largest=True).indices
    direction = -torch.sign(g[top])
    direction = torch.where(direction == 0, torch.ones_like(direction), direction)
    flat[:, top] = eps * direction.reshape(1, -1).to(flat.dtype)

    # Also include a few full-gradient-sign variants via random scaling/jitter.
    if n >= 2:
        full = -torch.sign(g)
        full = torch.where(full == 0, torch.ones_like(full), full)
        scales = torch.linspace(0.25, 1.0, steps=min(n, 4), device=flat.device, dtype=flat.dtype)
        for r, s in enumerate(scales):
            flat[r, :] = eps * s * full.to(flat.dtype)
    return x0.unsqueeze(0) + flat.reshape_as(X)


def _pgd_min_margin_samples(
    inst: "VerificationInstance",
    n_restarts: int,
    *,
    n_steps: int = 20,
    step_size: Optional[float] = None,
):
    """Best-effort projected gradient search for small-margin points."""
    import torch

    x0 = inst.x0.detach()
    eps = float(inst.epsilon)
    if n_restarts <= 0:
        return x0.detach().unsqueeze(0)[:0]
    if eps <= 0:
        return x0.detach().unsqueeze(0).expand(n_restarts, *x0.shape)

    X = _sample_uniform(inst, n_restarts).detach()
    alpha = float(step_size if step_size is not None else max(eps / 4.0, eps / max(n_steps, 1)))

    for _ in range(max(1, int(n_steps))):
        try:
            xb = X.detach().clone().requires_grad_(True)
            m = inst.margin(xb).reshape(-1)
            if not m.requires_grad:
                raise RuntimeError("margin is not differentiable")
            g, = torch.autograd.grad(m.sum(), xb, create_graph=False, retain_graph=False)
            if getattr(inst, "norm_p", "inf") == "inf":
                X = xb.detach() - alpha * torch.sign(g.detach())
            elif getattr(inst, "norm_p", "inf") == "2":
                gf = g.detach().reshape(g.shape[0], -1)
                denom = gf.norm(p=2, dim=1).clamp(min=_EPS)
                step = (alpha * gf / denom.reshape(-1, 1)).reshape_as(g)
                X = xb.detach() - step
            else:
                X = xb.detach() - alpha * torch.sign(g.detach())
            X = _project_to_ball(inst, X.detach())
        except Exception:
            break
    return X.detach()


def _sample_stress_points(
    inst: "VerificationInstance",
    n: int,
    *,
    n_pgd_steps: int = 20,
):
    """Mixed sampler for boundary-proximity margin statistics.

    The returned set deliberately over-represents likely hard regions of an
    L_inf box: uniform interior points, random faces, random corners,
    local-gradient faces/sign directions, and a small PGD search.  For non-L_inf
    norms it falls back to uniform plus PGD/projection.
    """
    import torch

    x0 = inst.x0.detach()
    if n <= 0:
        return x0.unsqueeze(0)

    chunks = []
    n_uniform = max(1, int(round(0.25 * n)))
    n_faces = max(0, int(round(0.25 * n)))
    n_corners = max(0, int(round(0.15 * n)))
    n_grad = max(0, int(round(0.20 * n)))
    n_pgd = max(0, n - n_uniform - n_faces - n_corners - n_grad)

    chunks.append(_sample_uniform(inst, n_uniform))

    norm_p = getattr(inst, "norm_p", "inf")
    if norm_p == "inf":
        if n_faces:
            chunks.append(_linf_random_faces(inst, n_faces))
        if n_corners:
            chunks.append(_linf_random_corners(inst, n_corners))
    else:
        if n_faces + n_corners:
            chunks.append(_sample_uniform(inst, n_faces + n_corners))

    g0 = None
    try:
        g0 = _compute_margin_gradients(inst, x0.unsqueeze(0))[0]
    except Exception:
        g0 = None

    if g0 is not None and norm_p == "inf" and n_grad:
        try:
            chunks.append(_linf_gradient_face_samples(inst, g0, n_grad))
        except Exception:
            chunks.append(_sample_uniform(inst, n_grad))
    elif n_grad:
        chunks.append(_sample_uniform(inst, n_grad))

    if n_pgd:
        try:
            chunks.append(_pgd_min_margin_samples(inst, n_pgd, n_steps=n_pgd_steps))
        except Exception:
            chunks.append(_sample_uniform(inst, n_pgd))

    # Always include the exact local first-order worst-case point when possible.
    if g0 is not None and norm_p == "inf":
        g = g0.detach().reshape(-1)
        if g.numel() == int(inst.input_dim):
            sign = -torch.sign(g)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            x_adv_lin = x0 + float(inst.epsilon) * sign.reshape_as(x0)
            chunks.append(x_adv_lin.unsqueeze(0))

    X = torch.cat([c.detach() for c in chunks if c is not None and c.numel() > 0], dim=0)
    return _project_to_ball(inst, X)


def _margin_values(inst: "VerificationInstance", X, batch_size: int = 256):
    import torch

    vals = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            m = inst.margin(X[i:i + batch_size])
            vals.append(m.reshape(-1).detach())
    return torch.cat(vals, dim=0)


def _compute_margin_gradients(inst: "VerificationInstance", X, batch_size: int = 128):
    """Return flattened ∇_x margin(x) for every x in X.

    Prefer the repository's gradients helper if present.  Fall back to direct
    autograd through inst.margin.  This intentionally requires only first-order
    autodiff.
    """
    import torch

    try:
        from .gradients import compute_margin_gradients  # type: ignore
        G = compute_margin_gradients(inst, X)
        return G.reshape(G.shape[0], -1).detach()
    except Exception:
        pass

    grads = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].detach().clone().requires_grad_(True)
        m = inst.margin(xb).reshape(-1)
        # Sum is valid because batch elements are independent.
        s = m.sum()
        if not s.requires_grad:
            raise RuntimeError("inst.margin(x) is not differentiable; no gradient available")
        g, = torch.autograd.grad(s, xb, create_graph=False, retain_graph=False)
        grads.append(g.reshape(g.shape[0], -1).detach())
    return torch.cat(grads, dim=0)


def _lp_distance(dx, norm_p: str):
    import torch

    flat = dx.reshape(dx.shape[0], -1)
    if norm_p == "inf":
        return flat.abs().amax(dim=1)
    if norm_p == "2":
        return flat.norm(p=2, dim=1)
    if norm_p == "1":
        return flat.norm(p=1, dim=1)
    # Conservative fallback.
    return flat.norm(p=2, dim=1)


def _project_to_ball(inst: "VerificationInstance", X):
    """Project X back to the perturbation set around x0."""
    import torch

    x0 = inst.x0.detach()
    eps = float(inst.epsilon)
    if eps <= 0:
        return x0.unsqueeze(0).expand_as(X)
    if inst.norm_p == "inf":
        return torch.max(torch.min(X, (x0 + eps).unsqueeze(0)), (x0 - eps).unsqueeze(0))
    if inst.norm_p == "2":
        delta = (X - x0.unsqueeze(0)).reshape(X.shape[0], -1)
        norm = delta.norm(p=2, dim=1, keepdim=True).clamp(min=_EPS)
        scale = torch.minimum(torch.ones_like(norm), torch.full_like(norm, eps) / norm)
        return x0.unsqueeze(0) + (delta * scale).reshape_as(X)
    return X


def _random_directions_like(inst: "VerificationInstance", n: int):
    import torch

    x0 = inst.x0.detach()
    if inst.norm_p == "inf":
        U = torch.empty((n, *x0.shape), device=x0.device, dtype=x0.dtype).uniform_(-1.0, 1.0)
        # Put directions on the l_inf sphere-ish so h has a consistent meaning.
        denom = U.reshape(n, -1).abs().amax(dim=1).clamp(min=_EPS)
        return U / denom.reshape(-1, *([1] * x0.ndim))
    U = torch.randn((n, *x0.shape), device=x0.device, dtype=x0.dtype)
    denom = U.reshape(n, -1).norm(p=2, dim=1).clamp(min=_EPS)
    return U / denom.reshape(-1, *([1] * x0.ndim))


def _effective_rank_from_eigs(eigs: np.ndarray) -> float:
    eigs = np.asarray(eigs, dtype=np.float64)
    eigs = eigs[np.isfinite(eigs) & (eigs > 1e-15)]
    if eigs.size == 0:
        return 0.0
    p = eigs / eigs.sum()
    return float(np.exp(-(p * np.log(np.clip(p, 1e-15, 1.0))).sum()))


def _rand_pair_indices(n: int, n_pairs: int, device=None):
    import torch

    n_eff = max(0, int(n))
    if n_eff < 2 or n_pairs <= 0:
        return None, None
    i = torch.randint(0, n_eff, (n_pairs,), device=device)
    j = torch.randint(0, n_eff, (n_pairs,), device=device)
    same = i == j
    if same.any():
        j[same] = (j[same] + 1) % n_eff
    return i, j


def _quantile_np(x: np.ndarray, q: float) -> Optional[float]:
    if x.size == 0:
        return None
    return float(np.quantile(x, q))


# ---------------------------------------------------------------------------
# Main generic estimator: components 1-8, 16-30 except IBP-specific items.
# ---------------------------------------------------------------------------

def estimate_generic_components(
    inst: "VerificationInstance",
    *,
    n_samples: int = 200,
    n_pairs: int = 450,
    n_second_diff: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute architecture-agnostic margin/gradient/nonlinearity components.

    Primary outputs used by the compact profile:
      margin_sample_min, margin_to_variation_ratio, effective_grad_dim_mean,
      ibp_relative_gap, unstable_frac, A_tau_local_log, A_tau_effective_log,
      relative_linearization_error, grad_sensitivity_concentration,
      first_order_margin_ratio, pgd_improvement_ratio, low_margin_basin_frac,
      best_point_boundary_frac, first_order_adv_error_ratio,
      stress_min_grad_alignment.

    Some auxiliary columns are still emitted for diagnostics/backward compatibility.
    """
    import torch

    result: Dict[str, Any] = {"warnings": []}
    eps = float(inst.epsilon)
    d = int(inst.input_dim)
    n_second_diff = int(n_second_diff or min(n_samples, n_pairs, 256))

    # ------------------------------------------------------------------
    # Margin samples.
    # ------------------------------------------------------------------
    # Use a mixed stress sampler for boundary-proximity metrics.  This keeps the
    # name margin_sample_min backward-compatible, but the semantics are now:
    #   min over {uniform, random faces, random corners, gradient faces, PGD}.
    try:
        X_stress = _sample_stress_points(inst, n_samples, n_pgd_steps=min(30, max(8, n_samples // 10)))
        X = torch.cat([inst.x0.detach().unsqueeze(0), X_stress], dim=0)
        m_t = _margin_values(inst, X)
        m = m_t.detach().cpu().numpy().astype(np.float64).reshape(-1)
    except Exception as e:
        msg = f"stress margin sampling failed: {e}"
        result["warnings"].append(msg)
        for key in [
            "margin_nominal", "margin_sample_min", "margin_sample_q01",
            "margin_std", "margin_range", "margin_to_variation_ratio",
            "relative_sample_min_margin", "pgd_improvement_ratio",
            "low_margin_basin_frac", "best_point_boundary_frac",
        ]:
            _set_metric(result, key, None, reason=_classify_reason_from_message(msg), detail=msg)
        return result

    margin_nominal = float(getattr(inst, "margin_at_x0", m[0]))
    margin_sample_min = float(np.min(m))
    margin_q01 = _quantile_np(m, 0.01)
    margin_q05 = _quantile_np(m, 0.05)
    margin_std = float(np.std(m))
    margin_range = float(np.max(m) - np.min(m))
    margin_iqr = float(np.quantile(m, 0.75) - np.quantile(m, 0.25))

    result.update({
        "margin_nominal": margin_nominal,
        "margin_sample_min": margin_sample_min,
        "margin_sample_q01": margin_q01,
        "margin_sample_q05": margin_q05,
        "margin_std": margin_std,
        "margin_range": margin_range,
        "margin_iqr": margin_iqr,
        "n_margin_samples": int(m.size),
        "margin_sampler": "mixed_stress_uniform_faces_corners_gradient_pgd",
    })

    # Use the low quantile rather than the raw minimum in the numerator so that
    # one numerical outlier does not dominate the scale ratio.
    result["margin_to_variation_ratio"] = float(margin_q01 / (margin_std + _EPS)) if margin_q01 is not None else None
    result["relative_sample_min_margin"] = float(margin_sample_min / (abs(margin_nominal) + _EPS))

    # ------------------------------------------------------------------
    # New optimization-geometry metrics that are deliberately not just new
    # names for previous gradient/IBP/A_tau quantities.
    # ------------------------------------------------------------------
    try:
        # 1) PGD improvement over ordinary uniform sampling.  This asks whether
        # the hard part of the box is hidden from naive random probes.
        n_probe = max(16, min(max(1, n_samples), max(32, n_samples // 3)))
        X_uniform_probe = _sample_uniform(inst, n_probe)
        m_uniform_probe = _margin_values(inst, X_uniform_probe).detach().cpu().numpy().astype(np.float64).reshape(-1)
        uniform_min = float(np.min(m_uniform_probe))

        n_pgd_probe = max(4, min(32, n_probe // 2))
        X_pgd_probe = _pgd_min_margin_samples(
            inst, n_pgd_probe, n_steps=min(40, max(8, n_samples // 8))
        )
        m_pgd_probe = _margin_values(inst, X_pgd_probe).detach().cpu().numpy().astype(np.float64).reshape(-1)
        pgd_min = float(np.min(m_pgd_probe))

        result["uniform_margin_min"] = uniform_min
        result["pgd_margin_min"] = pgd_min
        result["pgd_improvement_ratio"] = float(max(0.0, uniform_min - pgd_min) / (abs(uniform_min) + _EPS))
    except Exception as e:
        msg = f"PGD-vs-uniform improvement metric failed: {e}"
        result["warnings"].append(msg)
        for key in ["uniform_margin_min", "pgd_margin_min", "pgd_improvement_ratio"]:
            _set_metric(result, key, None, reason=_classify_reason_from_message(msg), detail=msg)

    try:
        # 2) Low-margin basin width.  The scale uses the larger of the margin
        # standard deviation and the median-to-min gap so that the threshold is
        # meaningful for both narrow and broad margin distributions.
        median_m = float(np.median(m))
        basin_scale = max(float(margin_std), float(median_m - margin_sample_min), _EPS)
        basin_threshold = float(margin_sample_min + 0.05 * basin_scale)
        result["low_margin_basin_threshold"] = basin_threshold
        result["low_margin_basin_frac"] = float(np.mean(m <= basin_threshold))
    except Exception as e:
        msg = f"low-margin basin metric failed: {e}"
        result["warnings"].append(msg)
        for key in ["low_margin_basin_threshold", "low_margin_basin_frac"]:
            _set_metric(result, key, None, reason=_classify_reason_from_message(msg), detail=msg)

    try:
        # 3) Boundary/corner-likeness of the best observed point.  For L_inf
        # specifications this distinguishes a corner/face minimum from an
        # interior nonlinear minimum.
        best_idx = int(np.argmin(m))
        if eps > 0 and getattr(inst, "norm_p", "inf") == "inf":
            delta = (X[best_idx] - inst.x0.detach()).detach().reshape(-1)
            rel = (delta.abs() / max(eps, _EPS)).detach().cpu().numpy().astype(np.float64)
            result["best_point_boundary_frac"] = float(np.mean(rel >= 0.99))
            result["stress_min_linf_radius"] = float(np.max(rel))
        else:
            _set_metric(result, "best_point_boundary_frac", None, reason="math_degenerate", detail="defined for nonzero L_inf boxes")
            _set_metric(result, "stress_min_linf_radius", None, reason="math_degenerate", detail="defined for nonzero L_inf boxes")
    except Exception as e:
        msg = f"best-point boundary fraction failed: {e}"
        result["warnings"].append(msg)
        for key in ["best_point_boundary_frac", "stress_min_linf_radius"]:
            _set_metric(result, key, None, reason=_classify_reason_from_message(msg), detail=msg)

    # Raw/effective input volume.
    if eps > 0:
        log_width = math.log(max(2.0 * eps, _EPS))
        result["input_box_log_volume"] = float(d * log_width)
    else:
        result["input_box_log_volume"] = float("-inf")
        result["warnings"].append("epsilon <= 0; input_box_log_volume = -inf")

    # ------------------------------------------------------------------
    # Gradients.
    # ------------------------------------------------------------------
    grads = None
    try:
        grads_t = _compute_margin_gradients(inst, X)
        grads = grads_t.detach().cpu().numpy().astype(np.float64)
        abs_g = np.abs(grads)
        l1 = abs_g.sum(axis=1)
        l2 = np.linalg.norm(grads, ord=2, axis=1)
        linf = abs_g.max(axis=1)

        grad_l1_x0 = float(l1[0])
        grad_l1_mean = float(np.mean(l1))
        grad_l1_p95 = float(np.quantile(l1, 0.95))
        grad_l2_mean = float(np.mean(l2))
        grad_linf_mean = float(np.mean(linf))
        grad_linf_p95 = float(np.quantile(linf, 0.95))

        result.update({
            "grad_l1_mean": grad_l1_mean,
            "grad_l1_p95": grad_l1_p95,
            "grad_l2_mean": grad_l2_mean,
            "grad_linf_mean": grad_linf_mean,
            "grad_linf_p95": grad_linf_p95,
            "grad_l1_x0": grad_l1_x0,
            "n_gradient_samples": int(grads.shape[0]),
        })

        result["first_order_margin_ratio"] = float(margin_nominal / (eps * grad_l1_x0 + _EPS))
        result["sampled_first_order_margin_ratio"] = float((margin_q01 or 0.0) / (eps * grad_l1_p95 + _EPS))

        eff_dim = (l1 ** 2) / (l2 ** 2 + _EPS)
        result["effective_grad_dim_mean"] = float(np.mean(eff_dim))
        result["effective_grad_dim_p95"] = float(np.quantile(eff_dim, 0.95))
        result["grad_sensitivity_concentration"] = float(np.mean(linf / (l1 + _EPS)))
        result["effective_box_log_volume"] = float(result["effective_grad_dim_mean"] * math.log(max(2.0 * eps, _EPS))) if eps > 0 else float("-inf")

        g_mean = grads.mean(axis=0, keepdims=True)
        centered = grads - g_mean
        row_norms = np.linalg.norm(centered, ord=2, axis=1)
        result["grad_variation_l2"] = float(row_norms.mean())
        result["grad_variation_l2_p95"] = float(np.quantile(row_norms, 0.95))

        # Directional dispersion = 1 - mean cosine similarity over random pairs.
        i, j = _rand_pair_indices(grads.shape[0], min(n_pairs, 5000))
        if i is not None:
            i_np = i.cpu().numpy()
            j_np = j.cpu().numpy()
            gi = grads[i_np]
            gj = grads[j_np]
            denom = np.linalg.norm(gi, axis=1) * np.linalg.norm(gj, axis=1)
            valid = denom > 1e-15
            if valid.any():
                cos = (gi[valid] * gj[valid]).sum(axis=1) / denom[valid]
                result["grad_direction_dispersion"] = float(1.0 - np.mean(cos))
                result["grad_abs_direction_dispersion"] = float(1.0 - np.mean(np.abs(cos)))
            else:
                result["grad_direction_dispersion"] = 0.0
                result["grad_abs_direction_dispersion"] = 0.0
        else:
            result["grad_direction_dispersion"] = 0.0
            result["grad_abs_direction_dispersion"] = 0.0

        # Effective rank of gradient covariance.  Use SVD in sample space.
        if grads.shape[0] >= 3 and np.linalg.norm(centered) > 1e-15:
            s = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
            eigs = (s ** 2) / max(grads.shape[0] - 1, 1)
            result["grad_cov_effective_rank"] = _effective_rank_from_eigs(eigs)
            result["grad_cov_trace"] = float(np.sum(eigs))
        else:
            result["grad_cov_effective_rank"] = 0.0
            result["grad_cov_trace"] = 0.0

        # Linearization error around x0.
        x_flat = X.detach().reshape(X.shape[0], -1).cpu().numpy().astype(np.float64)
        x0_flat = x_flat[0]
        g0 = grads[0]
        lin_pred = margin_nominal + (x_flat - x0_flat) @ g0
        lin_err = np.abs(m - lin_pred)
        result["linearization_error_mean"] = float(np.mean(lin_err))
        result["linearization_error_p95"] = float(np.quantile(lin_err, 0.95))
        result["linearization_error_max"] = float(np.max(lin_err))
        result["relative_linearization_error"] = float(result["linearization_error_p95"] / (abs(margin_q01 or margin_sample_min) + _EPS))

        # 4) Targeted first-order adversarial-direction error.  This is a more
        # optimization-specific Hessian/curvature proxy than the average
        # linearization error: it evaluates the first-order worst-case point.
        try:
            x0_t = inst.x0.detach()
            if eps <= 0:
                raise RuntimeError("epsilon <= 0")
            if getattr(inst, "norm_p", "inf") == "inf":
                g0_t = grads_t[0].detach().reshape_as(x0_t)
                adv_dir = -torch.sign(g0_t)
                adv_dir = torch.where(adv_dir == 0, torch.ones_like(adv_dir), adv_dir)
                x_fo = _project_to_ball(inst, x0_t.unsqueeze(0) + eps * adv_dir.unsqueeze(0))
                pred = margin_nominal - eps * grad_l1_x0
            elif getattr(inst, "norm_p", "inf") == "2":
                g0_t = grads_t[0].detach().reshape_as(x0_t)
                g0_norm = torch.linalg.norm(g0_t.reshape(-1), ord=2).clamp(min=_EPS)
                adv_dir = -g0_t / g0_norm
                x_fo = _project_to_ball(inst, x0_t.unsqueeze(0) + eps * adv_dir.unsqueeze(0))
                pred = margin_nominal - eps * float(g0_norm.item())
            else:
                raise RuntimeError("first-order adversarial point implemented for L_inf/L2 only")
            m_fo = float(_margin_values(inst, x_fo).detach().cpu().numpy().reshape(-1)[0])
            result["first_order_adv_margin"] = m_fo
            result["first_order_adv_predicted_margin"] = float(pred)
            result["first_order_adv_error_ratio"] = float(abs(m_fo - pred) / (abs(margin_q01 or margin_sample_min) + _EPS))
        except Exception as e_inner:
            msg_inner = f"first-order adversarial-direction error failed: {e_inner}"
            result["warnings"].append(msg_inner)
            for key in ["first_order_adv_margin", "first_order_adv_predicted_margin", "first_order_adv_error_ratio"]:
                _set_metric(result, key, None, reason=_classify_reason_from_message(msg_inner), detail=msg_inner)

        # 5) Alignment between the best observed perturbation and the local
        # first-order adversarial direction at x0.
        try:
            best_idx = int(np.argmin(m))
            delta_best = x_flat[best_idx] - x0_flat
            if getattr(inst, "norm_p", "inf") == "inf":
                denom = eps * np.sum(np.abs(g0)) + _EPS
                align = -float(np.dot(g0, delta_best)) / denom
            else:
                denom = float(np.linalg.norm(g0, ord=2) * np.linalg.norm(delta_best, ord=2) + _EPS)
                align = -float(np.dot(g0, delta_best)) / denom
            result["stress_min_grad_alignment"] = float(np.clip(align, -1.0, 1.0))
        except Exception as e_inner:
            msg_inner = f"stress-min gradient-alignment metric failed: {e_inner}"
            result["warnings"].append(msg_inner)
            _set_metric(result, "stress_min_grad_alignment", None, reason=_classify_reason_from_message(msg_inner), detail=msg_inner)
    except Exception as e:
        msg = f"gradient-based components failed: {e}"
        result["warnings"].append(msg)
        for key in [
            "grad_l1_mean", "grad_l1_p95", "grad_l2_mean", "grad_linf_mean",
            "first_order_margin_ratio", "sampled_first_order_margin_ratio",
            "effective_grad_dim_mean", "grad_sensitivity_concentration",
            "effective_box_log_volume", "grad_variation_l2",
            "grad_direction_dispersion", "grad_cov_effective_rank",
            "linearization_error_mean", "relative_linearization_error",
            "first_order_adv_error_ratio", "stress_min_grad_alignment",
        ]:
            _set_metric(result, key, None, reason=_classify_reason_from_message(msg), detail=msg)

    # ------------------------------------------------------------------
    # Empirical Lipschitz on random pairs.
    # ------------------------------------------------------------------
    try:
        i, j = _rand_pair_indices(X.shape[0], min(n_pairs, max(1, X.shape[0] * (X.shape[0] - 1))), device=X.device)
        if i is None:
            raise RuntimeError("fewer than 2 samples for empirical Lipschitz")
        dist = _lp_distance(X[i] - X[j], getattr(inst, "norm_p", "inf")).detach().cpu().numpy()
        dm = np.abs(m[i.detach().cpu().numpy()] - m[j.detach().cpu().numpy()])
        valid = dist > 1e-12
        ratios = dm[valid] / dist[valid]
        if ratios.size == 0:
            raise RuntimeError("no valid nonzero-distance sample pairs")
        result["lip_empirical_max"] = float(np.max(ratios))
        result["lip_empirical_p95"] = float(np.quantile(ratios, 0.95))
        result["lip_empirical_mean"] = float(np.mean(ratios))
    except Exception as e:
        msg = f"empirical Lipschitz failed: {e}"
        result["warnings"].append(msg)
        for key in ["lip_empirical_max", "lip_empirical_p95"]:
            _set_metric(result, key, None, reason=_classify_reason_from_message(msg), detail=msg)

    # ------------------------------------------------------------------
    # Chord nonlinearity: midpoint deviation from affine interpolation.
    # ------------------------------------------------------------------
    try:
        i, j = _rand_pair_indices(X.shape[0], min(n_pairs, 512), device=X.device)
        if i is None:
            raise RuntimeError("fewer than 2 samples for chord nonlinearity")
        Xm = 0.5 * (X[i] + X[j])
        Xm = _project_to_ball(inst, Xm)
        mm = _margin_values(inst, Xm).detach().cpu().numpy().astype(np.float64)
        mi = m[i.detach().cpu().numpy()]
        mj = m[j.detach().cpu().numpy()]
        chord = np.abs(mm - 0.5 * (mi + mj))
        result["chord_nonlinearity_mean"] = float(np.mean(chord))
        result["chord_nonlinearity_p95"] = float(np.quantile(chord, 0.95))
        result["chord_nonlinearity_max"] = float(np.max(chord))
    except Exception as e:
        msg = f"chord nonlinearity failed: {e}"
        result["warnings"].append(msg)
        for key in ["chord_nonlinearity_mean", "chord_nonlinearity_p95"]:
            _set_metric(result, key, None, reason=_classify_reason_from_message(msg), detail=msg)

    # ------------------------------------------------------------------
    # Directional second differences around x0.
    # ------------------------------------------------------------------
    try:
        if eps <= 0:
            raise RuntimeError("epsilon <= 0")
        U = _random_directions_like(inst, n_second_diff)
        h = eps
        Xp = _project_to_ball(inst, inst.x0.detach().unsqueeze(0) + h * U)
        Xm = _project_to_ball(inst, inst.x0.detach().unsqueeze(0) - h * U)
        mp = _margin_values(inst, Xp).detach().cpu().numpy().astype(np.float64)
        mn = _margin_values(inst, Xm).detach().cpu().numpy().astype(np.float64)
        sec = np.abs(mp - 2.0 * margin_nominal + mn)
        result["second_diff_mean"] = float(np.mean(sec))
        result["second_diff_p95"] = float(np.quantile(sec, 0.95))
        result["second_diff_max"] = float(np.max(sec))
    except Exception as e:
        msg = f"second-difference curvature failed: {e}"
        result["warnings"].append(msg)
        for key in ["second_diff_mean", "second_diff_p95"]:
            _set_metric(result, key, None, reason=_classify_reason_from_message(msg), detail=msg)

    # ------------------------------------------------------------------
    # Spec-direction conditioning.  Kept separate because it needs logits.
    # ------------------------------------------------------------------
    try:
        cond = estimate_spec_direction_conditioning(inst, n_points=min(16, max(1, n_samples // 8)), verbose=False)
        result.update({k: v for k, v in cond.items() if k != "warnings"})
        result["warnings"].extend(cond.get("warnings", []))
    except Exception as e:
        msg = f"spec direction conditioning failed: {e}"
        _set_metric(result, "spec_direction_conditioning", None, reason=_classify_reason_from_message(msg), detail=msg)
        result["warnings"].append(msg)

    if verbose:
        print(f"    margin_sample_q01             = {result.get('margin_sample_q01'):+.6f}")
        print(f"    margin_std                    = {result.get('margin_std'):.6f}")
        print(f"    sampled_first_order_ratio     = {_safe_float(result.get('sampled_first_order_margin_ratio'))}")
        print(f"    relative_linearization_error  = {_safe_float(result.get('relative_linearization_error'))}")
        print(f"    pgd_improvement_ratio         = {_safe_float(result.get('pgd_improvement_ratio'))}")
        print(f"    low_margin_basin_frac         = {_safe_float(result.get('low_margin_basin_frac'))}")
        print(f"    best_point_boundary_frac      = {_safe_float(result.get('best_point_boundary_frac'))}")
        print(f"    first_order_adv_error_ratio   = {_safe_float(result.get('first_order_adv_error_ratio'))}")
        print(f"    stress_min_grad_alignment     = {_safe_float(result.get('stress_min_grad_alignment'))}")

    return result


# ---------------------------------------------------------------------------
# Spec-output conditioning
# ---------------------------------------------------------------------------

def _forward_logits_torch(inst: "VerificationInstance", X):
    """Best-effort differentiable forward returning logits for X."""
    fwd = getattr(inst, "forward_fn", None) or getattr(inst, "forward", None)
    if fwd is not None:
        return fwd(X)
    model = getattr(inst, "model", None)
    if model is not None:
        return model(X)
    raise RuntimeError("no differentiable torch forward/model found on instance")


def _spec_direction_from_logits(inst: "VerificationInstance", logits) -> Any:
    import torch

    flat = logits.reshape(logits.shape[0], -1)
    k = flat.shape[1]
    c = torch.zeros((flat.shape[0], k), device=flat.device, dtype=flat.dtype)
    if k == 1:
        c[:, 0] = 1.0
        return c
    true_c = int(inst.true_class)
    with torch.no_grad():
        tmp = flat.detach().clone()
        if 0 <= true_c < k:
            tmp[:, true_c] = -float("inf")
        comp = torch.argmax(tmp, dim=1)
    c[:, true_c] = 1.0
    c[torch.arange(flat.shape[0], device=flat.device), comp] = -1.0
    return c


def estimate_spec_direction_conditioning(
    inst: "VerificationInstance",
    *,
    n_points: int = 8,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Estimate ||J_f||_F / ||J_f^T c||_2.

    This measures how much output sensitivity exists in directions orthogonal to
    the verification spec.  It is optional/NA when no differentiable logits
    forward is available.  To keep the cost predictable, it only computes this
    exactly for outputs with <= 64 scalar logits.
    """
    import torch

    result: Dict[str, Any] = {"warnings": []}
    try:
        X = _sample_uniform(inst, max(1, n_points))
        vals = []
        for x in X:
            xb = x.detach().clone().unsqueeze(0).requires_grad_(True)
            logits = _forward_logits_torch(inst, xb).reshape(1, -1)
            k = int(logits.shape[1])
            if k > 64:
                raise RuntimeError(f"output dimension {k} too large for exact output Jacobian")
            c = _spec_direction_from_logits(inst, logits)[0]
            rows = []
            for j in range(k):
                gj, = torch.autograd.grad(logits[0, j], xb, retain_graph=True, create_graph=False)
                rows.append(gj.reshape(-1))
            J = torch.stack(rows, dim=0)
            J_frob = float(torch.linalg.norm(J).item())
            spec_grad = (c.reshape(1, -1) @ J).reshape(-1)
            spec_norm = float(torch.linalg.norm(spec_grad).item())
            vals.append(J_frob / (spec_norm + _EPS))
        result["spec_direction_conditioning"] = float(np.mean(vals))
        result["spec_direction_conditioning_p95"] = float(np.quantile(vals, 0.95))
        result["n_spec_conditioning_points"] = int(len(vals))
        if verbose:
            print(f"    spec_direction_conditioning = {result['spec_direction_conditioning']:.4f}")
        return result
    except Exception as e:
        msg = f"spec_direction_conditioning unavailable: {e}"
        _set_metric(result, "spec_direction_conditioning", None, reason=_classify_reason_from_message(msg), detail=msg)
        result["warnings"].append(msg)
        return result

def estimate_local_region_count(
    inst: "VerificationInstance",
    *,
    n_samples: int = 2048,
    projection_dim: int = 10,
    quantize_decimals: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Estimate saturation-resistant A_tau proxies from local gradient fingerprints.

    Outputs:
      - A_tau_effective_log   : Shannon log-effective support over fingerprints
      - A_tau_collision_log   : Renyi-2 / collision log-effective support
      - A_tau_growth_slope    : slope of log distinct count vs log sample size
      - A_tau_singleton_frac  : fraction of distinct fingerprints seen exactly once
      - A_tau_unseen_mass_gt  : Good-Turing unseen-mass proxy = f1 / n
      - A_tau_local_log_lower : raw sampled log-distinct-count lower bound (aux only)

    The old A_tau_local_log is retained as an alias to A_tau_local_log_lower for
    backward compatibility, but the intended replacements are the three quantities
    above.
    """
    import torch
    from collections import Counter

    result: Dict[str, Any] = {"warnings": []}

    try:
        X = _sample_uniform(inst, n_samples)
        grads = _compute_margin_gradients(inst, X).detach().cpu().numpy().astype(np.float64)
        if grads.ndim != 2 or grads.shape[0] == 0:
            raise RuntimeError("no gradients collected")

        n, d = grads.shape

        # Normalize gradients so fingerprinting reflects direction/region structure
        # more than raw scale.
        norms = np.linalg.norm(grads, axis=1, keepdims=True)
        G = grads / np.maximum(norms, 1e-12)

        if d > projection_dim:
            rng = np.random.default_rng(1)
            R = rng.normal(size=(d, projection_dim)).astype(np.float64) / math.sqrt(projection_dim)
            F = G @ R
        else:
            F = G

        Q = np.round(F, decimals=quantize_decimals)
        keys = [row.tobytes() for row in Q]

        # ------------------------------------------------------------------
        # Final occupancy statistics.
        # ------------------------------------------------------------------
        counts = Counter(keys)
        n_distinct = len(counts)
        freq = np.array(list(counts.values()), dtype=np.float64)

        probs = freq / max(n, 1)
        entropy = float(-(probs * np.log(np.clip(probs, 1e-15, 1.0))).sum())   # Shannon
        collision = float(-math.log(np.sum(probs ** 2) + 1e-15))               # Renyi-2

        f1 = int(np.sum(freq == 1))
        singleton_frac = float(f1 / max(n_distinct, 1))
        unseen_mass_gt = float(f1 / max(n, 1))

        # ------------------------------------------------------------------
        # Growth-curve statistic: how fast distinct fingerprints keep appearing
        # as sample size increases.  This is the key replacement when raw counts
        # saturate at N/N.
        # ------------------------------------------------------------------
        ladder = []
        m = 64
        while m < n:
            ladder.append(m)
            m *= 2
        if n not in ladder:
            ladder.append(n)

        seen = set()
        distinct_curve = []
        idx_next = 0
        ladder_sorted = sorted(ladder)
        for i, k in enumerate(keys, start=1):
            seen.add(k)
            while idx_next < len(ladder_sorted) and i == ladder_sorted[idx_next]:
                distinct_curve.append(len(seen))
                idx_next += 1

        ladder_arr = np.asarray(ladder_sorted, dtype=np.float64)
        distinct_arr = np.asarray(distinct_curve, dtype=np.float64)

        if len(ladder_arr) >= 2 and np.all(distinct_arr > 0):
            growth_slope = float(np.polyfit(np.log(ladder_arr), np.log(distinct_arr), 1)[0])
        else:
            growth_slope = None

        # Clamp to a sensible range for readability.
        if growth_slope is not None:
            growth_slope = float(np.clip(growth_slope, 0.0, 1.0))

        # ------------------------------------------------------------------
        # Backward-compatible raw sampled count (auxiliary only).
        # ------------------------------------------------------------------
        raw_log_lower = float(math.log(max(n_distinct, 1)))

        result.update({
            # New preferred A_tau proxies
            "A_tau_effective_log": entropy,
            "A_tau_collision_log": collision,
            "A_tau_growth_slope": growth_slope,
            "A_tau_singleton_frac": singleton_frac,
            "A_tau_unseen_mass_gt": unseen_mass_gt,

            # Auxiliary / diagnostic outputs
            "A_tau_local_log_lower": raw_log_lower,
            "A_tau_local_n_distinct_patterns": int(n_distinct),
            "A_tau_local_n_samples": int(n),
            "A_tau_local_region_diversity": float(n_distinct / max(n, 1)),
            "A_tau_effective_regions": float(math.exp(entropy)),
            "A_tau_collision_effective_regions": float(math.exp(collision)),
            "A_tau_growth_curve_sizes": ladder_sorted,
            "A_tau_growth_curve_distinct": distinct_curve,
            "A_tau_fingerprint_method": f"gradient_projection_q{quantize_decimals}",

            # Backward compatibility: keep old name, but make it explicit that
            # it is only the sampled lower bound.
            "A_tau_local_log": raw_log_lower,
        })

        if verbose:
            print(f"    A_tau_effective_log    = {result['A_tau_effective_log']:.4f}")
            print(f"    A_tau_collision_log    = {result['A_tau_collision_log']:.4f}")
            print(f"    A_tau_growth_slope     = {result['A_tau_growth_slope']}")
            print(f"    A_tau_local_log_lower  = {result['A_tau_local_log_lower']:.4f}")
            print(f"    n fingerprints         = {n_distinct} / {n}")
            print(f"    singleton_frac         = {result['A_tau_singleton_frac']:.4f}")
            print(f"    unseen_mass_gt         = {result['A_tau_unseen_mass_gt']:.4f}")

        return result

    except Exception as e:
        msg = f"A_tau local-region proxies failed: {e}"
        for key in [
            "A_tau_effective_log",
            "A_tau_collision_log",
            "A_tau_growth_slope",
            "A_tau_singleton_frac",
            "A_tau_unseen_mass_gt",
            "A_tau_local_log_lower",
            "A_tau_local_log",
        ]:
            _set_metric(result, key, None, reason=_classify_reason_from_message(msg), detail=msg)
        result["warnings"].append(msg)
        return result
# ---------------------------------------------------------------------------
# IBP components and generalized unstable_frac
# ---------------------------------------------------------------------------

def estimate_ibp_components(
    inst: "VerificationInstance",
    *,
    sample_min_margin: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute IBP-based generic components plus generalized unstable_frac."""
    import onnx

    result: Dict[str, Any] = {"warnings": []}
    try:
        model = onnx.load(inst.onnx_path)
        initializers = {init.name: numpy_helper_to_array(init) for init in model.graph.initializer}
        prop = _interval_propagate(model.graph, initializers, inst, verbose=verbose)
    except Exception as e:
        msg = f"IBP failed: {e}"
        result["warnings"].append(msg)
        for key in [
            "ibp_margin_lb", "ibp_margin_width", "ibp_sample_gap",
            "ibp_relative_gap", "ibp_output_width_sum", "ibp_amplification_ratio",
            "ibp_width_log_slope", "ibp_explosion_depth", "ibp_margin_snr",
            "unstable_frac",
        ]:
            _set_metric(result, key, None, reason=_classify_reason_from_message(msg), detail=msg)
        return result

    result.update({
        "ibp_margin_lb": prop.get("ibp_margin_lower"),
        "ibp_margin_ub": prop.get("ibp_margin_upper"),
        "ibp_margin_width": prop.get("ibp_margin_width"),
        "ibp_output_width_sum": prop.get("ibp_output_width_sum"),
        "ibp_output_width_mean": prop.get("ibp_output_width_mean"),
        "per_layer_widths_pre": prop.get("widths_pre"),
        "per_layer_widths_post": prop.get("widths_post"),
        "per_layer_widths_pre_max": prop.get("widths_pre_max"),
        "n_unstable": prop.get("n_unstable"),
        "n_total_neurons": prop.get("n_total"),
        "unstable_frac": prop.get("unstable_frac"),
        "n_unstable_per_layer": prop.get("n_unstable_per_layer"),
    })

    eps = float(inst.epsilon)
    d = int(inst.input_dim)
    in_width_sum = max(2.0 * eps * d, _EPS)
    out_width_sum = _safe_float(result.get("ibp_output_width_sum"))
    if out_width_sum is not None:
        result["ibp_amplification_ratio"] = float(out_width_sum / in_width_sum)
    else:
        result["ibp_amplification_ratio"] = None

    lb = _safe_float(result.get("ibp_margin_lb"))
    width = _safe_float(result.get("ibp_margin_width"))
    if lb is not None and width is not None:
        result["ibp_margin_snr"] = float(lb / (width + _EPS))

    if lb is not None and sample_min_margin is not None:
        gap = float(sample_min_margin - lb)
        result["ibp_sample_gap"] = gap
        result["ibp_relative_gap"] = float(gap / (abs(sample_min_margin) + _EPS))
    else:
        _set_metric(result, "ibp_sample_gap", None, reason="math_degenerate", detail="sample_min_margin or IBP lower bound unavailable")
        _set_metric(result, "ibp_relative_gap", None, reason="math_degenerate", detail="sample_min_margin or IBP lower bound unavailable")

    widths = [float(w) for w in (prop.get("widths_pre") or []) if w is not None and math.isfinite(float(w))]
    if len(widths) >= 2:
        y = np.log(np.maximum(np.asarray(widths, dtype=np.float64), 1e-12))
        x = np.arange(len(y), dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
        result["ibp_width_log_slope"] = slope
        input_mean_width = max(2.0 * eps, 1e-12)
        threshold = 10.0 * input_mean_width
        explosion_depth = None
        for idx, w in enumerate(widths):
            if w > threshold:
                explosion_depth = idx
                break
        result["ibp_explosion_depth"] = explosion_depth
    else:
        _set_metric(result, "ibp_width_log_slope", None, reason="math_degenerate", detail="fewer than two nonlinear/width-tracked layers")
        _set_metric(result, "ibp_explosion_depth", None, reason="math_degenerate", detail="fewer than two nonlinear/width-tracked layers")

    if verbose:
        print(f"    ibp_margin_lb        = {result.get('ibp_margin_lb')}")
        print(f"    ibp_sample_gap       = {result.get('ibp_sample_gap')}")
        print(f"    ibp_width_log_slope  = {result.get('ibp_width_log_slope')}")
        print(f"    unstable_frac        = {result.get('unstable_frac')}")

    return result


# Backward-compatible old name used by profile.py and maybe analysis scripts.
def estimate_representation_stats(inst: "VerificationInstance", *, verbose: bool = True) -> Dict[str, Any]:
    res = estimate_ibp_components(inst, sample_min_margin=None, verbose=verbose)
    # Old exploratory keys kept where possible.
    if res.get("n_total_neurons"):
        res["U_phi"] = res.get("ibp_output_width_mean")
        res["D_eff_soft"] = None
        res["D_eff_hard"] = None
    return res


# ---------------------------------------------------------------------------
# ONNX / NumPy IBP implementation
# ---------------------------------------------------------------------------

def numpy_helper_to_array(tensor_proto) -> np.ndarray:
    from onnx import numpy_helper
    return numpy_helper.to_array(tensor_proto)


def _get_onnx_input_shape(graph) -> Optional[tuple]:
    try:
        dims = graph.input[0].type.tensor_type.shape.dim
        shape = []
        for d in dims:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(-1)
        return tuple(shape)
    except Exception:
        return None


def _get_interval(intervals: Dict[str, Tuple[np.ndarray, np.ndarray]], name: str):
    if name in intervals:
        return intervals[name]
    raise KeyError(f"missing interval for tensor {name!r}")


def _interval_matmul(x_lo, x_hi, W):
    W_pos = np.maximum(W, 0)
    W_neg = np.minimum(W, 0)
    out_lo = x_lo @ W_pos + x_hi @ W_neg
    out_hi = x_hi @ W_pos + x_lo @ W_neg
    return out_lo, out_hi


def _interval_mul(a_lo, a_hi, b_lo, b_hi):
    products = np.stack([a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi])
    return products.min(axis=0), products.max(axis=0)


def _as_shape_tuple(arr: np.ndarray, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    raw = [int(x) for x in np.asarray(arr).flatten()]
    out = []
    for i, s in enumerate(raw):
        if s == 0 and i < len(input_shape):
            out.append(input_shape[i])
        else:
            out.append(s)
    return tuple(out)


def _get_conv_attrs(node) -> dict:
    attrs = {}
    for a in node.attribute:
        if a.name == "kernel_shape":
            attrs["kernel_shape"] = list(a.ints)
        elif a.name == "pads":
            attrs["pads"] = list(a.ints)
        elif a.name == "strides":
            attrs["strides"] = list(a.ints)
        elif a.name == "dilations":
            attrs["dilations"] = list(a.ints)
        elif a.name == "group":
            attrs["group"] = int(a.i)
    return attrs


def _interval_conv2d(in_lo, in_hi, weight, bias, pads, strides, dilations, groups):
    import torch
    import torch.nn.functional as F

    t_lo = torch.from_numpy(np.asarray(in_lo, dtype=np.float64))
    t_hi = torch.from_numpy(np.asarray(in_hi, dtype=np.float64))
    t_w = torch.from_numpy(np.asarray(weight, dtype=np.float64))

    w_pos = t_w.clamp(min=0)
    w_neg = t_w.clamp(max=0)

    if len(pads) == 4:
        if pads[0] == pads[2] and pads[1] == pads[3]:
            torch_padding = (pads[0], pads[1])
        else:
            t_lo = F.pad(t_lo, (pads[1], pads[3], pads[0], pads[2]))
            t_hi = F.pad(t_hi, (pads[1], pads[3], pads[0], pads[2]))
            torch_padding = (0, 0)
    elif len(pads) == 2:
        torch_padding = tuple(pads)
    else:
        torch_padding = (0, 0)

    kwargs = dict(padding=torch_padding, stride=tuple(strides), dilation=tuple(dilations), groups=groups)
    out_lo = F.conv2d(t_lo, w_pos, **kwargs) + F.conv2d(t_hi, w_neg, **kwargs)
    out_hi = F.conv2d(t_hi, w_pos, **kwargs) + F.conv2d(t_lo, w_neg, **kwargs)
    if bias is not None:
        t_b = torch.from_numpy(np.asarray(bias, dtype=np.float64)).view(1, -1, 1, 1)
        out_lo = out_lo + t_b
        out_hi = out_hi + t_b
    return out_lo.numpy(), out_hi.numpy()


def _softmax_interval_simple(in_lo: np.ndarray, in_hi: np.ndarray, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Safe coarse softmax bounds from exponent interval monotonicity."""
    lo = np.asarray(in_lo, dtype=np.float64)
    hi = np.asarray(in_hi, dtype=np.float64)
    # Stabilize per row.
    lo_shift = lo - np.max(hi, axis=axis, keepdims=True)
    hi_shift = hi - np.max(hi, axis=axis, keepdims=True)
    exp_lo = np.exp(np.clip(lo_shift, -60, 60))
    exp_hi = np.exp(np.clip(hi_shift, -60, 60))
    sum_exp_hi = np.sum(exp_hi, axis=axis, keepdims=True)
    sum_exp_lo = np.sum(exp_lo, axis=axis, keepdims=True)
    out_lo = exp_lo / np.maximum(sum_exp_hi, 1e-12)
    out_hi = exp_hi / np.maximum(sum_exp_lo, 1e-12)
    return out_lo.astype(np.float32), np.minimum(out_hi, 1.0).astype(np.float32)


def _interval_propagate(
    graph,
    initializers: Dict[str, np.ndarray],
    inst: "VerificationInstance",
    verbose: bool = False,
) -> dict:
    """Best-effort IBP through common ONNX ops.

    Unknown ops pass through their first input interval and add a warning.  This
    keeps profiling useful for compatibility experiments while making the raw
    warning visible in the output.
    """
    x0_np = inst.x0.detach().cpu().numpy()
    eps = float(inst.epsilon)
    input_name = graph.input[0].name
    onnx_shape = _get_onnx_input_shape(graph)

    if onnx_shape is not None and len(onnx_shape) >= 2:
        shape = list(onnx_shape)
        if shape[0] <= 0:
            shape[0] = 1
        input_shape = tuple(shape)
    else:
        input_shape = (1, x0_np.size)

    x0_shaped = x0_np.reshape(input_shape).astype(np.float32)
    lo = (x0_shaped - eps).astype(np.float32)
    hi = (x0_shaped + eps).astype(np.float32)

    intervals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {input_name: (lo, hi)}
    for name, arr in initializers.items():
        a = np.asarray(arr).astype(np.float32)
        intervals[name] = (a, a.copy())

    # ONNX Constant nodes.
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    import onnx
                    val = onnx.numpy_helper.to_array(attr.t).astype(np.float32)
                    intervals[node.output[0]] = (val, val.copy())

    widths_pre: List[float] = []
    widths_post: List[float] = []
    widths_pre_max: List[float] = []
    n_unstable_per_layer: List[int] = []
    n_unstable = 0
    n_total = 0
    warnings_seen: List[str] = []

    def record_nonlinearity(pre_lo, pre_hi, post_lo, post_hi, unstable_mask):
        nonlocal n_unstable, n_total
        pre_lo_f = np.asarray(pre_lo).astype(np.float64).reshape(-1)
        pre_hi_f = np.asarray(pre_hi).astype(np.float64).reshape(-1)
        post_lo_f = np.asarray(post_lo).astype(np.float64).reshape(-1)
        post_hi_f = np.asarray(post_hi).astype(np.float64).reshape(-1)
        mask = np.asarray(unstable_mask).reshape(-1).astype(bool)
        n_total += int(pre_lo_f.size)
        n_unstable += int(mask.sum())
        n_unstable_per_layer.append(int(mask.sum()))
        pre_width = pre_hi_f - pre_lo_f
        post_width = post_hi_f - post_lo_f
        widths_pre.append(float(pre_width.mean()) if pre_width.size else 0.0)
        widths_post.append(float(post_width.mean()) if post_width.size else 0.0)
        widths_pre_max.append(float(pre_width.max()) if pre_width.size else 0.0)

    for node in graph.node:
        op = node.op_type
        if op == "Constant":
            continue
        try:
            if op == "Conv":
                in_lo, in_hi = _get_interval(intervals, node.input[0])
                w = initializers.get(node.input[1], _get_interval(intervals, node.input[1])[0])
                if in_lo.ndim == 1:
                    in_ch = w.shape[1]
                    spatial = in_lo.size // in_ch
                    h_dim = int(round(np.sqrt(spatial)))
                    w_dim = max(1, spatial // max(h_dim, 1))
                    in_lo = in_lo.reshape(1, in_ch, h_dim, w_dim)
                    in_hi = in_hi.reshape(1, in_ch, h_dim, w_dim)
                elif in_lo.ndim == 2:
                    in_ch = w.shape[1]
                    spatial = in_lo.shape[1] // in_ch
                    h_dim = int(round(np.sqrt(spatial)))
                    w_dim = max(1, spatial // max(h_dim, 1))
                    in_lo = in_lo.reshape(1, in_ch, h_dim, w_dim)
                    in_hi = in_hi.reshape(1, in_ch, h_dim, w_dim)
                elif in_lo.ndim == 3:
                    in_lo = in_lo.reshape(1, *in_lo.shape)
                    in_hi = in_hi.reshape(1, *in_hi.shape)
                attrs = _get_conv_attrs(node)
                bias = initializers.get(node.input[2]) if len(node.input) > 2 and node.input[2] else None
                out_lo, out_hi = _interval_conv2d(
                    in_lo, in_hi, w, bias,
                    pads=attrs.get("pads", [0, 0, 0, 0]),
                    strides=attrs.get("strides", [1, 1]),
                    dilations=attrs.get("dilations", [1, 1]),
                    groups=attrs.get("group", 1),
                )
                intervals[node.output[0]] = (out_lo.astype(np.float32), out_hi.astype(np.float32))

            elif op in ("MatMul", "Gemm"):
                x_lo, x_hi = _get_interval(intervals, node.input[0])
                w = initializers.get(node.input[1], _get_interval(intervals, node.input[1])[0])
                attrs = {a.name: a for a in node.attribute} if node.attribute else {}
                transA = int(attrs["transA"].i) if "transA" in attrs else 0
                transB = int(attrs["transB"].i) if "transB" in attrs else 0
                alpha = float(attrs["alpha"].f) if "alpha" in attrs else 1.0
                beta_val = float(attrs["beta"].f) if "beta" in attrs else 1.0

                if op == "MatMul" and x_lo.ndim >= 3 and w.ndim >= 2:
                    W = w.T if (w.ndim == 2 and transB) else w
                    out_lo, out_hi = _interval_matmul(x_lo, x_hi, W)
                else:
                    W = np.asarray(w)
                    if W.ndim == 1:
                        W = W.reshape(-1, 1)
                    elif W.ndim > 2:
                        W = W.reshape(W.shape[0], -1)
                    xlo = x_lo.flatten().reshape(1, -1)
                    xhi = x_hi.flatten().reshape(1, -1)
                    if transA:
                        xlo, xhi = xlo.T, xhi.T
                    if transB:
                        W = W.T
                    if xlo.shape[-1] != W.shape[0]:
                        if xlo.shape[-1] == W.shape[1]:
                            W = W.T
                        else:
                            raise RuntimeError(f"{op} shape mismatch: x={xlo.shape}, W={W.shape}")
                    out_lo, out_hi = _interval_matmul(xlo, xhi, W)
                    out_lo = out_lo.flatten()
                    out_hi = out_hi.flatten()

                if alpha != 1.0:
                    if alpha >= 0:
                        out_lo, out_hi = alpha * out_lo, alpha * out_hi
                    else:
                        out_lo, out_hi = alpha * out_hi, alpha * out_lo
                if op == "Gemm" and len(node.input) > 2 and node.input[2]:
                    b = initializers.get(node.input[2])
                    if b is not None:
                        out_lo = out_lo + beta_val * b.reshape(-1)
                        out_hi = out_hi + beta_val * b.reshape(-1)
                intervals[node.output[0]] = (out_lo.astype(np.float32), out_hi.astype(np.float32))

            elif op == "Add":
                a_lo, a_hi = _get_interval(intervals, node.input[0])
                b_lo, b_hi = _get_interval(intervals, node.input[1])
                intervals[node.output[0]] = (a_lo + b_lo, a_hi + b_hi)

            elif op == "Sub":
                a_lo, a_hi = _get_interval(intervals, node.input[0])
                b_lo, b_hi = _get_interval(intervals, node.input[1])
                intervals[node.output[0]] = (a_lo - b_hi, a_hi - b_lo)

            elif op == "Mul":
                a_lo, a_hi = _get_interval(intervals, node.input[0])
                b_lo, b_hi = _get_interval(intervals, node.input[1])
                out_lo, out_hi = _interval_mul(a_lo, a_hi, b_lo, b_hi)
                intervals[node.output[0]] = (out_lo.astype(np.float32), out_hi.astype(np.float32))

            elif op == "Div":
                a_lo, a_hi = _get_interval(intervals, node.input[0])
                b_lo, b_hi = _get_interval(intervals, node.input[1])
                if np.any((b_lo <= 0) & (b_hi >= 0)):
                    raise RuntimeError("Div interval crosses zero")
                out_lo, out_hi = _interval_mul(a_lo, a_hi, 1.0 / b_hi, 1.0 / b_lo)
                intervals[node.output[0]] = (out_lo.astype(np.float32), out_hi.astype(np.float32))

            elif op == "Neg":
                in_lo, in_hi = _get_interval(intervals, node.input[0])
                intervals[node.output[0]] = (-in_hi, -in_lo)

            elif op in ("Relu", "LeakyRelu"):
                pre_lo, pre_hi = _get_interval(intervals, node.input[0])
                alpha = 0.01
                if op == "LeakyRelu":
                    for a in node.attribute:
                        if a.name == "alpha":
                            alpha = float(a.f)
                if op == "Relu":
                    out_lo = np.maximum(pre_lo, 0.0)
                    out_hi = np.maximum(pre_hi, 0.0)
                    unstable = (pre_lo < 0) & (pre_hi > 0)
                else:
                    vals = np.stack([pre_lo, pre_hi, alpha * pre_lo, alpha * pre_hi])
                    out_lo, out_hi = vals.min(axis=0), vals.max(axis=0)
                    unstable = (pre_hi - pre_lo) > 1e-12
                record_nonlinearity(pre_lo, pre_hi, out_lo, out_hi, unstable)
                intervals[node.output[0]] = (out_lo.astype(np.float32), out_hi.astype(np.float32))

            elif op == "Sigmoid":
                pre_lo, pre_hi = _get_interval(intervals, node.input[0])
                out_lo = 1.0 / (1.0 + np.exp(-pre_lo.astype(np.float64)))
                out_hi = 1.0 / (1.0 + np.exp(-pre_hi.astype(np.float64)))
                unstable = (pre_hi - pre_lo) > 1e-12
                record_nonlinearity(pre_lo, pre_hi, out_lo, out_hi, unstable)
                intervals[node.output[0]] = (out_lo.astype(np.float32), out_hi.astype(np.float32))

            elif op == "Tanh":
                pre_lo, pre_hi = _get_interval(intervals, node.input[0])
                out_lo = np.tanh(pre_lo.astype(np.float64))
                out_hi = np.tanh(pre_hi.astype(np.float64))
                unstable = (pre_hi - pre_lo) > 1e-12
                record_nonlinearity(pre_lo, pre_hi, out_lo, out_hi, unstable)
                intervals[node.output[0]] = (out_lo.astype(np.float32), out_hi.astype(np.float32))

            elif op == "Softmax":
                pre_lo, pre_hi = _get_interval(intervals, node.input[0])
                axis = -1
                for a in node.attribute:
                    if a.name == "axis":
                        axis = int(a.i)
                out_lo, out_hi = _softmax_interval_simple(pre_lo, pre_hi, axis=axis)
                unstable = (pre_hi - pre_lo) > 1e-12
                record_nonlinearity(pre_lo, pre_hi, out_lo, out_hi, unstable)
                intervals[node.output[0]] = (out_lo, out_hi)

            elif op == "Flatten":
                in_lo, in_hi = _get_interval(intervals, node.input[0])
                axis = 1
                for a in node.attribute:
                    if a.name == "axis":
                        axis = int(a.i)
                if in_lo.ndim > 1 and axis > 0:
                    new_shape = list(in_lo.shape[:axis]) + [-1]
                    intervals[node.output[0]] = (in_lo.reshape(new_shape), in_hi.reshape(new_shape))
                else:
                    intervals[node.output[0]] = (in_lo.flatten(), in_hi.flatten())

            elif op == "Reshape":
                in_lo, in_hi = _get_interval(intervals, node.input[0])
                if len(node.input) > 1:
                    shape_lo, _ = _get_interval(intervals, node.input[1])
                    target = _as_shape_tuple(shape_lo, in_lo.shape)
                    intervals[node.output[0]] = (in_lo.reshape(target), in_hi.reshape(target))
                else:
                    intervals[node.output[0]] = (in_lo.flatten(), in_hi.flatten())

            elif op == "Squeeze":
                in_lo, in_hi = _get_interval(intervals, node.input[0])
                axes = None
                for a in node.attribute:
                    if a.name == "axes":
                        axes = tuple(int(x) for x in a.ints)
                if axes is None and len(node.input) > 1:
                    ax_lo, _ = _get_interval(intervals, node.input[1])
                    axes = tuple(int(x) for x in ax_lo.flatten())
                intervals[node.output[0]] = (np.squeeze(in_lo, axis=axes), np.squeeze(in_hi, axis=axes))

            elif op == "Unsqueeze":
                in_lo, in_hi = _get_interval(intervals, node.input[0])
                axes = None
                for a in node.attribute:
                    if a.name == "axes":
                        axes = [int(x) for x in a.ints]
                if axes is None and len(node.input) > 1:
                    ax_lo, _ = _get_interval(intervals, node.input[1])
                    axes = [int(x) for x in ax_lo.flatten()]
                out_lo, out_hi = in_lo, in_hi
                for ax in sorted(axes or []):
                    out_lo = np.expand_dims(out_lo, axis=ax)
                    out_hi = np.expand_dims(out_hi, axis=ax)
                intervals[node.output[0]] = (out_lo, out_hi)

            elif op == "Transpose":
                in_lo, in_hi = _get_interval(intervals, node.input[0])
                perm = None
                for a in node.attribute:
                    if a.name == "perm":
                        perm = tuple(int(x) for x in a.ints)
                intervals[node.output[0]] = (np.transpose(in_lo, perm), np.transpose(in_hi, perm))

            elif op == "Concat":
                axis = 0
                for a in node.attribute:
                    if a.name == "axis":
                        axis = int(a.i)
                all_lo = [_get_interval(intervals, inp)[0] for inp in node.input]
                all_hi = [_get_interval(intervals, inp)[1] for inp in node.input]
                intervals[node.output[0]] = (np.concatenate(all_lo, axis=axis), np.concatenate(all_hi, axis=axis))

            elif op == "ReduceSum":
                in_lo, in_hi = _get_interval(intervals, node.input[0])
                axes = None
                keepdims = 1
                for a in node.attribute:
                    if a.name == "axes":
                        axes = tuple(int(x) for x in a.ints)
                    elif a.name == "keepdims":
                        keepdims = int(a.i)
                if axes is None and len(node.input) > 1:
                    ax_lo, _ = _get_interval(intervals, node.input[1])
                    axes = tuple(int(x) for x in ax_lo.flatten())
                intervals[node.output[0]] = (np.sum(in_lo, axis=axes, keepdims=bool(keepdims)), np.sum(in_hi, axis=axes, keepdims=bool(keepdims)))

            elif op in ("Identity", "Dropout"):
                intervals[node.output[0]] = _get_interval(intervals, node.input[0])

            elif op == "BatchNormalization":
                in_lo, in_hi = _get_interval(intervals, node.input[0])
                if len(node.input) >= 5 and all(inp in initializers for inp in node.input[1:5]):
                    scale = initializers[node.input[1]].astype(np.float32)
                    bn_bias = initializers[node.input[2]].astype(np.float32)
                    mean = initializers[node.input[3]].astype(np.float32)
                    var = initializers[node.input[4]].astype(np.float32)
                    bn_eps = 1e-5
                    for a in node.attribute:
                        if a.name == "epsilon":
                            bn_eps = float(a.f)
                    s = scale / np.sqrt(var + bn_eps)
                    shift = bn_bias - mean * s
                    if in_lo.ndim == 4:
                        s = s.reshape(1, -1, 1, 1)
                        shift = shift.reshape(1, -1, 1, 1)
                    s_pos = np.maximum(s, 0)
                    s_neg = np.minimum(s, 0)
                    out_lo = s_pos * in_lo + s_neg * in_hi + shift
                    out_hi = s_pos * in_hi + s_neg * in_lo + shift
                    intervals[node.output[0]] = (out_lo, out_hi)
                else:
                    intervals[node.output[0]] = (in_lo, in_hi)

            else:
                # Compatibility fallback: pass through first input.  Mark warning.
                if not node.input:
                    raise RuntimeError(f"unknown op {op} has no inputs")
                intervals[node.output[0]] = _get_interval(intervals, node.input[0])
                msg = f"unknown op {op}; passed through first input interval"
                warnings_seen.append(msg)
                if verbose:
                    print(f"    WARNING: {msg}")
        except Exception as e:
            raise RuntimeError(f"IBP failed at {op} node {node.name!r}: {e}") from e

    out_name = graph.output[0].name if graph.output else None
    if out_name is None or out_name not in intervals:
        raise RuntimeError("final output interval unavailable")
    out_lo, out_hi = intervals[out_name]
    lo_flat = np.asarray(out_lo).flatten().astype(np.float64)
    hi_flat = np.asarray(out_hi).flatten().astype(np.float64)

    true_c = int(getattr(inst, "true_class", 0))
    if lo_flat.size == 1:
        margin_lb = float(lo_flat[0])
        margin_ub = float(hi_flat[0])
    elif 0 <= true_c < lo_flat.size:
        comp_hi = hi_flat.copy()
        comp_lo = lo_flat.copy()
        comp_hi[true_c] = -np.inf
        comp_lo[true_c] = np.inf
        margin_lb = float(lo_flat[true_c] - np.max(comp_hi))
        margin_ub = float(hi_flat[true_c] - np.min(comp_lo))
    else:
        margin_lb = None
        margin_ub = None

    output_width = hi_flat - lo_flat
    return {
        "widths_pre": widths_pre,
        "widths_post": widths_post,
        "widths_pre_max": widths_pre_max,
        "n_unstable_per_layer": n_unstable_per_layer,
        "n_unstable": int(n_unstable),
        "n_total": int(n_total),
        "unstable_frac": float(n_unstable / n_total) if n_total > 0 else 0.0,
        "ibp_margin_lower": margin_lb,
        "ibp_margin_upper": margin_ub,
        "ibp_margin_width": (float(margin_ub - margin_lb) if margin_lb is not None and margin_ub is not None else None),
        "ibp_output_width_sum": float(np.sum(output_width)),
        "ibp_output_width_mean": float(np.mean(output_width)) if output_width.size else 0.0,
        "warnings": warnings_seen,
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases for old scripts that import these names.
# ---------------------------------------------------------------------------

def estimate_clarke_lipschitz(inst: "VerificationInstance", *, n_samples: int = 200, verbose: bool = True) -> Dict[str, Any]:
    res = estimate_generic_components(inst, n_samples=n_samples, n_pairs=min(450, n_samples * 2), verbose=False)
    L = res.get("grad_l1_p95") if getattr(inst, "norm_p", "inf") == "inf" else res.get("grad_l2_mean")
    out = {
        "Lc": L,
        "n_samples": res.get("n_gradient_samples", 0),
        "mean_grad_norm": res.get("grad_l1_mean"),
        "std_grad_norm": None,
        "warnings": res.get("warnings", []),
    }
    if verbose:
        print(f"    L_c proxy = {out['Lc']}")
    return out


def estimate_margin_gap(inst: "VerificationInstance", *, ibp_lower_at_x0: Optional[float] = None, verbose: bool = True) -> Dict[str, Any]:
    sample_min = float(getattr(inst, "margin_at_x0", 0.0))
    res = estimate_ibp_components(inst, sample_min_margin=sample_min, verbose=False)
    g0 = float(getattr(inst, "margin_at_x0", sample_min))
    lb = ibp_lower_at_x0 if ibp_lower_at_x0 is not None else res.get("ibp_margin_lb")
    out: Dict[str, Any] = {"warnings": []}
    if lb is None or abs(g0) < 1e-12:
        _set_metric(out, "margin_gap", None, reason="math_degenerate", detail="IBP lower or nominal margin unavailable")
    else:
        out["margin_gap"] = float((g0 - float(lb)) / (abs(g0) + _EPS))
    out["ibp_lower_at_x0"] = lb
    out["margin_at_x0"] = g0
    if verbose:
        print(f"    margin_gap = {out.get('margin_gap')}")
    return out
