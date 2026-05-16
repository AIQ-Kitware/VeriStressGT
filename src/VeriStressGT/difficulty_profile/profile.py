from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Canonical component registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentHypothesis:
    name: str
    paper_symbol: str
    styles: Tuple[str, ...]
    direction: str
    rationale: str


COMPONENT_HYPOTHESES: Dict[str, ComponentHypothesis] = {
    # Boundary / margin proximity, now estimated with a mixed stress sampler.
    "margin_sample_min": ComponentHypothesis(
        "margin_sample_min", "min_{x in S_stress} m(x)", ("R", "D", "P", "S"), "higher is easier",
        "Empirical closest approach to the decision boundary using uniform, boundary, gradient, and PGD stress samples.",
    ),
    "margin_to_variation_ratio": ComponentHypothesis(
        "margin_to_variation_ratio", "Q_{0.01}[m(S_stress)] / (Std[m(S_stress)] + eps)", ("R", "D", "P", "S"), "higher is easier",
        "Stress-sampled margin buffer relative to empirical margin variation.",
    ),

    # First-order sensitivity geometry requested for the compact profile.
    "effective_grad_dim_mean": ComponentHypothesis(
        "effective_grad_dim_mean", "E ||g||_1^2/||g||_2^2", ("P", "S", "D"), "higher is harder",
        "Participation-ratio dimension of dangerous input sensitivity.",
    ),
    "grad_sensitivity_concentration": ComponentHypothesis(
        "grad_sensitivity_concentration", "E ||g||_inf/||g||_1", ("P", "S"), "higher is easier",
        "Whether sensitivity is concentrated on a few coordinates rather than spread across the box.",
    ),
    "first_order_margin_ratio": ComponentHypothesis(
        "first_order_margin_ratio", "m(x0)/(eps ||grad m(x0)||_1)", ("R", "D", "P", "S"), "higher is easier",
        "Local linear robustness ratio at x0.",
    ),

    # Relaxation / interval geometry requested for the compact profile.
    "ibp_relative_gap": ComponentHypothesis(
        "ibp_relative_gap", "(min_stress m - lower_IBP(m))/|min_stress m|", ("R", "D", "P", "S"), "higher is harder",
        "Scale-normalized IBP looseness relative to the stress-sampled margin floor.",
    ),
    "unstable_frac": ComponentHypothesis(
        "unstable_frac", "nontrivial nonlinear IBP coords / total nonlinear coords", ("P", "S", "D"), "higher is harder",
        "ReLU instability fraction generalized to nonlinear interval exposure.",
    ),

    "A_tau_local_log": ComponentHypothesis(
        "A_tau_local_log", "log # sampled local gradient fingerprints", ("P", "S"), "higher is harder",
        "Sampled lower bound on local affine/gradient fingerprints in the input set.",
    ),
    "A_tau_effective_log": ComponentHypothesis(
        "A_tau_effective_log", "Shannon log-effective # fingerprints", ("P", "S"), "higher is harder",
        "Entropy-based effective number of local affine/gradient fingerprints.",
    )
}

CANONICAL_COMPONENT_NAMES: Tuple[str, ...] = tuple(COMPONENT_HYPOTHESES.keys())


# ---------------------------------------------------------------------------
# DifficultyProfile dataclass
# ---------------------------------------------------------------------------

@dataclass
class DifficultyProfile:
    # ---- Instance identification ----
    onnx_path: str = ""
    vnnlib_path: str = ""
    epsilon: float = 0.0
    input_dim: int = 0
    num_classes: int = 0
    true_class: int = -1
    norm_p: str = "inf"

    # ---- Canonical compact generic components ----
    margin_nominal: Optional[float] = None
    margin_sample_min: Optional[float] = None
    margin_sample_q01: Optional[float] = None
    margin_std: Optional[float] = None
    margin_to_variation_ratio: Optional[float] = None

    lip_empirical_max: Optional[float] = None
    grad_l1_mean: Optional[float] = None
    first_order_margin_ratio: Optional[float] = None
    sampled_first_order_margin_ratio: Optional[float] = None

    ibp_margin_lb: Optional[float] = None
    ibp_sample_gap: Optional[float] = None
    ibp_relative_gap: Optional[float] = None
    ibp_output_width_sum: Optional[float] = None
    ibp_margin_width: Optional[float] = None
    ibp_amplification_ratio: Optional[float] = None
    ibp_width_log_slope: Optional[float] = None
    ibp_explosion_depth: Optional[int] = None
    ibp_margin_snr: Optional[float] = None

    chord_nonlinearity_mean: Optional[float] = None
    second_diff_mean: Optional[float] = None
    grad_variation_l2: Optional[float] = None
    grad_direction_dispersion: Optional[float] = None
    linearization_error_mean: Optional[float] = None
    relative_linearization_error: Optional[float] = None

    effective_grad_dim_mean: Optional[float] = None
    grad_sensitivity_concentration: Optional[float] = None
    grad_cov_effective_rank: Optional[float] = None
    input_box_log_volume: Optional[float] = None
    effective_box_log_volume: Optional[float] = None
    relative_sample_min_margin: Optional[float] = None
    spec_direction_conditioning: Optional[float] = None

    # ---- New optimization-geometry components ----
    pgd_improvement_ratio: Optional[float] = None
    low_margin_basin_frac: Optional[float] = None
    best_point_boundary_frac: Optional[float] = None
    first_order_adv_error_ratio: Optional[float] = None
    stress_min_grad_alignment: Optional[float] = None
    # ---- A_tau / local-region fingerprint family ----
    A_tau_local_log: Optional[float] = None
    A_tau_effective_log: Optional[float] = None
    A_tau_collision_log: Optional[float] = None
    A_tau_growth_slope: Optional[float] = None
    unstable_frac: Optional[float] = None

    # ---- Auxiliary columns that are useful downstream ----
    margin_sample_q05: Optional[float] = None
    margin_range: Optional[float] = None
    margin_iqr: Optional[float] = None
    uniform_margin_min: Optional[float] = None
    pgd_margin_min: Optional[float] = None
    low_margin_basin_threshold: Optional[float] = None
    stress_min_linf_radius: Optional[float] = None
    first_order_adv_margin: Optional[float] = None
    first_order_adv_predicted_margin: Optional[float] = None
    lip_empirical_p95: Optional[float] = None
    lip_empirical_mean: Optional[float] = None
    grad_l1_p95: Optional[float] = None
    grad_l2_mean: Optional[float] = None
    grad_linf_mean: Optional[float] = None
    grad_linf_p95: Optional[float] = None
    grad_l1_x0: Optional[float] = None
    effective_grad_dim_p95: Optional[float] = None
    grad_abs_direction_dispersion: Optional[float] = None
    grad_cov_trace: Optional[float] = None
    linearization_error_p95: Optional[float] = None
    linearization_error_max: Optional[float] = None
    chord_nonlinearity_p95: Optional[float] = None
    chord_nonlinearity_max: Optional[float] = None
    second_diff_p95: Optional[float] = None
    second_diff_max: Optional[float] = None
    ibp_margin_ub: Optional[float] = None
    ibp_output_width_mean: Optional[float] = None                 # backward-compatible raw lower bound
    A_tau_local_log_lower: Optional[float] = None
    A_tau_local_n_distinct_patterns: Optional[int] = None
    A_tau_local_n_samples: Optional[int] = None
    A_tau_local_region_diversity: Optional[float] = None
    A_tau_effective_regions: Optional[float] = None
    A_tau_collision_effective_regions: Optional[float] = None
    A_tau_singleton_frac: Optional[float] = None
    A_tau_unseen_mass_gt: Optional[float] = None
    A_tau_growth_curve_sizes: Optional[list] = None
    A_tau_growth_curve_distinct: Optional[list] = None
    A_tau_fingerprint_method: Optional[str] = None

    # ---- Raw IBP / representation diagnostics ----
    n_unstable: Optional[int] = None
    n_total_neurons: Optional[int] = None
    n_unstable_per_layer: Optional[list] = None
    per_layer_widths_pre: Optional[list] = None
    per_layer_widths_post: Optional[list] = None
    per_layer_widths_pre_max: Optional[list] = None

    # ---- Backward-compatible legacy names, filled when cheap ----
    margin_at_x0: Optional[float] = None
    Lc: Optional[float] = None
    Lc_samples: int = 0
    tau: float = 0.1
    C_score: Optional[float] = None
    S: Optional[float] = None
    eta: Optional[float] = None
    U_phi: Optional[float] = None
    G_ratio: Optional[float] = None
    D_eff_soft: Optional[float] = None
    N_hat: Optional[float] = None

    # ---- Metadata ----
    estimation_time_s: float = 0.0
    warnings: list = field(default_factory=list)
    component_times: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)

        def _convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        return _convert(d)

    def to_flat_dict(self) -> Dict[str, Any]:
        raw = self.to_dict()
        flat: Dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, (list, dict)):
                flat[k] = json.dumps(v)
            else:
                flat[k] = v
        return flat

    def to_canonical_dict(self) -> Dict[str, Any]:
        base = {
            "onnx_path": self.onnx_path,
            "vnnlib_path": self.vnnlib_path,
            "epsilon": self.epsilon,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "true_class": self.true_class,
            "norm_p": self.norm_p,
            "margin_at_x0": self.margin_at_x0,
            "n_total_neurons": self.n_total_neurons,
        }
        for name in CANONICAL_COMPONENT_NAMES:
            base[name] = getattr(self, name, None)
        return base

    def save_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def save_csv(self, path: str) -> None:
        row = self.to_flat_dict()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    def append_csv_row(self, path: str) -> None:
        row = self.to_flat_dict()
        out_path = Path(path)
        write_header = not out_path.exists() or out_path.stat().st_size == 0
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def append_canonical_csv_row(self, path: str) -> None:
        row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
               for k, v in self.to_canonical_dict().items()}
        out_path = Path(path)
        write_header = not out_path.exists() or out_path.stat().st_size == 0
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def save_both(self, json_path: str, csv_path: str) -> None:
        self.save_json(json_path)
        self.save_csv(csv_path)

    @classmethod
    def load_json(cls, path: str) -> "DifficultyProfile":
        data = json.loads(Path(path).read_text())
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def C(self) -> Optional[float]:
        return self.C_score

    def summary(self) -> str:
        def _fmt(name, val, spec=".4f"):
            if val is None:
                return f"    {name:<34} = N/A"
            try:
                return f"    {name:<34} = {format(val, spec)}"
            except Exception:
                return f"    {name:<34} = {val}"

        lines = [
            "=" * 72,
            "Compact Difficulty Profile Estimate",
            "=" * 72,
            f"  Instance:    {Path(self.onnx_path).name}",
            f"  epsilon:     {self.epsilon}",
            f"  input dim:   {self.input_dim}",
            f"  true class:  {self.true_class}",
            f"  margin g(x0): {self.margin_at_x0}",
            "",
            "  Canonical components:",
        ]
        for name in CANONICAL_COMPONENT_NAMES:
            val = getattr(self, name, None)
            lines.append(_fmt(name, val))
        if self.warnings:
            lines += ["", "  Warnings:"] + [f"    - {w}" for w in self.warnings[:25]]
            if len(self.warnings) > 25:
                lines.append(f"    - ... {len(self.warnings) - 25} more warnings")
        lines += ["", f"  Total estimation time: {self.estimation_time_s:.2f}s", "=" * 72]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _copy_result_to_profile(profile: DifficultyProfile, result: Dict[str, Any]) -> None:
    for k, v in result.items():
        if k == "warnings":
            profile.warnings.extend(v or [])
        elif hasattr(profile, k):
            setattr(profile, k, v)


def estimate_profile(
    onnx_path: str,
    vnnlib_path: str,
    *,
    n_pgd_restarts: int = 3,
    n_pgd_steps: int = 60,
    n_gradient_samples: int = 200,
    n_beta_pairs: int = 450,
    tau: float = 0.1,
    component_timeout: float = 960.0,
    device: str = "cpu",
    verbose: bool = True,
    run_exploratory: bool = True,
) -> DifficultyProfile:
    """Estimate the generic Difficulty Profile for an ONNX + VNNLIB pair.

    The signature is intentionally compatible with the prior version.  PGD
    arguments are retained for callers/CLIs but are not central to the new
    generic profile; n_pgd_steps is used only as a soft cap for curvature-style
    finite-difference probes if future extensions need it.
    """
    import signal
    from .instance_loader import load_instance
    from . import components

    class _ComponentTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise _ComponentTimeout()

    component_errors: Dict[str, str] = {}

    def _run_with_timeout(fn, label: str):
        import traceback as _tb
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(component_timeout))
            out = fn()
            signal.alarm(0)
            return out
        except _ComponentTimeout:
            msg = f"TIMEOUT after {component_timeout:.0f}s"
            component_errors[label] = msg
            if verbose:
                print(f"    {msg} — skipping {label}")
            return None
        except Exception as e:
            signal.alarm(0)
            msg = f"{type(e).__name__}: {e}"
            component_errors[label] = msg
            if verbose:
                print(f"    ERROR in {label}: {msg}")
                print(f"    Traceback:\n{_tb.format_exc()}")
            return None
        finally:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    total_start = time.time()
    profile = DifficultyProfile(onnx_path=onnx_path, vnnlib_path=vnnlib_path, tau=tau)

    if verbose:
        print("Loading instance...")
    inst = load_instance(onnx_path, vnnlib_path, device=device)
    profile.epsilon = float(inst.epsilon)
    profile.input_dim = int(inst.input_dim)
    profile.num_classes = int(inst.num_classes)
    profile.true_class = int(inst.true_class)
    profile.norm_p = str(inst.norm_p)
    profile.margin_at_x0 = float(inst.margin_at_x0)
    profile.margin_nominal = float(inst.margin_at_x0)

    if verbose:
        print(f"  margin g(x0) = {profile.margin_at_x0:.6f}")
        if profile.margin_at_x0 <= 0:
            print("  WARNING: margin <= 0, instance may not be robust.")

    # ------------------------------------------------------------------
    # Main generic sample/gradient/nonlinearity pass.
    # ------------------------------------------------------------------
    if verbose:
        print("\nEstimating stress-margin / gradient / nonlinearity components...")
    t0 = time.time()
    generic_result = _run_with_timeout(
        lambda: components.estimate_generic_components(
            inst,
            n_samples=n_gradient_samples,
            n_pairs=n_beta_pairs,
            verbose=verbose,
        ),
        "generic_components",
    )
    if generic_result:
        _copy_result_to_profile(profile, generic_result)
        # Backward-compatible Lipschitz proxy.
        profile.Lc = generic_result.get("grad_l1_p95") or generic_result.get("lip_empirical_p95")
        profile.Lc_samples = generic_result.get("n_gradient_samples", 0) or 0
        # Backward-compatible coarse C: empirical low margin divided by first-order scale.
        profile.C_score = generic_result.get("sampled_first_order_margin_ratio")
    else:
        profile.warnings.append("Generic/stress component estimation timed out or failed.")
    profile.component_times["generic_components"] = time.time() - t0

    # ------------------------------------------------------------------
    # IBP pass.
    # ------------------------------------------------------------------
    if verbose:
        print("\nEstimating IBP / interval components...")
    t0 = time.time()
    ibp_result = _run_with_timeout(
        lambda: components.estimate_ibp_components(
            inst,
            sample_min_margin=profile.margin_sample_min,
            verbose=verbose,
        ),
        "ibp_components",
    )
    if ibp_result:
        _copy_result_to_profile(profile, ibp_result)
        # Cheap legacy aliases, mostly for old plotting code.
        profile.U_phi = profile.ibp_output_width_mean
        profile.G_ratio = profile.ibp_width_log_slope
        if profile.n_total_neurons is not None and profile.unstable_frac is not None:
            profile.N_hat = float(profile.unstable_frac * profile.n_total_neurons)
    else:
        profile.warnings.append("IBP component estimation timed out or failed.")
    profile.component_times["ibp_components"] = time.time() - t0

    # ------------------------------------------------------------------
    # A_tau proxies via generic gradient fingerprints.
    # ------------------------------------------------------------------
    if verbose:
        print("\nEstimating A_tau proxies (effective / collision / growth)...")
    t0 = time.time()
    atau_result = _run_with_timeout(
        lambda: components.estimate_local_region_count(
            inst,
            n_samples=min(1024, n_gradient_samples * 2),
            verbose=verbose,
        ),
        "A_tau_proxies",
    )
    if atau_result:
        _copy_result_to_profile(profile, atau_result)
    else:
        profile.warnings.append("A_tau_proxies estimation timed out or failed.")
    profile.component_times["A_tau_proxies"] = time.time() - t0

    # ------------------------------------------------------------------
    # Fill any obvious derived/backward-compatible fields.
    # ------------------------------------------------------------------
    if profile.margin_nominal is None:
        profile.margin_nominal = profile.margin_at_x0
    if profile.S is None and profile.margin_std is not None and profile.margin_sample_min is not None:
        # Not the old sharpness definition; this is just a harmless legacy proxy
        # so old CSV readers do not see an entirely blank S column.
        denom = abs(profile.margin_sample_min) + profile.margin_std + 1e-12
        profile.S = float(profile.margin_std / denom)
    if profile.eta is None:
        profile.eta = profile.effective_grad_dim_mean

    profile.estimation_time_s = time.time() - total_start

    for label, msg in component_errors.items():
        profile.warnings.append(f"[{label}] {msg}")

    if verbose:
        print(profile.summary())

    return profile
