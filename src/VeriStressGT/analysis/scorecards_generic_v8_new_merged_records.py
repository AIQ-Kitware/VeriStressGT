# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# from pathlib import Path
# from typing import List

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from profile_viz_utils_generic_v8 import (
#     add_figure_title,
#     apply_style,
#     available_candidate_components,
#     choose_timeout_cap,
#     compute_metrics_for_subset,
#     display_outlier_mask,
#     ensure_dir,
#     fastest_verifier,
#     infer_verifiers,
#     load_wide_records,
#     nice_component_name,
#     robust_certification_subset,
#     safe_float,
#     winner_legend_handles,
#     VERIFIER_COLORS,
# )


# def heatmap_matrix(metrics: pd.DataFrame, components: List[str], verifiers: List[str], value_col: str) -> np.ndarray:
#     mat = np.full((len(components), len(verifiers)), np.nan)
#     for i, c in enumerate(components):
#         for j, v in enumerate(verifiers):
#             row = metrics[(metrics["component"] == c) & (metrics["verifier"] == v)]
#             if row.empty:
#                 continue
#             val = safe_float(row.iloc[0][value_col])
#             mat[i, j] = np.nan if val is None else val
#     return mat


# def count_matrix(metrics: pd.DataFrame, components: List[str], verifiers: List[str], value_col: str = "n_used") -> np.ndarray:
#     mat = np.full((len(components), len(verifiers)), np.nan)
#     for i, c in enumerate(components):
#         for j, v in enumerate(verifiers):
#             row = metrics[(metrics["component"] == c) & (metrics["verifier"] == v)]
#             if row.empty:
#                 continue
#             try:
#                 mat[i, j] = float(row.iloc[0][value_col])
#             except Exception:
#                 mat[i, j] = np.nan
#     return mat


# def optional_count_matrix(metrics: pd.DataFrame, components: List[str], verifiers: List[str], value_col: str) -> np.ndarray:
#     if value_col not in metrics.columns:
#         return np.full((len(components), len(verifiers)), np.nan)
#     return count_matrix(metrics, components, verifiers, value_col=value_col)


# def annotate_metric_matrix(ax, mat: np.ndarray, threshold: float = 0.28):
#     for i in range(mat.shape[0]):
#         for j in range(mat.shape[1]):
#             val = mat[i, j]
#             text = "N/A" if np.isnan(val) else f"{val:+.2f}"
#             color = "white" if (not np.isnan(val) and abs(val) > threshold) else "#1f2937"
#             ax.text(j, i, text, ha="center", va="center", fontsize=8.0, color=color)


# def annotate_count_matrix(
#     ax,
#     mat: np.ndarray,
#     unsupported: np.ndarray | None = None,
#     math_na: np.ndarray | None = None,
#     numerical_na: np.ndarray | None = None,
# ):
#     for i in range(mat.shape[0]):
#         for j in range(mat.shape[1]):
#             val = mat[i, j]
#             lines = ["N/A" if np.isnan(val) else f"n={int(round(val))}"]
#             extras = []
#             if unsupported is not None:
#                 u = unsupported[i, j]
#                 if not np.isnan(u) and u > 0:
#                     extras.append(f"U{int(round(u))}")
#             if math_na is not None:
#                 m = math_na[i, j]
#                 if not np.isnan(m) and m > 0:
#                     extras.append(f"M{int(round(m))}")
#             if numerical_na is not None:
#                 n = numerical_na[i, j]
#                 if not np.isnan(n) and n > 0:
#                     extras.append(f"E{int(round(n))}")
#             if extras:
#                 lines.append(" / ".join(extras))
#             text = "\n".join(lines)
#             ax.text(j, i, text, ha="center", va="center", fontsize=7.6, color="#1f2937")


# def plot_three_lens_grid(metrics: pd.DataFrame, row_key: str, title: str, subtitle: str,
#                          out_path_png: Path, out_path_pdf: Path) -> None:
#     groups = list(dict.fromkeys(metrics[row_key]))
#     verifiers = list(dict.fromkeys(metrics["verifier"]))
#     components = list(dict.fromkeys(metrics["component"]))

#     fig_h = max(5.5, 3.1 * len(groups))
#     fig, axes = plt.subplots(len(groups), 3, figsize=(19.2, fig_h), constrained_layout=False)
#     if len(groups) == 1:
#         axes = np.array([axes])
#     fig.subplots_adjust(left=0.22, right=0.98, bottom=0.05, top=0.92, hspace=0.42, wspace=0.24)
#     add_figure_title(fig, title, subtitle, top=0.955)

#     vmax_l = max(0.55, np.nanmax(np.abs(pd.to_numeric(metrics["spearman_runtime"], errors="coerce"))))
#     vmax_r = max(0.22, np.nanmax(np.abs(pd.to_numeric(metrics["timeout_effect"], errors="coerce"))))
#     vmax_n = max(1.0, np.nanmax(pd.to_numeric(metrics["n_used"], errors="coerce")))
#     have_na_reason_counts = any(c in metrics.columns for c in ["n_na_unsupported", "n_na_math", "n_na_numerical"])

#     for gi, group in enumerate(groups):
#         sub = metrics[metrics[row_key] == group]
#         mats = [
#             heatmap_matrix(sub, components, verifiers, "spearman_runtime"),
#             heatmap_matrix(sub, components, verifiers, "timeout_effect"),
#             count_matrix(sub, components, verifiers, "n_used"),
#         ]
#         mat_unsupported = optional_count_matrix(sub, components, verifiers, "n_na_unsupported") if have_na_reason_counts else None
#         mat_math = optional_count_matrix(sub, components, verifiers, "n_na_math") if have_na_reason_counts else None
#         mat_numerical = optional_count_matrix(sub, components, verifiers, "n_na_numerical") if have_na_reason_counts else None
#         titles = [
#             f"{group}: runtime lens",
#             f"{group}: timeout lens",
#             f"{group}: sample size",
#         ]
#         vmaxs = [vmax_l, vmax_r, vmax_n]
#         cmaps = ["RdBu_r", "RdBu_r", "Blues"]

#         for ci, (mat, title_local, vmax, cmap) in enumerate(zip(mats, titles, vmaxs, cmaps)):
#             ax = axes[gi, ci]
#             if ci < 2:
#                 im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
#             else:
#                 im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
#             ax.set_title(title_local, loc="left", pad=8)
#             ax.set_xticks(np.arange(len(verifiers)))
#             ax.set_xticklabels(verifiers, fontsize=9.3)
#             ax.set_yticks(np.arange(len(components)))
#             if ci == 0:
#                 ax.set_yticklabels([nice_component_name(c) for c in components], fontsize=9.2)
#             else:
#                 ax.set_yticklabels([])

#             if ci < 2:
#                 annotate_metric_matrix(ax, mat)
#             else:
#                 annotate_count_matrix(ax, mat, unsupported=mat_unsupported, math_na=mat_math, numerical_na=mat_numerical)

#             for spine in ax.spines.values():
#                 spine.set_visible(False)
#             cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)
#             if ci == 0:
#                 cbar.set_label("spearman runtime", fontsize=8.5)
#             elif ci == 1:
#                 cbar.set_label("AUC - 0.5", fontsize=8.5)
#             else:
#                 label = "n used"
#                 if have_na_reason_counts:
#                     label += "\nannot: U=unsupported, M=math, E=numerical"
#                 cbar.set_label(label, fontsize=8.5)
#             cbar.ax.tick_params(labelsize=8.5)

#     fig.savefig(out_path_png, dpi=250)
#     fig.savefig(out_path_pdf)
#     plt.close(fig)


# def plot_verifier_planes_all_components(records: pd.DataFrame, out_dir: Path, outlier_method: str, low_q: float, high_q: float) -> None:
#     verifiers = infer_verifiers(records)
#     timeout_cap = choose_timeout_cap(records, verifiers)
#     components = available_candidate_components(records)

#     xcomp = "margin_gap" if "margin_gap" in components else components[0]
#     ycomps = [c for c in components if c != xcomp]
#     if not ycomps:
#         return

#     ncols = 3
#     nrows = int(np.ceil(len(ycomps) / ncols))
#     fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.6 * nrows), constrained_layout=False)
#     axes = np.array(axes).reshape(nrows, ncols)
#     fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.88, hspace=0.42, wspace=0.28)
#     add_figure_title(
#         fig,
#         "Explicit verifier-regime planes across all candidate components",
#         f"Scatter panels are trimmed for readability using {outlier_method}; the scorecard tables use the same component-wise outlier rule.",
#         top=0.935,
#     )

#     total_hidden = 0

#     for ax, yc in zip(axes.flat, ycomps):
#         sub = records.copy()
#         sub = sub[pd.to_numeric(sub[xcomp], errors="coerce").notna() & pd.to_numeric(sub[yc], errors="coerce").notna()].copy()
#         keep_mask, _ = display_outlier_mask(sub, [xcomp, yc], method=outlier_method, low_q=low_q, high_q=high_q)
#         total_hidden += int((~keep_mask).sum())
#         sub = sub.loc[keep_mask].copy()

#         winners, gaps = [], []
#         for _, row in sub.iterrows():
#             w, _, g = fastest_verifier(row, verifiers, timeout_cap)
#             winners.append(w)
#             gaps.append(0.0 if g is None else g)
#         sub["winner"] = winners
#         sub["adv_gap"] = gaps

#         for v in ["abcrown", "neuralsat", "nnenum", "marabou", "all_timeout"]:
#             g = sub[sub["winner"] == v]
#             if g.empty:
#                 continue
#             ax.scatter(
#                 pd.to_numeric(g[xcomp], errors="coerce"),
#                 pd.to_numeric(g[yc], errors="coerce"),
#                 s=24 + 14 * np.log1p(pd.to_numeric(g["adv_gap"], errors="coerce").fillna(0.0)),
#                 color=VERIFIER_COLORS.get(v, "#9D9DA1"),
#                 alpha=0.80, edgecolors="white", linewidth=0.4
#             )
#         ax.set_title(f"{nice_component_name(xcomp)} vs {nice_component_name(yc)}", loc="left", pad=7)
#         ax.set_xlabel(nice_component_name(xcomp))
#         ax.set_ylabel(nice_component_name(yc))
#         for spine in ax.spines.values():
#             spine.set_visible(False)

#     for ax in axes.flat[len(ycomps):]:
#         ax.axis("off")

#     handles = winner_legend_handles(["abcrown", "neuralsat", "nnenum", "marabou", "all_timeout"])
#     fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.985), ncol=3)
#     fig.text(0.07, 0.02, f"Scatter-only outliers hidden across panels: {total_hidden}", fontsize=9.2, color="#5C6373")
#     fig.savefig(out_dir / "fig_verifier_planes_all_components_generic_v8.png", dpi=250)
#     fig.savefig(out_dir / "fig_verifier_planes_all_components_generic_v8.pdf")
#     plt.close(fig)


# def main():
#     parser = argparse.ArgumentParser(description="Generic-profile story figures with sample-size heatmaps.")
#     parser.add_argument("--merged-records", required=True)
#     parser.add_argument("--out-dir", required=True)
#     parser.add_argument("--construction-benchmark", default="sweep_all",
#                         help="Benchmark name for within-construction scorecards (default: sweep_all)")
#     parser.add_argument("--outlier-method", choices=["none", "quantile", "mad"], default="quantile")
#     parser.add_argument("--outlier-low-q", type=float, default=0.01)
#     parser.add_argument("--outlier-high-q", type=float, default=0.99)
#     args = parser.parse_args()

#     apply_style()
#     out_dir = Path(args.out_dir)
#     ensure_dir(out_dir)

#     records = load_wide_records(Path(args.merged_records))
#     records = robust_certification_subset(records)
#     verifiers = infer_verifiers(records)
#     components = available_candidate_components(records)

#     pooled_metrics, pooled_audit = compute_metrics_for_subset(
#         records, verifiers, components, benchmark_name="all",
#         outlier_method=args.outlier_method,
#         outlier_low_q=args.outlier_low_q,
#         outlier_high_q=args.outlier_high_q,
#     )
#     pooled_metrics.to_csv(out_dir / "table_pooled_component_metrics_generic_v8.csv", index=False)
#     pooled_audit.to_csv(out_dir / "table_pooled_outlier_audit_generic_v8.csv", index=False)

#     benchmark_rows = []
#     benchmark_audits = []
#     if "benchmark" in records.columns:
#         for bench, g in records.groupby("benchmark"):
#             m, a = compute_metrics_for_subset(
#                 g.copy(), verifiers, components, benchmark_name=str(bench),
#                 outlier_method=args.outlier_method,
#                 outlier_low_q=args.outlier_low_q,
#                 outlier_high_q=args.outlier_high_q,
#             )
#             benchmark_rows.append(m)
#             benchmark_audits.append(a)
#     by_benchmark = pd.concat(benchmark_rows, ignore_index=True) if benchmark_rows else pd.DataFrame(columns=pooled_metrics.columns)
#     by_benchmark.to_csv(out_dir / "table_benchmark_component_metrics_generic_v8.csv", index=False)
#     if benchmark_audits:
#         pd.concat(benchmark_audits, ignore_index=True).to_csv(out_dir / "table_benchmark_outlier_audit_generic_v8.csv", index=False)

#     subtitle_suffix = f"Outlier rule for the tables: {args.outlier_method}"
#     if args.outlier_method == "quantile":
#         subtitle_suffix += f" [{args.outlier_low_q:.2f}, {args.outlier_high_q:.2f}]"

#     plot_three_lens_grid(
#         pooled_metrics.assign(scope="all"),
#         row_key="scope",
#         title="All candidate components: pooled scorecards across benchmarks",
#         subtitle="Left = runtime correlation on robust {UNSAT, TIMEOUT} instances with censored timeouts. Middle = timeout-separation effect, defined as AUC - 0.5. Right = sample size n used for each cell. " + subtitle_suffix,
#         out_path_png=out_dir / "fig_all_components_pooled_scorecards_generic_v8.png",
#         out_path_pdf=out_dir / "fig_all_components_pooled_scorecards_generic_v8.pdf",
#     )

#     if not by_benchmark.empty:
#         plot_three_lens_grid(
#             by_benchmark,
#             row_key="benchmark",
#             title="Scorecards by benchmark",
#             subtitle="Each row recomputes runtime, timeout, and sample-size lenses within a single benchmark after applying the same component-wise outlier rule used for the pooled tables. " + subtitle_suffix,
#             out_path_png=out_dir / "fig_scorecards_by_benchmark_generic_v8.png",
#             out_path_pdf=out_dir / "fig_scorecards_by_benchmark_generic_v8.pdf",
#         )

#     if "benchmark" in records.columns and "construction" in records.columns:
#         sub_records = records[records["benchmark"] == args.construction_benchmark].copy()
#         construction_rows = []
#         construction_audits = []
#         for construction, g in sub_records.groupby("construction"):
#             m, a = compute_metrics_for_subset(
#                 g.copy(), verifiers, components, benchmark_name=str(construction),
#                 outlier_method=args.outlier_method,
#                 outlier_low_q=args.outlier_low_q,
#                 outlier_high_q=args.outlier_high_q,
#             )
#             m["construction"] = construction
#             a["construction"] = construction
#             construction_rows.append(m)
#             construction_audits.append(a)

#         if construction_rows:
#             by_construction = pd.concat(construction_rows, ignore_index=True)
#             by_construction.to_csv(out_dir / f"table_{args.construction_benchmark}_construction_component_metrics_generic_v8.csv", index=False)
#             if construction_audits:
#                 pd.concat(construction_audits, ignore_index=True).to_csv(out_dir / f"table_{args.construction_benchmark}_construction_outlier_audit_generic_v8.csv", index=False)

#             plot_three_lens_grid(
#                 by_construction.rename(columns={"construction": "group"}),
#                 row_key="group",
#                 title=f"Scorecards by construction within {args.construction_benchmark}",
#                 subtitle="Each row recomputes runtime, timeout, and sample-size lenses within one construction family. Sample-size cells also annotate NA reasons when the merged records contain <component>__na_reason columns. This makes it easy to distinguish unsupported cells from mathematically degenerate ones. " + subtitle_suffix,
#                 out_path_png=out_dir / f"fig_{args.construction_benchmark}_scorecards_by_construction_generic_v8.png",
#                 out_path_pdf=out_dir / f"fig_{args.construction_benchmark}_scorecards_by_construction_generic_v8.pdf",
#             )

#     plot_verifier_planes_all_components(records, out_dir, args.outlier_method, args.outlier_low_q, args.outlier_high_q)


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from profile_viz_utils_generic_v8 import (
    add_figure_title,
    apply_style,
    available_candidate_components,
    choose_timeout_cap,
    compute_metrics_for_subset,
    display_outlier_mask,
    ensure_dir,
    fastest_verifier,
    infer_verifiers,
    load_wide_records,
    nice_component_name,
    robust_certification_subset,
    safe_float,
    winner_legend_handles,
    VERIFIER_COLORS,
)


# ---------------------------------------------------------------------------
# Component direction helpers
# ---------------------------------------------------------------------------

# Canonical internal direction labels:
#   "higher_harder" = larger component values should correspond to harder verification
#   "higher_easier" = larger component values should correspond to easier verification
#   "unknown"       = no reliable hypothesis direction known
#
# Runtime correlations are interpreted against "higher runtime = harder".
# Therefore:
#   raw rho > 0 supports a higher_harder component
#   raw rho < 0 supports a higher_easier component
#
# The oriented heatmaps multiply raw correlations/effects by +1 or -1 so that:
#   oriented value > 0  means supportive
#   oriented value < 0  means contradictory


DEFAULT_COMPONENT_DIRECTIONS: Dict[str, str] = {
    # Canonical compact merged-record names from newer profile outputs.
    "M_hat_min": "higher_easier",
    "G_IBP": "higher_harder",
    "U": "higher_harder",
    "A_tau": "higher_harder",
    "d_eff": "higher_harder",

    # Relaxation / bound slack family: larger certified lower bound = easier.
    "ibp_margin_lb": "higher_easier",
    "margin_lb": "higher_easier",
    "crown_margin_lb": "higher_easier",
    "alpha_crown_margin_lb": "higher_easier",
    "relaxation_margin_lb": "higher_easier",

    # Raw / sampled margins: larger margin = easier.
    "margin_nominal": "higher_easier",
    "margin_sample_min": "higher_easier",
    "margin_sample_q01": "higher_easier",
    "margin_sample_q05": "higher_easier",
    "margin_sample_mean": "higher_easier",
    "first_order_margin": "higher_easier",
    "sampled_first_order_margin_ratio": "higher_easier",
    "margin_snr": "higher_easier",
    "ibp_margin_snr": "higher_easier",

    # Relaxation gap / looseness: larger gap = harder.
    "margin_gap": "higher_harder",
    "ibp_relative_gap": "higher_harder",
    "ibp_sample_gap": "higher_harder",
    "relaxation_gap": "higher_harder",
    "bound_gap": "higher_harder",

    # Width / overapproximation growth: larger = harder.
    "ibp_output_width_sum": "higher_harder",
    "ibp_output_width_mean": "higher_harder",
    "ibp_output_width_max": "higher_harder",
    "ibp_margin_width": "higher_harder",
    "ibp_width_log_slope": "higher_harder",
    "width_log_slope": "higher_harder",
    "overapprox_growth": "higher_harder",

    # Nonlinearity / branching ambiguity: larger = harder.
    "unstable_frac": "higher_harder",
    "unstable_count": "higher_harder",
    "relu_unstable_frac": "higher_harder",
    "activation_instability": "higher_harder",

    # Lipschitz / gradient size: larger = harder.
    "lip_empirical_max": "higher_harder",
    "lip_empirical_mean": "higher_harder",
    "grad_norm_mean": "higher_harder",
    "grad_norm_max": "higher_harder",
    "gradient_norm_mean": "higher_harder",
    "gradient_norm_max": "higher_harder",

    # Sensitivity geometry / effective dimension: larger = harder, but usually secondary.
    "grad_cov_effective_rank": "higher_harder",
    "effective_grad_dim_mean": "higher_harder",
    "effective_grad_dim_max": "higher_harder",
    "grad_sensitivity_concentration": "higher_harder",

    # Legacy / earlier profile names.
    "A_tau_local_log": "higher_harder",
    "A_tau": "higher_harder",
    "D_eff": "higher_harder",
    "N_hat": "higher_harder",
    "U_phi": "higher_harder",
    "R_p90": "higher_harder",
    "eta_delta": "higher_harder",
    "ηΔ": "higher_harder",

    # Volume / input-size quantities are often benchmark-confounded, but
    # larger search space is directionally harder if interpreted literally.
    "effective_box_log_volume": "higher_harder",
    "input_box_log_volume": "higher_harder",
}


def _normalize_direction(raw: object) -> str:
    """
    Normalize human-readable direction strings to:
        higher_harder, higher_easier, unknown
    """
    if raw is None:
        return "unknown"

    s = str(raw).strip().lower()
    if not s or s in {"nan", "none", "null", "unknown", "?"}:
        return "unknown"

    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)

    # Higher = harder variants.
    if s in {
        "higher harder",
        "higher is harder",
        "larger harder",
        "larger is harder",
        "increases hardness",
        "positive harder",
        "harder when higher",
        "high harder",
        "up harder",
        "↑ harder",
    }:
        return "higher_harder"

    # Higher = easier variants.
    if s in {
        "higher easier",
        "higher is easier",
        "larger easier",
        "larger is easier",
        "increases easiness",
        "positive easier",
        "easier when higher",
        "high easier",
        "up easier",
        "↑ easier",
    }:
        return "higher_easier"

    # Lower = harder is equivalent to higher = easier.
    if s in {
        "lower harder",
        "lower is harder",
        "smaller harder",
        "smaller is harder",
        "harder when lower",
        "low harder",
        "down harder",
        "↓ harder",
    }:
        return "higher_easier"

    # Lower = easier is equivalent to higher = harder.
    if s in {
        "lower easier",
        "lower is easier",
        "smaller easier",
        "smaller is easier",
        "easier when lower",
        "low easier",
        "down easier",
        "↓ easier",
    }:
        return "higher_harder"

    if "higher" in s and "harder" in s:
        return "higher_harder"
    if "higher" in s and "easier" in s:
        return "higher_easier"
    if "lower" in s and "harder" in s:
        return "higher_easier"
    if "lower" in s and "easier" in s:
        return "higher_harder"

    return "unknown"


def infer_component_direction(component: str) -> str:
    """
    Infer a default hypothesis direction from the component name.

    This is intentionally conservative. Explicit CSV directions override this.
    """
    if component in DEFAULT_COMPONENT_DIRECTIONS:
        return DEFAULT_COMPONENT_DIRECTIONS[component]

    c = component.lower()

    # Easier when larger.
    if "margin_lb" in c or "lower_bound" in c:
        return "higher_easier"
    if c.startswith("margin_") and not any(k in c for k in ["gap", "width"]):
        return "higher_easier"
    if "first_order_margin" in c:
        return "higher_easier"
    if "margin_snr" in c:
        return "higher_easier"

    # Harder when larger.
    hard_patterns = [
        "gap",
        "width",
        "unstable",
        "instability",
        "lip",
        "lipschitz",
        "grad_norm",
        "gradient_norm",
        "effective_rank",
        "effective_grad_dim",
        "sensitivity",
        "concentration",
        "volume",
        "log_slope",
        "overapprox",
        "branch",
        "split",
    ]
    if any(p in c for p in hard_patterns):
        return "higher_harder"

    # Common compact names from earlier profile experiments.
    if c in {"a_tau", "a_tau_local_log", "d_eff", "n_hat", "u_phi", "r_p90"}:
        return "higher_harder"

    return "unknown"


def load_component_directions_from_csv(path: Path) -> Dict[str, str]:
    """
    Load component direction overrides from a CSV.

    Accepted component columns:
        component, name, variable, feature

    Accepted direction columns:
        hypothesis_direction, direction, expected_direction, higher_means
    """
    if not path.exists():
        raise FileNotFoundError(f"Component direction CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return {}

    component_col = None
    for c in ["component", "name", "variable", "feature"]:
        if c in df.columns:
            component_col = c
            break

    direction_col = None
    for c in ["hypothesis_direction", "direction", "expected_direction", "higher_means"]:
        if c in df.columns:
            direction_col = c
            break

    if component_col is None or direction_col is None:
        raise ValueError(
            f"Could not find component/direction columns in {path}. "
            "Expected component/name/variable/feature and "
            "hypothesis_direction/direction/expected_direction/higher_means."
        )

    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        comp = str(row[component_col]).strip()
        if not comp or comp.lower() in {"nan", "none"}:
            continue
        direction = _normalize_direction(row[direction_col])
        if direction != "unknown":
            out[comp] = direction

    return out


def build_component_direction_map(
    components: List[str],
    direction_csv: Optional[Path] = None,
) -> Dict[str, str]:
    direction_map = {c: infer_component_direction(c) for c in components}

    if direction_csv is not None:
        overrides = load_component_directions_from_csv(direction_csv)
        for c in components:
            if c in overrides:
                direction_map[c] = overrides[c]

    return direction_map


def direction_suffix(direction: str) -> str:
    if direction == "higher_harder":
        return "↑ harder"
    if direction == "higher_easier":
        return "↑ easier"
    return "dir?"


def _title_component_label(label: str) -> str:
    """
    Paper-style capitalization for component labels.
    Keeps common acronyms/symbol-like tokens in their preferred form.
    """
    token_map = {
        "ibp": "IBP",
        "auc": "AUC",
        "lb": "LB",
        "snr": "SNR",
        "relu": "ReLU",
        "lip": "Lip",
        "a_tau": r"$A_\tau$",
        "d_eff": r"$D_{\mathrm{eff}}$",
        "n_hat": r"$\hat{N}$",
        "u_phi": r"$U_\phi$",
        "r_p90": r"$R_{90}$",
        "eta_delta": r"$\eta_\Delta$",
    }

    words = []
    for raw in label.replace("_", " ").split():
        key = raw.strip().lower()
        words.append(token_map.get(key, raw.capitalize()))
    return " ".join(words)



# Manual paper labels for the five components used in the final scorecard.
# These are applied before nice_component_name(), wrapping, or capitalization.
DISPLAY_COMPONENT_NAMES = {
    "M_hat_min": r"Min Margin" "\n" r"($\widehat{M}_{\min}$)",
    "G_IBP": r"IBP Relative Gap" "\n" r"($G_{IBP}$)",
    "U": r"Unstable Fraction" "\n" r"($U$)",
    "A_tau": r"Local Region" "\n" r"Complexity ($A_{\tau}$)",
    "d_eff": r"Effective" "\n" r"Grad Dim ($d_{eff}$)",

    "margin_sample_mean": r"Min Margin" "\n" r"($\widehat{M}_{\min}$)",
    "margin_sample_min": r"Min Margin" "\n" r"($\widehat{M}_{\min}$)",
    "effective_grad_dim_mean": r"Effective" "\n" r"Grad Dim ($d_{eff}$)",
    "effective_grad_dim_max": r"Effective" "\n" r"Grad Dim ($d_{eff}$)",
    "ibp_relative_gap": r"IBP Relative Gap" "\n" r"($G_{IBP}$)",
    "unstable_frac": r"Unstable Fraction" "\n" r"($U$)",
    "A_tau_local_log": r"Local Region" "\n" r"Complexity ($A_{\tau}$)",
    "a_tau_local_log": r"Local Region" "\n"r"Complexity ($A_{\tau}$)",
}

def short_component_label(component: str, max_chars: int = 28) -> str:
    """
    Compact, paper-style multi-line component label.
    Direction markers are intentionally omitted.
    """
    if component in DISPLAY_COMPONENT_NAMES:
        return DISPLAY_COMPONENT_NAMES[component]

    component_lower = component.lower()
    if component_lower in DISPLAY_COMPONENT_NAMES:
        return DISPLAY_COMPONENT_NAMES[component_lower]

    pretty = nice_component_name(component)

    replacements = {
        "sampled first order margin ratio": "First-Order\nMargin Ratio",
        "grad cov effective rank": "Grad Cov\nEff. Rank",
        "effective grad dim mean": "Eff. Grad\nDim Mean",
        "ibp output width sum": "IBP Output\nWidth Sum",
        "ibp output width mean": "IBP Output\nWidth Mean",
        "ibp output width max": "IBP Output\nWidth Max",
        "ibp width log slope": "IBP Width\nLog Slope",
        "ibp relative gap": "IBP Relative\nGap",
        "ibp sample gap": "IBP Sample\nGap",
        "ibp margin width": "IBP Margin\nWidth",
        "ibp margin lb": "IBP Margin\nLB",
        "ibp margin snr": "IBP Margin\nSNR",
        "unstable frac": "Unstable\nFrac",
        "margin nominal": "Nominal\nMargin",
        "margin sample min": "Sample Min\nMargin",
        "margin sample q01": "Q01 Sample\nMargin",
        "margin sample mean": "Sample Mean\nMargin",
        "lip empirical max": "Empirical\nLip Max",
        "a tau local log": r"Local Region Complexity ($A_{\tau}$)",
        "effective grad dim max": "Eff. Grad\nDim Max",
    }

    key = pretty.lower().replace("_", " ")
    if key in replacements:
        return replacements[key]

    paper_label = _title_component_label(pretty)
    wrapped = textwrap.wrap(paper_label, width=max_chars, break_long_words=False)
    if len(wrapped) <= 2:
        return "\n".join(wrapped)
    return "\n".join(wrapped[:2])


def component_axis_label(component: str, direction_map: Optional[Dict[str, str]] = None) -> str:
    """
    Return only the cleaned component name for paper figures.
    The direction map is accepted for backward compatibility but is not displayed.
    """
    return short_component_label(component)


def orientation_sign(direction: str) -> float:
    """
    Multiplicative sign for runtime/timeout correlations.

    Runtime higher = harder.
    Timeout-membership higher = harder.
    """
    if direction == "higher_harder":
        return 1.0
    if direction == "higher_easier":
        return -1.0
    return np.nan


def add_oriented_metric_columns(
    metrics: pd.DataFrame,
    direction_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Add oriented versions of the two correlation/effect columns.

    Positive oriented values mean agreement with the difficulty hypothesis.
    Negative oriented values mean contradiction.
    """
    metrics = metrics.copy()

    if "component" not in metrics.columns:
        return metrics

    metrics["component_direction"] = metrics["component"].map(
        lambda c: direction_map.get(c, "unknown")
    )
    metrics["orientation_sign"] = metrics["component_direction"].map(orientation_sign)

    for raw_col in ["spearman_runtime", "timeout_effect"]:
        if raw_col in metrics.columns:
            oriented_col = f"oriented_{raw_col}"
            metrics[oriented_col] = (
                pd.to_numeric(metrics[raw_col], errors="coerce")
                * pd.to_numeric(metrics["orientation_sign"], errors="coerce")
            )

    return metrics


def finite_vmax(series: pd.Series, fallback: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return fallback
    return max(fallback, float(np.nanmax(np.abs(vals))))


def finite_max(series: pd.Series, fallback: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return fallback
    return max(fallback, float(np.nanmax(vals)))


VERIFIER_DISPLAY_NAMES: Dict[str, str] = {
    "abcrown": r"$\alpha,\beta$-CROWN",
    "alpha_beta_crown": r"$\alpha,\beta$-CROWN",
    "alpha-beta-crown": r"$\alpha,\beta$-CROWN",
    "pyrat": "PyRAT",
    "neuralsat": "NeuralSAT",
    "nnenum": "nnenum",
    "marabou": "Marabou",
}


def verifier_display_label(verifier: str) -> str:
    return VERIFIER_DISPLAY_NAMES.get(str(verifier), str(verifier))


def centered_score_to_unit_interval(mat: np.ndarray) -> np.ndarray:
    """
    Convert a centered association score in [-1, 1] to a paper-friendly
    common 0--1 scale with 0.5 as neutral.
    """
    return 0.5 + 0.5 * mat


def auc_effect_to_auc(mat: np.ndarray) -> np.ndarray:
    """
    Convert AUC - 0.5 effects back to AUC, so 0.5 is neutral.
    """
    return mat + 0.5


# ---------------------------------------------------------------------------
# Matrix construction / annotation
# ---------------------------------------------------------------------------

def heatmap_matrix(metrics: pd.DataFrame, components: List[str], verifiers: List[str], value_col: str) -> np.ndarray:
    mat = np.full((len(components), len(verifiers)), np.nan)
    for i, c in enumerate(components):
        for j, v in enumerate(verifiers):
            row = metrics[(metrics["component"] == c) & (metrics["verifier"] == v)]
            if row.empty:
                continue
            val = safe_float(row.iloc[0][value_col])
            mat[i, j] = np.nan if val is None else val
    return mat


def count_matrix(metrics: pd.DataFrame, components: List[str], verifiers: List[str], value_col: str = "n_used") -> np.ndarray:
    mat = np.full((len(components), len(verifiers)), np.nan)
    for i, c in enumerate(components):
        for j, v in enumerate(verifiers):
            row = metrics[(metrics["component"] == c) & (metrics["verifier"] == v)]
            if row.empty:
                continue
            try:
                mat[i, j] = float(row.iloc[0][value_col])
            except Exception:
                mat[i, j] = np.nan
    return mat


def optional_count_matrix(metrics: pd.DataFrame, components: List[str], verifiers: List[str], value_col: str) -> np.ndarray:
    if value_col not in metrics.columns:
        return np.full((len(components), len(verifiers)), np.nan)
    return count_matrix(metrics, components, verifiers, value_col=value_col)


def annotate_metric_matrix(
    ax,
    mat: np.ndarray,
    threshold: float = 0.72,
    fontsize: float = 11.0,
    signed: bool = False,
):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "#1f2937"
            else:
                text = f"{val:+.2f}" if signed else f"{val:.2f}"
                if signed:
                    color = "white" if abs(val) >= threshold else "#1f2937"
                else:
                    color = "white" if val >= threshold or val <= (1.0 - threshold) else "#1f2937"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=color,
            )


def annotate_count_matrix(
    ax,
    mat: np.ndarray,
    unsupported: np.ndarray | None = None,
    math_na: np.ndarray | None = None,
    numerical_na: np.ndarray | None = None,
):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            lines = ["N/A" if np.isnan(val) else f"n={int(round(val))}"]
            extras = []
            if unsupported is not None:
                u = unsupported[i, j]
                if not np.isnan(u) and u > 0:
                    extras.append(f"U{int(round(u))}")
            if math_na is not None:
                m = math_na[i, j]
                if not np.isnan(m) and m > 0:
                    extras.append(f"M{int(round(m))}")
            if numerical_na is not None:
                n = numerical_na[i, j]
                if not np.isnan(n) and n > 0:
                    extras.append(f"E{int(round(n))}")
            if extras:
                lines.append(" / ".join(extras))
            text = "\n".join(lines)
            ax.text(j, i, text, ha="center", va="center", fontsize=7.6, color="#1f2937")


# ---------------------------------------------------------------------------
# Heatmap grids
# ---------------------------------------------------------------------------

def plot_three_lens_grid(
    metrics: pd.DataFrame,
    row_key: str,
    title: str,
    subtitle: str,
    out_path_png: Path,
    out_path_pdf: Path,
    direction_map: Optional[Dict[str, str]] = None,
    oriented: bool = False,
) -> None:
    """
    Paper-ready two-lens scorecard.

    Current layout choices:
      * no sample-size panel;
      * no overall title/subtitle and no per-panel titles;
      * original orientation is preserved: rows are components, columns are verifiers;
      * verifier names are paper labels;
      * separate colorbars for the runtime and timeout panels;
      * runtime values are shown as raw Spearman correlations in [-1, 1];
      * timeout values are shown as AUC rather than AUC - 0.5.
    """
    groups = list(dict.fromkeys(metrics[row_key]))
    verifiers = list(dict.fromkeys(metrics["verifier"]))
    components = list(dict.fromkeys(metrics["component"]))

    if oriented:
        runtime_col = "oriented_spearman_runtime"
        timeout_col = "oriented_timeout_effect"
    else:
        runtime_col = "spearman_runtime"
        timeout_col = "timeout_effect"

    if runtime_col not in metrics.columns or timeout_col not in metrics.columns:
        return

    n_components = len(components)
    n_verifiers = len(verifiers)

    # Keep the original rows/columns but make the canvas wide. The compressed
    # row height prevents the figure from becoming too tall, while larger
    # inter-panel spacing keeps axis labels and colorbars from colliding.
    fig_w = max(17.0, min(28.0, 2.70 * n_verifiers * 2.0 + 6.0))
    fig_h = max(7.8, min(12.0, 0.54 * n_components * len(groups) + 1.65))

    fig, axes = plt.subplots(
        len(groups),
        2,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.245,
        right=0.965,
        bottom=0.13,
        top=0.965,
        hspace=0.44,
        wspace=0.58,
    )

    cmap = "RdBu_r"

    for gi, group in enumerate(groups):
        sub = metrics[metrics[row_key] == group]

        # Original orientation: rows = components, columns = verifiers.
        # Runtime is shown as raw Spearman correlation; timeout is shown as AUC.
        runtime_spearman = heatmap_matrix(sub, components, verifiers, runtime_col)
        timeout_auc = auc_effect_to_auc(heatmap_matrix(sub, components, verifiers, timeout_col))

        panels = [
            {
                "mat": runtime_spearman,
                "cbar_label": "Runtime Spearman Correlation",
                "cbar_ticks": [-1.0, -0.5, 0.0, 0.5, 1.0],
                "vmin": -1.0,
                "vmax": 1.0,
                "signed": True,
                "annot_threshold": 0.45,
            },
            {
                "mat": timeout_auc,
                "cbar_label": "Timeout AUC",
                "cbar_ticks": [0.0, 0.25, 0.5, 0.75, 1.0],
                "vmin": 0.0,
                "vmax": 1.0,
                "signed": False,
                "annot_threshold": 0.72,
            },
        ]

        for ci, panel in enumerate(panels):
            ax = axes[gi, ci]
            mat = panel["mat"]
            im = ax.imshow(
                mat,
                cmap=cmap,
                vmin=panel["vmin"],
                vmax=panel["vmax"],
                aspect="auto",
            )

            ax.set_xticks(np.arange(n_verifiers))
            ax.set_xticklabels(
                [verifier_display_label(v) for v in verifiers],
                rotation=0,
                ha="center",
                fontsize=16.0,
                fontweight="bold",
            )
            ax.set_yticks(np.arange(n_components))

            # Show component labels on both panels. This is intentionally
            # redundant, but it makes each heatmap readable on its own.
            ax.set_yticklabels(
                [component_axis_label(c, direction_map) for c in components],
                fontsize=16.0,
                fontweight="bold",
                linespacing=0.90,
            )
            ylabel = "Component" if len(groups) == 1 else f"{group}\nComponent"
            ax.set_ylabel(ylabel, fontsize=22.0, fontweight="bold", labelpad=10)

            ax.set_xlabel("Verifier", fontsize=22.0, fontweight="bold", labelpad=10)
            ax.tick_params(axis="both", which="major", length=0, pad=7)

            for tick in ax.get_yticklabels():
                tick.set_linespacing(0.90)
                tick.set_va("center")

            for spine in ax.spines.values():
                spine.set_visible(False)

            annotate_metric_matrix(
                ax,
                mat,
                threshold=panel["annot_threshold"],
                fontsize=20.0,
                signed=panel["signed"],
            )

            # Separate colorbar per panel; no shared/common colorbar.
            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.030,
                ticks=panel["cbar_ticks"],
            )
            cbar.set_label(panel["cbar_label"], fontsize=22, fontweight="bold", labelpad=10)
            cbar.ax.tick_params(labelsize=11.5)
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontweight("bold")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.savefig(out_path_png, dpi=350, bbox_inches="tight",facecolor="white", edgecolor="white")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Scatter planes
# ---------------------------------------------------------------------------

def plot_verifier_planes_all_components(
    records: pd.DataFrame,
    out_dir: Path,
    outlier_method: str,
    low_q: float,
    high_q: float,
    direction_map: Optional[Dict[str, str]] = None,
) -> None:
    verifiers = infer_verifiers(records)
    timeout_cap = choose_timeout_cap(records, verifiers)
    components = available_candidate_components(records)

    xcomp = "margin_gap" if "margin_gap" in components else components[0]
    ycomps = [c for c in components if c != xcomp]
    if not ycomps:
        return

    ncols = 3
    nrows = int(np.ceil(len(ycomps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.6 * nrows), constrained_layout=False)
    axes = np.array(axes).reshape(nrows, ncols)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.88, hspace=0.42, wspace=0.28)
    add_figure_title(
        fig,
        "Explicit verifier-regime planes across all candidate components",
        (
            f"Scatter panels are trimmed for readability using {outlier_method}; "
            "axis labels include hypothesized difficulty direction. "
            "The scorecard tables use the same component-wise outlier rule."
        ),
        top=0.935,
    )

    total_hidden = 0

    for ax, yc in zip(axes.flat, ycomps):
        sub = records.copy()
        sub = sub[pd.to_numeric(sub[xcomp], errors="coerce").notna() & pd.to_numeric(sub[yc], errors="coerce").notna()].copy()
        keep_mask, _ = display_outlier_mask(sub, [xcomp, yc], method=outlier_method, low_q=low_q, high_q=high_q)
        total_hidden += int((~keep_mask).sum())
        sub = sub.loc[keep_mask].copy()

        winners, gaps = [], []
        for _, row in sub.iterrows():
            w, _, g = fastest_verifier(row, verifiers, timeout_cap)
            winners.append(w)
            gaps.append(0.0 if g is None else g)
        sub["winner"] = winners
        sub["adv_gap"] = gaps

        for v in ["abcrown", "neuralsat", "nnenum", "marabou", "all_timeout"]:
            g = sub[sub["winner"] == v]
            if g.empty:
                continue
            ax.scatter(
                pd.to_numeric(g[xcomp], errors="coerce"),
                pd.to_numeric(g[yc], errors="coerce"),
                s=24 + 14 * np.log1p(pd.to_numeric(g["adv_gap"], errors="coerce").fillna(0.0)),
                color=VERIFIER_COLORS.get(v, "#9D9DA1"),
                alpha=0.80,
                edgecolors="white",
                linewidth=0.4,
            )

        ax.set_title(
            f"{nice_component_name(xcomp)} vs {nice_component_name(yc)}",
            loc="left",
            pad=7,
        )
        ax.set_xlabel(component_axis_label(xcomp, direction_map))
        ax.set_ylabel(component_axis_label(yc, direction_map))

        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes.flat[len(ycomps):]:
        ax.axis("off")

    handles = winner_legend_handles(["abcrown", "neuralsat", "nnenum", "marabou", "all_timeout"])
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.985), ncol=3)
    fig.text(
        0.07,
        0.02,
        f"Scatter-only outliers hidden across panels: {total_hidden}",
        fontsize=9.2,
        color="#5C6373",
    )
    fig.savefig(out_dir / "fig_verifier_planes_all_components_generic_v8.png", dpi=250)
    fig.savefig(out_dir / "fig_verifier_planes_all_components_generic_v8.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# New merged-records compatibility
# ---------------------------------------------------------------------------

CANONICAL_TO_LEGACY_COMPONENTS: Dict[str, List[str]] = {
    # New canonical name -> legacy names used by older plotting utilities.
    "M_hat_min": ["margin_sample_min", "margin_sample_mean"],
    "G_IBP": ["ibp_relative_gap"],
    "U": ["unstable_frac"],
    "A_tau": ["A_tau_local_log"],
    "d_eff": ["effective_grad_dim_mean"],
}

FINAL_PROFILE_COMPONENTS: List[str] = [
    "margin_sample_min",
    "effective_grad_dim_mean",
    "ibp_relative_gap",
    "unstable_frac",
    "A_tau_local_log",
]


def load_merged_records_adaptive(path: Path) -> pd.DataFrame:
    """
    Load merged records from either the old loader-supported format or the newer
    JSON-list format, e.g.

        [
          {"id": "sweep_all/meap1", "G_IBP": ..., "U": ..., ...},
          ...
        ]

    Supported direct formats:
      * .json containing a list of records;
      * .json containing {"records": [...]} or {"data": [...]} etc.;
      * .jsonl / .ndjson with one JSON object per line;
      * .csv;
      * legacy loader fallback through load_wide_records(...).
    """
    path = Path(path)

    # Prefer explicit parsing for the newer plain JSON-list format.
    try:
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return pd.DataFrame(rows)

        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix == ".json" or suffix == "":
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)

            if isinstance(obj, list):
                return pd.DataFrame(obj)

            if isinstance(obj, dict):
                for key in ["records", "merged_records", "data", "rows", "instances"]:
                    if isinstance(obj.get(key), list):
                        return pd.DataFrame(obj[key])

                # Column-oriented JSON dict, if present.
                try:
                    df = pd.DataFrame(obj)
                    if len(df) > 0:
                        return df
                except Exception:
                    pass
    except Exception:
        # Fall through to the legacy repo loader.
        pass

    return load_wide_records(path)


def normalize_new_merged_record_columns(records: pd.DataFrame) -> pd.DataFrame:
    """
    Make newer merged-records files compatible with older plotting utilities.

    The new files may expose the final profile components as compact canonical
    columns:
        M_hat_min, G_IBP, U, A_tau, d_eff

    Older scripts/utilities often expect:
        margin_sample_min, ibp_relative_gap, unstable_frac,
        A_tau_local_log, effective_grad_dim_mean

    This function preserves the canonical columns and fills missing legacy alias
    columns from them. Existing legacy columns are not overwritten.
    """
    records = records.copy()

    for canonical, legacy_names in CANONICAL_TO_LEGACY_COMPONENTS.items():
        if canonical not in records.columns:
            continue
        for legacy in legacy_names:
            if legacy not in records.columns:
                records[legacy] = records[canonical]
            else:
                # Fill holes but do not overwrite existing values.
                records[legacy] = records[legacy].where(
                    pd.to_numeric(records[legacy], errors="coerce").notna(),
                    records[canonical],
                )

    # Some old utilities expect margin_sample_min specifically. If only the old
    # mean-name exists, mirror it into min as a fallback.
    if "margin_sample_min" not in records.columns and "margin_sample_mean" in records.columns:
        records["margin_sample_min"] = records["margin_sample_mean"]

    # Ensure benchmark/instance_id exist when only id like "sweep_all/meap1" is present.
    if "id" in records.columns:
        ids = records["id"].astype(str)
        if "benchmark" not in records.columns:
            records["benchmark"] = ids.map(lambda s: s.split("/", 1)[0] if "/" in s else None)
        if "instance_id" not in records.columns:
            records["instance_id"] = ids.map(lambda s: s.split("/", 1)[1] if "/" in s else s)

    return records


def robust_certification_subset_adaptive(records: pd.DataFrame) -> pd.DataFrame:
    """
    Use the repo's robust_certification_subset when it works, but do not let old
    assumptions accidentally drop all rows from the newer already-merged format.
    """
    try:
        filtered = robust_certification_subset(records)
        if filtered is not None and len(filtered) > 0:
            return filtered
    except Exception as e:
        print(f"Warning: robust_certification_subset failed; using all rows. Error: {e}")
    return records.copy()


def available_candidate_components_adaptive(records: pd.DataFrame) -> List[str]:
    """
    Prefer the final five VeriStress-GT profile components when present, after
    alias normalization. Fall back to profile_viz_utils_generic_v8's discovery.
    """
    final_present = []
    for c in FINAL_PROFILE_COMPONENTS:
        if c in records.columns and pd.to_numeric(records[c], errors="coerce").notna().sum() > 0:
            final_present.append(c)

    # If the new canonical format is present, this is almost always the desired
    # paper scorecard set and avoids accidentally including epsilon/Lc/etc.
    canonical_present = any(c in records.columns for c in CANONICAL_TO_LEGACY_COMPONENTS)
    if canonical_present and final_present:
        return final_present

    try:
        comps = available_candidate_components(records)
    except Exception:
        comps = []

    # De-duplicate canonical/legacy aliases if both happen to appear.
    alias_drop = set(CANONICAL_TO_LEGACY_COMPONENTS.keys())
    comps = [c for c in comps if c not in alias_drop]

    return comps or final_present


def enforce_white_background() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    })

def main():
    parser = argparse.ArgumentParser(description="Generic-profile story figures with sample-size heatmaps.")
    parser.add_argument("--merged-records", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--construction-benchmark",
        default="sweep_all",
        help="Benchmark name for within-construction scorecards (default: sweep_all)",
    )
    parser.add_argument("--outlier-method", choices=["none", "quantile", "mad"], default="quantile")
    parser.add_argument("--outlier-low-q", type=float, default=0.01)
    parser.add_argument("--outlier-high-q", type=float, default=0.99)
    parser.add_argument(
        "--component-directions-csv",
        default=None,
        help=(
            "Optional CSV with component direction overrides. "
            "Expected columns include component plus hypothesis_direction/direction. "
            "Examples: 'higher is harder', 'higher is easier'."
        ),
    )
    parser.add_argument(
        "--skip-oriented-heatmaps",
        action="store_true",
        help="Only make raw correlation/effect heatmaps; skip extra oriented heatmaps.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help=(
            "Optional component columns to use. Accepts either new canonical names "
            "(M_hat_min G_IBP U A_tau d_eff) or legacy names "
            "(margin_sample_min ibp_relative_gap unstable_frac A_tau_local_log effective_grad_dim_mean)."
        ),
    )
    args = parser.parse_args()

    apply_style()
    enforce_white_background()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    records = load_merged_records_adaptive(Path(args.merged_records))
    records = normalize_new_merged_record_columns(records)
    records = robust_certification_subset_adaptive(records)
    verifiers = infer_verifiers(records)
    components = available_candidate_components_adaptive(records)
    if args.components:
        requested = []
        alias_lookup = {
            "M_hat_min": "margin_sample_min",
            "G_IBP": "ibp_relative_gap",
            "U": "unstable_frac",
            "A_tau": "A_tau_local_log",
            "d_eff": "effective_grad_dim_mean",
        }
        for c in args.components:
            cc = alias_lookup.get(c, c)
            if cc in records.columns and cc not in requested:
                requested.append(cc)
        components = requested

    print(f"Rows used   : {len(records)}")
    print(f"Verifiers   : {verifiers}")
    print(f"Components  : {components}")

    direction_csv = Path(args.component_directions_csv) if args.component_directions_csv else None
    direction_map = build_component_direction_map(components, direction_csv=direction_csv)
    pd.DataFrame(
        [
            {
                "component": c,
                "component_pretty": nice_component_name(c),
                "direction": direction_map.get(c, "unknown"),
                "axis_label_suffix": direction_suffix(direction_map.get(c, "unknown")),
                "orientation_sign": orientation_sign(direction_map.get(c, "unknown")),
            }
            for c in components
        ]
    ).to_csv(out_dir / "table_component_directions_generic_v8.csv", index=False)

    pooled_metrics, pooled_audit = compute_metrics_for_subset(
        records,
        verifiers,
        components,
        benchmark_name="all",
        outlier_method=args.outlier_method,
        outlier_low_q=args.outlier_low_q,
        outlier_high_q=args.outlier_high_q,
    )
    pooled_metrics = add_oriented_metric_columns(pooled_metrics, direction_map)
    pooled_metrics.to_csv(out_dir / "table_pooled_component_metrics_generic_v8.csv", index=False)
    pooled_audit.to_csv(out_dir / "table_pooled_outlier_audit_generic_v8.csv", index=False)

    benchmark_rows = []
    benchmark_audits = []
    if "benchmark" in records.columns:
        for bench, g in records.groupby("benchmark"):
            m, a = compute_metrics_for_subset(
                g.copy(),
                verifiers,
                components,
                benchmark_name=str(bench),
                outlier_method=args.outlier_method,
                outlier_low_q=args.outlier_low_q,
                outlier_high_q=args.outlier_high_q,
            )
            m = add_oriented_metric_columns(m, direction_map)
            benchmark_rows.append(m)
            benchmark_audits.append(a)

    by_benchmark = (
        pd.concat(benchmark_rows, ignore_index=True)
        if benchmark_rows
        else pd.DataFrame(columns=pooled_metrics.columns)
    )
    by_benchmark.to_csv(out_dir / "table_benchmark_component_metrics_generic_v8.csv", index=False)

    if benchmark_audits:
        pd.concat(benchmark_audits, ignore_index=True).to_csv(
            out_dir / "table_benchmark_outlier_audit_generic_v8.csv",
            index=False,
        )

    subtitle_suffix = f"Outlier rule for the tables: {args.outlier_method}"
    if args.outlier_method == "quantile":
        subtitle_suffix += f" [{args.outlier_low_q:.2f}, {args.outlier_high_q:.2f}]"

    raw_subtitle = (
        "Left = raw runtime correlation on robust {UNSAT, TIMEOUT} instances with censored timeouts. "
        "Middle = raw timeout-separation effect, defined as AUC - 0.5. "
        "Right = sample size n used for each cell. "
        + subtitle_suffix
    )

    oriented_subtitle = (
        "Left = oriented runtime correlation; middle = oriented timeout-separation effect. "
        "Positive means the component agrees with its hypothesized difficulty direction, "
        "negative means contradictory. Right = sample size n used for each cell. "
        + subtitle_suffix
    )

    plot_three_lens_grid(
        pooled_metrics.assign(scope="all"),
        row_key="scope",
        title="All candidate components: pooled scorecards across benchmarks",
        subtitle=raw_subtitle,
        out_path_png=out_dir / "fig_all_components_pooled_scorecards_generic_v8.png",
        out_path_pdf=out_dir / "fig_all_components_pooled_scorecards_generic_v8.pdf",
        direction_map=direction_map,
        oriented=False,
    )

    if not args.skip_oriented_heatmaps:
        plot_three_lens_grid(
            pooled_metrics.assign(scope="all"),
            row_key="scope",
            title="All candidate components: pooled oriented scorecards across benchmarks",
            subtitle=oriented_subtitle,
            out_path_png=out_dir / "fig_all_components_pooled_oriented_scorecards_generic_v8.png",
            out_path_pdf=out_dir / "fig_all_components_pooled_oriented_scorecards_generic_v8.pdf",
            direction_map=direction_map,
            oriented=True,
        )

    if not by_benchmark.empty:
        plot_three_lens_grid(
            by_benchmark,
            row_key="benchmark",
            title="Scorecards by benchmark",
            subtitle=(
                "Each row recomputes raw runtime, timeout, and sample-size lenses within a single benchmark "
                "after applying the same component-wise outlier rule used for the pooled tables. "
                + subtitle_suffix
            ),
            out_path_png=out_dir / "fig_scorecards_by_benchmark_generic_v8.png",
            out_path_pdf=out_dir / "fig_scorecards_by_benchmark_generic_v8.pdf",
            direction_map=direction_map,
            oriented=False,
        )

        if not args.skip_oriented_heatmaps:
            plot_three_lens_grid(
                by_benchmark,
                row_key="benchmark",
                title="Oriented scorecards by benchmark",
                subtitle=(
                    "Each row recomputes oriented runtime and timeout lenses within a single benchmark. "
                    "Positive means agreement with the hypothesized difficulty direction; negative means contradiction. "
                    + subtitle_suffix
                ),
                out_path_png=out_dir / "fig_scorecards_by_benchmark_oriented_generic_v8.png",
                out_path_pdf=out_dir / "fig_scorecards_by_benchmark_oriented_generic_v8.pdf",
                direction_map=direction_map,
                oriented=True,
            )

    if "benchmark" in records.columns and "construction" in records.columns:
        sub_records = records[records["benchmark"] == args.construction_benchmark].copy()
        construction_rows = []
        construction_audits = []

        for construction, g in sub_records.groupby("construction"):
            m, a = compute_metrics_for_subset(
                g.copy(),
                verifiers,
                components,
                benchmark_name=str(construction),
                outlier_method=args.outlier_method,
                outlier_low_q=args.outlier_low_q,
                outlier_high_q=args.outlier_high_q,
            )
            m = add_oriented_metric_columns(m, direction_map)
            m["construction"] = construction
            a["construction"] = construction
            construction_rows.append(m)
            construction_audits.append(a)

        if construction_rows:
            by_construction = pd.concat(construction_rows, ignore_index=True)
            by_construction.to_csv(
                out_dir / f"table_{args.construction_benchmark}_construction_component_metrics_generic_v8.csv",
                index=False,
            )
            if construction_audits:
                pd.concat(construction_audits, ignore_index=True).to_csv(
                    out_dir / f"table_{args.construction_benchmark}_construction_outlier_audit_generic_v8.csv",
                    index=False,
                )

            plot_three_lens_grid(
                by_construction.rename(columns={"construction": "group"}),
                row_key="group",
                title=f"Scorecards by construction within {args.construction_benchmark}",
                subtitle=(
                    "Each row recomputes raw runtime, timeout, and sample-size lenses within one construction family. "
                    "Sample-size cells also annotate NA reasons when the merged records contain <component>__na_reason columns. "
                    "This makes it easy to distinguish unsupported cells from mathematically degenerate ones. "
                    + subtitle_suffix
                ),
                out_path_png=out_dir / f"fig_{args.construction_benchmark}_scorecards_by_construction_generic_v8.png",
                out_path_pdf=out_dir / f"fig_{args.construction_benchmark}_scorecards_by_construction_generic_v8.pdf",
                direction_map=direction_map,
                oriented=False,
            )

            if not args.skip_oriented_heatmaps:
                plot_three_lens_grid(
                    by_construction.rename(columns={"construction": "group"}),
                    row_key="group",
                    title=f"Oriented scorecards by construction within {args.construction_benchmark}",
                    subtitle=(
                        "Each row recomputes oriented runtime and timeout lenses within one construction family. "
                        "Positive means agreement with the hypothesized difficulty direction; negative means contradiction. "
                        + subtitle_suffix
                    ),
                    out_path_png=out_dir / f"fig_{args.construction_benchmark}_scorecards_by_construction_oriented_generic_v8.png",
                    out_path_pdf=out_dir / f"fig_{args.construction_benchmark}_scorecards_by_construction_oriented_generic_v8.pdf",
                    direction_map=direction_map,
                    oriented=True,
                )

    plot_verifier_planes_all_components(
        records,
        out_dir,
        args.outlier_method,
        args.outlier_low_q,
        args.outlier_high_q,
        direction_map=direction_map,
    )


if __name__ == "__main__":
    main()