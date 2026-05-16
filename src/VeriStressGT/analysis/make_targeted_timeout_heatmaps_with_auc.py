# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Dict, List, Optional, Sequence, Tuple

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# # -----------------------------------------------------------------------------
# # Component aliases for old and new merged-record formats
# # -----------------------------------------------------------------------------

# COMPONENT_ALIASES: Dict[str, List[str]] = {
#     "M_hat_min": ["M_hat_min", "margin_sample_min", "sample_margin_min", "margin_sample_mean"],
#     "A_tau": ["A_tau", "A_tau_local_log", "a_tau_local_log", "local_affine_cover_log"],
#     "G_IBP": ["G_IBP", "ibp_relative_gap", "g_ibp"],
#     "U": ["U", "unstable_frac", "U_phi", "u_phi"],
#     "d_eff": ["d_eff", "effective_grad_dim_mean", "effective_grad_dim", "D_eff"],
# }

# # For the two pairwise timeout heatmaps.
# PAIR_DISPLAY_LABELS: Dict[str, str] = {
#     "A_tau": r"$A_{\tau}$ Quantile",
#     "G_IBP": r"$G_{\mathrm{IBP}}$ Quantile",
#     "U": r"$U$ Quantile",
#     "d_eff": r"$d_{\mathrm{eff}}$ Quantile",
# }

# # # For the oriented Timeout AUC scorecard.
# # COMPONENT_DISPLAY_NAMES: Dict[str, str] = {
# #     "M_hat_min": r"Min Margin" "\n" r"($\widehat{M}_{\min}$)",
# #     "margin_sample_min": r"Min Margin" "\n" r"($\widehat{M}_{\min}$)",
# #     "sample_margin_min": r"Min Margin" "\n" r"($\widehat{M}_{\min}$)",
# #     "margin_sample_mean": r"Min Margin" "\n" r"($\widehat{M}_{\min}$)",
# #     "d_eff": r"Effective" "\n" r"Grad Dim ($d_{\mathrm{eff}}$)",
# #     "effective_grad_dim_mean": r"Effective" "\n" r"Grad Dim ($d_{\mathrm{eff}}$)",
# #     "effective_grad_dim": r"Effective" "\n" r"Grad Dim ($d_{\mathrm{eff}}$)",
# #     "D_eff": r"Effective" "\n" r"Grad Dim ($d_{\mathrm{eff}}$)",
# #     "G_IBP": r"IBP Relative Gap" "\n" r"($G_{\mathrm{IBP}}$)",
# #     "ibp_relative_gap": r"IBP Relative Gap" "\n" r"($G_{\mathrm{IBP}}$)",
# #     "U": r"Unstable Fraction" "\n" r"($U$)",
# #     "unstable_frac": r"Unstable Fraction" "\n" r"($U$)",
# #     "A_tau": r"Local Region" "\n" r"Complexity ($A_{\tau}$)",
# #     "A_tau_local_log": r"Local Region" "\n" r"Complexity ($A_{\tau}$)",
# # }

# COMPONENT_DISPLAY_NAMES: Dict[str, str] = {
#     "M_hat_min": "$\widehat{M}_{\min}$",
#     "margin_sample_min": r"$\widehat{M}_{\min}$",
#     "sample_margin_min": r"$\widehat{M}_{\min}$",
#     "margin_sample_mean":r"$\widehat{M}_{\min}$",
#     "d_eff": r"$d_{\mathrm{eff}}$",
#     "effective_grad_dim_mean": r"$d_{\mathrm{eff}}$",
#     "effective_grad_dim": r"$d_{\mathrm{eff}}$",
#     "D_eff": r"$d_{\mathrm{eff}}$",
#     "G_IBP": r"$G_{\mathrm{IBP}}$",
#     "ibp_relative_gap": r"$G_{\mathrm{IBP}}$",
#     "U":  r"$U$",
#     "unstable_frac":r"$U$",
#     "A_tau": r"$A_{\tau}$",
#     "A_tau_local_log": r"$A_{\tau}$",
# }

# VERIFIER_DISPLAY_NAMES: Dict[str, str] = {
#     "abcrown": r"$\alpha,\beta$-CROWN",
#     "alpha_beta_crown": r"$\alpha,\beta$-CROWN",
#     "alpha-beta-crown": r"$\alpha,\beta$-CROWN",
#     "marabou": "Marabou",
#     "neuralsat": "NeuralSAT",
#     "nnenum": "nnenum",
#     "pyrat": "PyRAT",
# }

# # Higher_harder means larger component values should be associated with timeout.
# # Higher_easier means smaller component values should be associated with timeout.
# COMPONENT_DIRECTIONS: Dict[str, str] = {
#     "M_hat_min": "higher_easier",
#     "margin_sample_min": "higher_easier",
#     "sample_margin_min": "higher_easier",
#     "margin_sample_mean": "higher_easier",
#     "d_eff": "higher_harder",
#     "effective_grad_dim_mean": "higher_harder",
#     "effective_grad_dim": "higher_harder",
#     "D_eff": "higher_harder",
#     "G_IBP": "higher_harder",
#     "ibp_relative_gap": "higher_harder",
#     "U": "higher_harder",
#     "unstable_frac": "higher_harder",
#     "A_tau": "higher_harder",
#     "A_tau_local_log": "higher_harder",
# }


# # -----------------------------------------------------------------------------
# # Shared figure style
# # -----------------------------------------------------------------------------

# # Keep all three outputs visually consistent. The AUC scorecard has a different
# # data shape than the pairwise quantile heatmaps, but the canvas size and label
# # styling are intentionally identical.
# FIG_SIZE = (12.0, 6)
# AXIS_LABEL_FONTSIZE = 22.0
# AXIS_LABEL_FONTWEIGHT = "bold"
# TICK_LABEL_FONTSIZE = 15.0
# TICK_LABEL_FONTWEIGHT = "bold"
# ANNOT_FONTSIZE = 22.0
# ANNOT_FONTWEIGHT = "bold"
# CBAR_LABEL_FONTSIZE = 18.0
# CBAR_TICK_FONTSIZE = 18.0



# # -----------------------------------------------------------------------------
# # Loading / cleaning
# # -----------------------------------------------------------------------------

# def load_records(path: Path) -> pd.DataFrame:
#     """
#     Load merged records from:
#       - JSON list of dicts
#       - JSON dict with records/data/rows/items/instances
#       - JSONL / NDJSON
#       - CSV
#     """
#     suffix = path.suffix.lower()

#     if suffix == ".csv":
#         return pd.read_csv(path)

#     if suffix in {".jsonl", ".ndjson"}:
#         return pd.read_json(path, lines=True)

#     if suffix == ".json":
#         text = path.read_text(encoding="utf-8")
#         data = json.loads(text)

#         if isinstance(data, list):
#             return pd.DataFrame(data)

#         if isinstance(data, dict):
#             for key in ["records", "data", "rows", "items", "instances"]:
#                 if key in data and isinstance(data[key], list):
#                     return pd.DataFrame(data[key])

#             # Last resort: dict of records.
#             if all(isinstance(v, dict) for v in data.values()):
#                 return pd.DataFrame(list(data.values()))

#         raise ValueError(
#             f"Unsupported JSON structure in {path}. Expected a list of records "
#             "or a dict containing records/data/rows/items/instances."
#         )

#     # Fallback: try JSONL, then CSV.
#     try:
#         return pd.read_json(path, lines=True)
#     except Exception:
#         return pd.read_csv(path)


# def safe_float(x: object) -> Optional[float]:
#     try:
#         if x is None:
#             return None
#         v = float(x)
#         if not np.isfinite(v):
#             return None
#         return v
#     except Exception:
#         return None


# def resolve_component_column(df: pd.DataFrame, logical_name: str) -> str:
#     for col in COMPONENT_ALIASES.get(logical_name, [logical_name]):
#         if col in df.columns:
#             return col
#     raise KeyError(
#         f"Could not find a column for {logical_name}. Tried: "
#         f"{COMPONENT_ALIASES.get(logical_name, [logical_name])}. "
#         f"Available columns include: {list(df.columns)[:50]}"
#     )


# def infer_verifiers(df: pd.DataFrame) -> List[str]:
#     """
#     Infer verifier prefixes from columns like:
#       abcrown_outcome, marabou_outcome, neuralsat_outcome, ...
#     """
#     verifiers = []
#     for col in df.columns:
#         if col.endswith("_outcome"):
#             prefix = col[: -len("_outcome")]
#             if prefix:
#                 verifiers.append(prefix)

#     preferred_order = ["abcrown", "marabou", "neuralsat", "nnenum", "pyrat"]
#     ordered = [v for v in preferred_order if v in verifiers]
#     ordered += sorted(v for v in verifiers if v not in ordered)
#     return ordered


# def normalize_outcome(x: object) -> str:
#     """
#     Normalize outcome/status to solved, timeout, error, missing, or other.
#     """
#     s = str(x).strip().lower()
#     if s in {"solved", "unsat", "safe", "verified", "true"}:
#         return "solved"
#     if s in {"timeout", "timed_out", "time_limit"}:
#         return "timeout"
#     if s in {"error", "failed", "exception"}:
#         return "error"
#     if s in {"missing", "nan", "none", "null", ""}:
#         return "missing"
#     return "other"


# def verifier_display_label(verifier: str) -> str:
#     return VERIFIER_DISPLAY_NAMES.get(str(verifier), str(verifier))


# def component_display_label(logical_component: str, resolved_col: str) -> str:
#     return (
#         COMPONENT_DISPLAY_NAMES.get(logical_component)
#         or COMPONENT_DISPLAY_NAMES.get(resolved_col)
#         or str(logical_component).replace("_", " ")
#     )


# def component_direction(logical_component: str, resolved_col: str) -> str:
#     return (
#         COMPONENT_DIRECTIONS.get(logical_component)
#         or COMPONENT_DIRECTIONS.get(resolved_col)
#         or "unknown"
#     )


# def orientation_sign(direction: str) -> float:
#     if direction == "higher_harder":
#         return 1.0
#     if direction == "higher_easier":
#         return -1.0
#     return np.nan


# # -----------------------------------------------------------------------------
# # Outlier handling
# # -----------------------------------------------------------------------------

# def display_outlier_mask(
#     df: pd.DataFrame,
#     cols: Sequence[str],
#     method: str,
#     low_q: float,
#     high_q: float,
# ) -> np.ndarray:
#     """
#     Match the common plotting behavior: trim display/metric outliers by component.
#     """
#     keep = np.ones(len(df), dtype=bool)

#     if method == "none":
#         for col in cols:
#             vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
#             keep &= np.isfinite(vals)
#         return keep

#     for col in cols:
#         vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
#         finite = np.isfinite(vals)
#         keep &= finite

#         if method == "quantile":
#             if finite.sum() < 3:
#                 continue
#             lo = float(np.nanquantile(vals[finite], low_q))
#             hi = float(np.nanquantile(vals[finite], high_q))
#             keep &= vals >= lo
#             keep &= vals <= hi

#         elif method == "mad":
#             if finite.sum() < 3:
#                 continue
#             med = float(np.nanmedian(vals[finite]))
#             mad = float(np.nanmedian(np.abs(vals[finite] - med)))
#             if mad <= 1e-12:
#                 continue
#             lo = med - 8.0 * mad
#             hi = med + 8.0 * mad
#             keep &= vals >= lo
#             keep &= vals <= hi

#         else:
#             raise ValueError(f"Unknown outlier method: {method}")

#     return keep


# # -----------------------------------------------------------------------------
# # Pairwise timeout heatmaps
# # -----------------------------------------------------------------------------

# def prepare_pooled_timeout_df(
#     records: pd.DataFrame,
#     verifiers: Sequence[str],
#     xcol: str,
#     ycol: str,
#     outlier_method: str,
#     outlier_low_q: float,
#     outlier_high_q: float,
# ) -> pd.DataFrame:
#     """
#     Build pooled verifier-instance timeout observations.

#     One row = one (instance, verifier) pair.
#     Keeps only solved and timeout outcomes.
#     """
#     rows = []

#     for verifier in verifiers:
#         outcome_col = f"{verifier}_outcome"
#         status_col = f"{verifier}_status_bucket"

#         if outcome_col not in records.columns and status_col not in records.columns:
#             continue

#         for _, r in records.iterrows():
#             raw_outcome = r.get(outcome_col, r.get(status_col, "missing"))
#             out = normalize_outcome(raw_outcome)

#             if out not in {"solved", "timeout"}:
#                 continue

#             x = safe_float(r.get(xcol))
#             y = safe_float(r.get(ycol))
#             if x is None or y is None:
#                 continue

#             rows.append({
#                 "x": x,
#                 "y": y,
#                 "timeout": 1 if out == "timeout" else 0,
#                 "verifier": verifier,
#                 "benchmark": r.get("benchmark", None),
#                 "id": r.get("id", r.get("instance_id", None)),
#             })

#     df = pd.DataFrame(rows)
#     if df.empty:
#         return df

#     if outlier_method != "none":
#         keep = display_outlier_mask(
#             df,
#             ["x", "y"],
#             method=outlier_method,
#             low_q=outlier_low_q,
#             high_q=outlier_high_q,
#         )
#         df = df.loc[keep].copy()

#     return df.reset_index(drop=True)


# def two_dim_binned_summary(
#     df: pd.DataFrame,
#     x: np.ndarray,
#     y: np.ndarray,
#     value_col: str,
#     bins: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Mean value_col in two-dimensional quantile bins.

#     B[j, i] = mean value in x-bin i and y-bin j.
#     N[j, i] = sample count in x-bin i and y-bin j.
#     """
#     x = np.asarray(x, float)
#     y = np.asarray(y, float)

#     finite = np.isfinite(x) & np.isfinite(y)
#     if not np.any(finite):
#         return np.full((1, 1), np.nan), np.zeros((1, 1), dtype=int)

#     df = df.loc[finite].copy()
#     x = x[finite]
#     y = y[finite]

#     qx = np.asarray(pd.qcut(x, q=bins, labels=False, duplicates="drop"), dtype=float)
#     qy = np.asarray(pd.qcut(y, q=bins, labels=False, duplicates="drop"), dtype=float)

#     valid = np.isfinite(qx) & np.isfinite(qy)
#     if not np.any(valid):
#         return np.full((1, 1), np.nan), np.zeros((1, 1), dtype=int)

#     qx = qx[valid].astype(int)
#     qy = qy[valid].astype(int)
#     df = df.loc[valid].copy()

#     nx = int(qx.max()) + 1
#     ny = int(qy.max()) + 1
#     B = np.full((ny, nx), np.nan)
#     N = np.zeros((ny, nx), dtype=int)

#     values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(float)

#     for i in range(nx):
#         for j in range(ny):
#             m = (qx == i) & (qy == j)
#             if int(m.sum()) > 0:
#                 B[j, i] = float(np.nanmean(values[m]))
#                 N[j, i] = int(m.sum())

#     return B, N


# def annotate_heatmap(ax, B: np.ndarray, N: np.ndarray, fontsize: float = 8.5) -> None:
#     baseline = np.nanmean(B)

#     for jj in range(B.shape[0]):
#         for ii in range(B.shape[1]):
#             if np.isnan(B[jj, ii]):
#                 text = "N/A"
#                 color = "#1f2937"
#             else:
#                 text = f"{B[jj, ii]:.2f}\n(n={N[jj, ii]})"
#                 color = "white"# if B[jj, ii] > baseline else "#1f2937"

#             ax.text(
#                 ii,
#                 jj,
#                 text,
#                 ha="center",
#                 va="center",
#                 fontsize=fontsize,
#                 fontweight=ANNOT_FONTWEIGHT,
#                 color=color,
#             )


# def plot_one_pair_heatmap(
#     records: pd.DataFrame,
#     verifiers: Sequence[str],
#     xlogical: str,
#     ylogical: str,
#     out_dir: Path,
#     bins: int,
#     outlier_method: str,
#     outlier_low_q: float,
#     outlier_high_q: float,
#     annotate: bool,
#     write_debug_csv: bool,
# ) -> None:
#     xcol = resolve_component_column(records, xlogical)
#     ycol = resolve_component_column(records, ylogical)

#     df = prepare_pooled_timeout_df(
#         records,
#         verifiers,
#         xcol,
#         ycol,
#         outlier_method,
#         outlier_low_q,
#         outlier_high_q,
#     )

#     if df.empty:
#         print(f"Skipping {xlogical}_by_{ylogical}: no solved/timeout observations.")
#         return

#     x = pd.to_numeric(df["x"], errors="coerce").to_numpy(float)
#     y = pd.to_numeric(df["y"], errors="coerce").to_numpy(float)
#     B, N = two_dim_binned_summary(df, x, y, "timeout", bins=bins)

#     if write_debug_csv:
#         rows = []
#         for jj in range(B.shape[0]):
#             for ii in range(B.shape[1]):
#                 rows.append({
#                     "x_logical": xlogical,
#                     "y_logical": ylogical,
#                     "x_column": xcol,
#                     "y_column": ycol,
#                     "x_quantile": ii + 1,
#                     "y_quantile": jj + 1,
#                     "timeout_rate": B[jj, ii],
#                     "n": N[jj, ii],
#                 })
#         pd.DataFrame(rows).to_csv(
#             out_dir / f"debug_timeout_heatmap_{xlogical}_by_{ylogical}_bins.csv",
#             index=False,
#         )

#     fig, ax = plt.subplots(figsize=(10.0, 6.0), constrained_layout=False)
#     fig.subplots_adjust(left=0.18, right=0.88, bottom=0.16, top=0.96)

#     im = ax.imshow(
#         B,
#         origin="lower",
#         aspect="auto",
#         cmap="viridis",
#         vmin=0.0,
#         vmax=1.0,
#     )

#     # No plot titles.
#     ax.set_xlabel(
#         PAIR_DISPLAY_LABELS.get(xlogical, f"{xlogical} Quantile"),
#         fontsize=AXIS_LABEL_FONTSIZE,
#         fontweight=AXIS_LABEL_FONTWEIGHT,
#     )
#     ax.set_ylabel(
#         PAIR_DISPLAY_LABELS.get(ylogical, f"{ylogical} Quantile"),
#         fontsize=AXIS_LABEL_FONTSIZE,
#         fontweight=AXIS_LABEL_FONTWEIGHT,
#     )

#     ax.set_xticks(range(B.shape[1]))
#     ax.set_xticklabels([f"Q{i + 1}" for i in range(B.shape[1])])
#     ax.set_yticks(range(B.shape[0]))
#     ax.set_yticklabels([f"Q{i + 1}" for i in range(B.shape[0])])

#     if annotate:
#         annotate_heatmap(ax, B, N,fontsize=16)

#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.025)
#     cbar.set_label("Timeout Rate", fontsize=CBAR_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT)
#     cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
#     for tick in cbar.ax.get_yticklabels():
#         tick.set_fontweight(TICK_LABEL_FONTWEIGHT)

#     stem = f"fig_timeout_heatmap_{xlogical}_by_{ylogical}"
#     fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
#     fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
#     plt.close(fig)

#     print(f"Wrote: {out_dir / (stem + '.png')}")
#     print(f"Wrote: {out_dir / (stem + '.pdf')}")


# # -----------------------------------------------------------------------------
# # Oriented Timeout AUC scorecard heatmap
# # -----------------------------------------------------------------------------

# def rankdata_average(x: np.ndarray) -> np.ndarray:
#     order = np.argsort(x, kind="mergesort")
#     ranks = np.empty(len(x), float)

#     i = 0
#     while i < len(x):
#         j = i + 1
#         while j < len(x) and x[order[j]] == x[order[i]]:
#             j += 1
#         ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
#         i = j

#     return ranks


# def roc_auc_from_scores(scores: Sequence[float], labels01: Sequence[int]) -> Optional[float]:
#     s = np.asarray(scores, float)
#     y = np.asarray(labels01, int)

#     mask = np.isfinite(s) & np.isfinite(y)
#     s = s[mask]
#     y = y[mask]

#     pos = y == 1
#     neg = y == 0
#     n_pos = int(pos.sum())
#     n_neg = int(neg.sum())

#     if n_pos == 0 or n_neg == 0:
#         return None

#     ranks = rankdata_average(s)
#     return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# def prepare_component_timeout_df(
#     records: pd.DataFrame,
#     verifier: str,
#     component_col: str,
#     outlier_method: str,
#     outlier_low_q: float,
#     outlier_high_q: float,
# ) -> pd.DataFrame:
#     rows = []

#     outcome_col = f"{verifier}_outcome"
#     status_col = f"{verifier}_status_bucket"

#     if outcome_col not in records.columns and status_col not in records.columns:
#         return pd.DataFrame()

#     for _, r in records.iterrows():
#         raw_outcome = r.get(outcome_col, r.get(status_col, "missing"))
#         out = normalize_outcome(raw_outcome)
#         if out not in {"solved", "timeout"}:
#             continue

#         value = safe_float(r.get(component_col))
#         if value is None:
#             continue

#         rows.append({
#             "component_value": value,
#             "timeout": 1 if out == "timeout" else 0,
#             "verifier": verifier,
#             "benchmark": r.get("benchmark", None),
#             "id": r.get("id", r.get("instance_id", None)),
#         })

#     df = pd.DataFrame(rows)
#     if df.empty:
#         return df

#     if outlier_method != "none":
#         keep = display_outlier_mask(
#             df,
#             ["component_value"],
#             method=outlier_method,
#             low_q=outlier_low_q,
#             high_q=outlier_high_q,
#         )
#         df = df.loc[keep].copy()

#     return df.reset_index(drop=True)


# def compute_oriented_timeout_auc_table(
#     records: pd.DataFrame,
#     verifiers: Sequence[str],
#     logical_components: Sequence[str],
#     outlier_method: str,
#     outlier_low_q: float,
#     outlier_high_q: float,
# ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
#     """
#     Compute the timeout AUC panel from the oriented scorecard.

#     For a component with direction:
#       higher_harder: oriented AUC = AUC(component_value, timeout)
#       higher_easier: oriented AUC = 1 - AUC(component_value, timeout)

#     This matches the uploaded scorecard code's logic:
#       oriented_timeout_effect = (raw_auc - 0.5) * orientation_sign
#       plotted_timeout_auc = oriented_timeout_effect + 0.5
#     """
#     resolved_cols = [resolve_component_column(records, c) for c in logical_components]
#     ylabels = [
#         component_display_label(logical, resolved)
#         for logical, resolved in zip(logical_components, resolved_cols)
#     ]

#     auc_mat = np.full((len(logical_components), len(verifiers)), np.nan)
#     n_mat = np.full((len(logical_components), len(verifiers)), np.nan)
#     rows = []

#     for i, (logical, component_col) in enumerate(zip(logical_components, resolved_cols)):
#         direction = component_direction(logical, component_col)
#         sign = orientation_sign(direction)

#         for j, verifier in enumerate(verifiers):
#             df = prepare_component_timeout_df(
#                 records,
#                 verifier,
#                 component_col,
#                 outlier_method,
#                 outlier_low_q,
#                 outlier_high_q,
#             )

#             n_used = int(len(df))
#             n_timeout = int(df["timeout"].sum()) if n_used else 0
#             n_solved = int(n_used - n_timeout)

#             raw_auc = None
#             oriented_auc = None

#             if n_used > 0 and n_timeout > 0 and n_solved > 0:
#                 raw_auc = roc_auc_from_scores(
#                     pd.to_numeric(df["component_value"], errors="coerce").to_numpy(float),
#                     pd.to_numeric(df["timeout"], errors="coerce").to_numpy(int),
#                 )

#                 if raw_auc is not None and np.isfinite(sign):
#                     # Equivalent to 0.5 + sign * (raw_auc - 0.5).
#                     oriented_auc = 0.5 + sign * (float(raw_auc) - 0.5)

#             if oriented_auc is not None:
#                 auc_mat[i, j] = float(oriented_auc)
#             n_mat[i, j] = float(n_used) if n_used else np.nan

#             rows.append({
#                 "component": logical,
#                 "component_column": component_col,
#                 "component_direction": direction,
#                 "orientation_sign": sign,
#                 "verifier": verifier,
#                 "n_used": n_used,
#                 "n_solved": n_solved,
#                 "n_timeout": n_timeout,
#                 "raw_timeout_auc": raw_auc,
#                 "oriented_timeout_auc": oriented_auc,
#             })

#     return pd.DataFrame(rows), auc_mat, n_mat, list(logical_components), list(verifiers), ylabels


# def annotate_auc_matrix(ax, auc_mat: np.ndarray, n_mat: np.ndarray, annotate_n: bool) -> None:
#     for i in range(auc_mat.shape[0]):
#         for j in range(auc_mat.shape[1]):
#             val = auc_mat[i, j]
#             if np.isnan(val):
#                 text = "N/A"
#                 color = "#1f2937"
#             else:
#                 if annotate_n and np.isfinite(n_mat[i, j]):
#                     text = f"{val:.2f}\n(n={int(round(n_mat[i, j]))})"
#                     fontsize = ANNOT_FONTSIZE
#                 else:
#                     text = f"{val:.2f}"
#                     fontsize = ANNOT_FONTSIZE
#                 color = "white" if val >= 0.72 or val <= 0.28 else "#1f2937"

#             ax.text(
#                 j,
#                 i,
#                 text,
#                 ha="center",
#                 va="center",
#                 fontsize=fontsize if not np.isnan(val) else ANNOT_FONTSIZE,
#                 fontweight=ANNOT_FONTWEIGHT,
#                 color=color,
#             )


# def plot_oriented_timeout_auc_scorecard(
#     records: pd.DataFrame,
#     verifiers: Sequence[str],
#     logical_components: Sequence[str],
#     out_dir: Path,
#     outlier_method: str,
#     outlier_low_q: float,
#     outlier_high_q: float,
#     annotate_n: bool,
# ) -> None:
#     table, auc_mat, n_mat, components, verifiers_used, ylabels = compute_oriented_timeout_auc_table(
#         records,
#         verifiers,
#         logical_components,
#         outlier_method,
#         outlier_low_q,
#         outlier_high_q,
#     )

#     table.to_csv(out_dir / "table_pooled_oriented_timeout_auc.csv", index=False)

#     fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=False)
#     fig.subplots_adjust(left=0.36, right=0.88, bottom=0.20, top=0.96)

#     im = ax.imshow(
#         auc_mat,
#         cmap="RdBu_r",
#         vmin=0.0,
#         vmax=1.0,
#         aspect="auto",
#     )

#     # No plot title.
#     ax.set_xticks(np.arange(len(verifiers_used)))
#     ax.set_xticklabels(
#         [verifier_display_label(v) for v in verifiers_used],
#         rotation=-15,
#         ha="center",
#         fontsize=TICK_LABEL_FONTSIZE,
#         fontweight=TICK_LABEL_FONTWEIGHT,
#     )

#     ax.set_yticks(np.arange(len(components)))
#     ax.set_yticklabels(
#         ylabels,
#         fontsize=25,#TICK_LABEL_FONTSIZE,
#         fontweight=TICK_LABEL_FONTWEIGHT,
#         linespacing=0.90,
#     )

#     ax.set_xlabel("Verifier", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT, labelpad=8)
#     ax.set_ylabel("Component", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT, labelpad=8)
#     ax.tick_params(axis="both", which="major", length=0, pad=6)

#     for tick in ax.get_yticklabels():
#         tick.set_linespacing(0.90)
#         tick.set_va("center")

#     annotate_auc_matrix(ax, auc_mat, n_mat, annotate_n=annotate_n)

#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     cbar = fig.colorbar(
#         im,
#         ax=ax,
#         fraction=0.055,
#         pad=0.035,
#         ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
#     )
#     cbar.set_label("Timeout AUC", fontsize=CBAR_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT, labelpad=8)
#     cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
#     for tick in cbar.ax.get_yticklabels():
#         tick.set_fontweight(TICK_LABEL_FONTWEIGHT)

#     stem = "fig_pooled_oriented_timeout_auc_scorecard"
#     fig.savefig(out_dir / f"{stem}.png", dpi=350, bbox_inches="tight", facecolor="white")
#     fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
#     plt.close(fig)

#     print(f"Wrote: {out_dir / (stem + '.png')}")
#     print(f"Wrote: {out_dir / (stem + '.pdf')}")
#     print(f"Wrote: {out_dir / 'table_pooled_oriented_timeout_auc.csv'}")


# # -----------------------------------------------------------------------------
# # Style / main
# # -----------------------------------------------------------------------------

# def set_white_style() -> None:
#     plt.rcParams.update({
#         "figure.facecolor": "white",
#         "axes.facecolor": "white",
#         "savefig.facecolor": "white",
#         "savefig.edgecolor": "white",
#         "font.size": 11,
#         "axes.labelsize": AXIS_LABEL_FONTSIZE,
#         "axes.labelweight": AXIS_LABEL_FONTWEIGHT,
#         "xtick.labelsize": TICK_LABEL_FONTSIZE,
#         "ytick.labelsize": TICK_LABEL_FONTSIZE,
#     })


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Make two separate pooled timeout 2D heatmaps plus the oriented "
#             "Timeout AUC scorecard heatmap."
#         )
#     )
#     parser.add_argument("--merged-records", required=True)
#     parser.add_argument("--out-dir", required=True)
#     parser.add_argument(
#         "--verifiers",
#         nargs="+",
#         default=None,
#         help="Optional verifier subset. Default: infer all *_outcome columns.",
#     )
#     parser.add_argument(
#         "--bins",
#         type=int,
#         default=4,
#         help="Number of quantile bins per axis for the pairwise timeout heatmaps.",
#     )
#     parser.add_argument(
#         "--outlier-method",
#         choices=["none", "quantile", "mad"],
#         default="quantile",
#     )
#     parser.add_argument("--outlier-low-q", type=float, default=0.01)
#     parser.add_argument("--outlier-high-q", type=float, default=0.99)
#     parser.add_argument(
#         "--auc-components",
#         nargs="+",
#         default=["M_hat_min", "d_eff", "G_IBP", "U", "A_tau"],
#         help=(
#             "Components to include in the oriented Timeout AUC scorecard. "
#             "Default: M_hat_min d_eff G_IBP U A_tau."
#         ),
#     )
#     parser.add_argument(
#         "--skip-pair-heatmaps",
#         action="store_true",
#         help="Skip A_tau×G_IBP and U×d_eff timeout-rate heatmaps.",
#     )
#     parser.add_argument(
#         "--skip-timeout-auc",
#         action="store_true",
#         help="Skip the oriented Timeout AUC scorecard heatmap.",
#     )
#     parser.add_argument(
#         "--no-annotations",
#         action="store_true",
#         help="Do not write timeout-rate/n annotations inside pairwise heatmap cells.",
#     )
#     parser.add_argument(
#         "--auc-annotate-n",
#         action="store_true",
#         help="Annotate oriented Timeout AUC cells with sample size n.",
#     )
#     parser.add_argument(
#         "--no-debug-csv",
#         action="store_true",
#         help="Do not write per-bin debug CSVs for pairwise heatmaps.",
#     )

#     args = parser.parse_args()

#     set_white_style()
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     records = load_records(Path(args.merged_records))

#     verifiers = infer_verifiers(records)
#     if args.verifiers:
#         requested = set(args.verifiers)
#         verifiers = [v for v in verifiers if v in requested]

#     if not verifiers:
#         raise ValueError(
#             "Could not infer any verifiers. Expected columns like abcrown_outcome, "
#             "marabou_outcome, neuralsat_outcome, etc."
#         )

#     print(f"Rows loaded : {len(records)}")
#     print(f"Verifiers   : {verifiers}")
#     print(f"Bins        : {args.bins}")

#     if not args.skip_pair_heatmaps:
#         plot_one_pair_heatmap(
#             records=records,
#             verifiers=verifiers,
#             xlogical="A_tau",
#             ylogical="G_IBP",
#             out_dir=out_dir,
#             bins=args.bins,
#             outlier_method=args.outlier_method,
#             outlier_low_q=args.outlier_low_q,
#             outlier_high_q=args.outlier_high_q,
#             annotate=not args.no_annotations,
#             write_debug_csv=not args.no_debug_csv,
#         )

#         plot_one_pair_heatmap(
#             records=records,
#             verifiers=verifiers,
#             xlogical="U",
#             ylogical="d_eff",
#             out_dir=out_dir,
#             bins=args.bins,
#             outlier_method=args.outlier_method,
#             outlier_low_q=args.outlier_low_q,
#             outlier_high_q=args.outlier_high_q,
#             annotate=not args.no_annotations,
#             write_debug_csv=not args.no_debug_csv,
#         )

#     if not args.skip_timeout_auc:
#         plot_oriented_timeout_auc_scorecard(
#             records=records,
#             verifiers=verifiers,
#             logical_components=args.auc_components,
#             out_dir=out_dir,
#             outlier_method=args.outlier_method,
#             outlier_low_q=args.outlier_low_q,
#             outlier_high_q=args.outlier_high_q,
#             annotate_n=args.auc_annotate_n,
#         )


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Component aliases for old and new merged-record formats
# -----------------------------------------------------------------------------

COMPONENT_ALIASES: Dict[str, List[str]] = {
    "M_hat_min": ["M_hat_min", "margin_sample_min", "sample_margin_min", "margin_sample_mean"],
    "A_tau": ["A_tau", "A_tau_local_log", "a_tau_local_log", "local_affine_cover_log"],
    "G_IBP": ["G_IBP", "ibp_relative_gap", "g_ibp"],
    "U": ["U", "unstable_frac", "U_phi", "u_phi"],
    "d_eff": ["d_eff", "effective_grad_dim_mean", "effective_grad_dim", "D_eff"],
}

# For the two pairwise timeout heatmaps.
PAIR_DISPLAY_LABELS: Dict[str, str] = {
    "A_tau": r"$A_{\tau}$ Quantile",
    "G_IBP": r"$G_{\mathrm{IBP}}$ Quantile",
    "U": r"$U$ Quantile",
    "d_eff": r"$d_{\mathrm{eff}}$ Quantile",
}

# # For the oriented Timeout AUC scorecard.
# COMPONENT_DISPLAY_NAMES: Dict[str, str] = {
#     "M_hat_min": r"Min Margin" "\n" r"($\widehat{M}_{\min}$)",
#     "margin_sample_min": r"Min Margin" "\n" r"($\widehat{M}_{\min}$)",
#     "sample_margin_min": r"Min Margin" "\n" r"($\widehat{M}_{\min}$)",
#     "margin_sample_mean": r"Min Margin" "\n" r"($\widehat{M}_{\min}$)",
#     "d_eff": r"Effective" "\n" r"Grad Dim ($d_{\mathrm{eff}}$)",
#     "effective_grad_dim_mean": r"Effective" "\n" r"Grad Dim ($d_{\mathrm{eff}}$)",
#     "effective_grad_dim": r"Effective" "\n" r"Grad Dim ($d_{\mathrm{eff}}$)",
#     "D_eff": r"Effective" "\n" r"Grad Dim ($d_{\mathrm{eff}}$)",
#     "G_IBP": r"IBP Relative Gap" "\n" r"($G_{\mathrm{IBP}}$)",
#     "ibp_relative_gap": r"IBP Relative Gap" "\n" r"($G_{\mathrm{IBP}}$)",
#     "U": r"Unstable Fraction" "\n" r"($U$)",
#     "unstable_frac": r"Unstable Fraction" "\n" r"($U$)",
#     "A_tau": r"Local Region" "\n" r"Complexity ($A_{\tau}$)",
#     "A_tau_local_log": r"Local Region" "\n" r"Complexity ($A_{\tau}$)",
# }

COMPONENT_DISPLAY_NAMES: Dict[str, str] = {
    "M_hat_min": "$\widehat{M}_{\min}$",
    "margin_sample_min": r"$\widehat{M}_{\min}$",
    "sample_margin_min": r"$\widehat{M}_{\min}$",
    "margin_sample_mean":r"$\widehat{M}_{\min}$",
    "d_eff": r"$d_{\mathrm{eff}}$",
    "effective_grad_dim_mean": r"$d_{\mathrm{eff}}$",
    "effective_grad_dim": r"$d_{\mathrm{eff}}$",
    "D_eff": r"$d_{\mathrm{eff}}$",
    "G_IBP": r"$G_{\mathrm{IBP}}$",
    "ibp_relative_gap": r"$G_{\mathrm{IBP}}$",
    "U":  r"$U$",
    "unstable_frac":r"$U$",
    "A_tau": r"$A_{\tau}$",
    "A_tau_local_log": r"$A_{\tau}$",
}

VERIFIER_DISPLAY_NAMES: Dict[str, str] = {
    "abcrown": r"$\alpha,\beta$-CROWN",
    "alpha_beta_crown": r"$\alpha,\beta$-CROWN",
    "alpha-beta-crown": r"$\alpha,\beta$-CROWN",
    "marabou": "Marabou",
    "neuralsat": "NeuralSAT",
    "nnenum": "nnenum",
    "pyrat": "PyRAT",
}

# Higher_harder means larger component values should be associated with timeout.
# Higher_easier means smaller component values should be associated with timeout.
COMPONENT_DIRECTIONS: Dict[str, str] = {
    "M_hat_min": "higher_easier",
    "margin_sample_min": "higher_easier",
    "sample_margin_min": "higher_easier",
    "margin_sample_mean": "higher_easier",
    "d_eff": "higher_harder",
    "effective_grad_dim_mean": "higher_harder",
    "effective_grad_dim": "higher_harder",
    "D_eff": "higher_harder",
    "G_IBP": "higher_harder",
    "ibp_relative_gap": "higher_harder",
    "U": "higher_harder",
    "unstable_frac": "higher_harder",
    "A_tau": "higher_harder",
    "A_tau_local_log": "higher_harder",
}


# -----------------------------------------------------------------------------
# Shared figure style
# -----------------------------------------------------------------------------

# Keep all three outputs visually consistent. The AUC scorecard has a different
# data shape than the pairwise quantile heatmaps, but the canvas size and label
# styling are intentionally identical.
FIG_SIZE = (12.0, 6)
PAIR_FIG_SIZE = (10.0, 6.0)
AUC_FIG_SIZE = FIG_SIZE

AXIS_LABEL_FONTSIZE = 22.0
AXIS_LABEL_FONTWEIGHT = "bold"
TICK_LABEL_FONTSIZE = 15.0
TICK_LABEL_FONTWEIGHT = "bold"
ANNOT_FONTSIZE = 22.0
ANNOT_FONTWEIGHT = "bold"
CBAR_LABEL_FONTSIZE = 18.0
CBAR_TICK_FONTSIZE = 18.0

# Fixed physical geometry for the colored heatmap matrix itself.
# These are inches, not fractions of the figure. Since both pairwise heatmaps
# and the AUC scorecard use this same heatmap width/height, the actual colored
# matrix area is identical across output files even though labels/ticks differ.
HEATMAP_WIDTH_IN = 6.20
HEATMAP_HEIGHT_IN = 4.55
HEATMAP_BOTTOM_IN = 0.95
PAIR_HEATMAP_LEFT_IN = 1.65
AUC_HEATMAP_LEFT_IN = 4.05
CBAR_PAD_IN = 0.25
CBAR_WIDTH_IN = 0.28


def _rect_from_inches(
    fig_size: Tuple[float, float],
    left_in: float,
    bottom_in: float,
    width_in: float,
    height_in: float,
) -> List[float]:
    """Convert an absolute inch rectangle to Matplotlib figure coordinates."""
    fig_w, fig_h = fig_size
    return [
        left_in / fig_w,
        bottom_in / fig_h,
        width_in / fig_w,
        height_in / fig_h,
    ]


def _set_fixed_heatmap_geometry(
    fig,
    ax,
    fig_size: Tuple[float, float],
    heatmap_left_in: float,
):
    """Set identical physical heatmap and colorbar geometry across figures."""
    ax.set_position(
        _rect_from_inches(
            fig_size,
            heatmap_left_in,
            HEATMAP_BOTTOM_IN,
            HEATMAP_WIDTH_IN,
            HEATMAP_HEIGHT_IN,
        )
    )
    cbar_left_in = heatmap_left_in + HEATMAP_WIDTH_IN + CBAR_PAD_IN
    cax = fig.add_axes(
        _rect_from_inches(
            fig_size,
            cbar_left_in,
            HEATMAP_BOTTOM_IN,
            CBAR_WIDTH_IN,
            HEATMAP_HEIGHT_IN,
        )
    )
    return cax



# -----------------------------------------------------------------------------
# Loading / cleaning
# -----------------------------------------------------------------------------

def load_records(path: Path) -> pd.DataFrame:
    """
    Load merged records from:
      - JSON list of dicts
      - JSON dict with records/data/rows/items/instances
      - JSONL / NDJSON
      - CSV
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)

    if suffix == ".json":
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            for key in ["records", "data", "rows", "items", "instances"]:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])

            # Last resort: dict of records.
            if all(isinstance(v, dict) for v in data.values()):
                return pd.DataFrame(list(data.values()))

        raise ValueError(
            f"Unsupported JSON structure in {path}. Expected a list of records "
            "or a dict containing records/data/rows/items/instances."
        )

    # Fallback: try JSONL, then CSV.
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        return pd.read_csv(path)


def safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def resolve_component_column(df: pd.DataFrame, logical_name: str) -> str:
    for col in COMPONENT_ALIASES.get(logical_name, [logical_name]):
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find a column for {logical_name}. Tried: "
        f"{COMPONENT_ALIASES.get(logical_name, [logical_name])}. "
        f"Available columns include: {list(df.columns)[:50]}"
    )


def infer_verifiers(df: pd.DataFrame) -> List[str]:
    """
    Infer verifier prefixes from columns like:
      abcrown_outcome, marabou_outcome, neuralsat_outcome, ...
    """
    verifiers = []
    for col in df.columns:
        if col.endswith("_outcome"):
            prefix = col[: -len("_outcome")]
            if prefix:
                verifiers.append(prefix)

    preferred_order = ["abcrown", "marabou", "neuralsat", "nnenum", "pyrat"]
    ordered = [v for v in preferred_order if v in verifiers]
    ordered += sorted(v for v in verifiers if v not in ordered)
    return ordered


def normalize_outcome(x: object) -> str:
    """
    Normalize outcome/status to solved, timeout, error, missing, or other.
    """
    s = str(x).strip().lower()
    if s in {"solved", "unsat", "safe", "verified", "true"}:
        return "solved"
    if s in {"timeout", "timed_out", "time_limit"}:
        return "timeout"
    if s in {"error", "failed", "exception"}:
        return "error"
    if s in {"missing", "nan", "none", "null", ""}:
        return "missing"
    return "other"


def verifier_display_label(verifier: str) -> str:
    return VERIFIER_DISPLAY_NAMES.get(str(verifier), str(verifier))


def component_display_label(logical_component: str, resolved_col: str) -> str:
    return (
        COMPONENT_DISPLAY_NAMES.get(logical_component)
        or COMPONENT_DISPLAY_NAMES.get(resolved_col)
        or str(logical_component).replace("_", " ")
    )


def component_direction(logical_component: str, resolved_col: str) -> str:
    return (
        COMPONENT_DIRECTIONS.get(logical_component)
        or COMPONENT_DIRECTIONS.get(resolved_col)
        or "unknown"
    )


def orientation_sign(direction: str) -> float:
    if direction == "higher_harder":
        return 1.0
    if direction == "higher_easier":
        return -1.0
    return np.nan


# -----------------------------------------------------------------------------
# Outlier handling
# -----------------------------------------------------------------------------

def display_outlier_mask(
    df: pd.DataFrame,
    cols: Sequence[str],
    method: str,
    low_q: float,
    high_q: float,
) -> np.ndarray:
    """
    Match the common plotting behavior: trim display/metric outliers by component.
    """
    keep = np.ones(len(df), dtype=bool)

    if method == "none":
        for col in cols:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            keep &= np.isfinite(vals)
        return keep

    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        finite = np.isfinite(vals)
        keep &= finite

        if method == "quantile":
            if finite.sum() < 3:
                continue
            lo = float(np.nanquantile(vals[finite], low_q))
            hi = float(np.nanquantile(vals[finite], high_q))
            keep &= vals >= lo
            keep &= vals <= hi

        elif method == "mad":
            if finite.sum() < 3:
                continue
            med = float(np.nanmedian(vals[finite]))
            mad = float(np.nanmedian(np.abs(vals[finite] - med)))
            if mad <= 1e-12:
                continue
            lo = med - 8.0 * mad
            hi = med + 8.0 * mad
            keep &= vals >= lo
            keep &= vals <= hi

        else:
            raise ValueError(f"Unknown outlier method: {method}")

    return keep


# -----------------------------------------------------------------------------
# Pairwise timeout heatmaps
# -----------------------------------------------------------------------------

def prepare_pooled_timeout_df(
    records: pd.DataFrame,
    verifiers: Sequence[str],
    xcol: str,
    ycol: str,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
) -> pd.DataFrame:
    """
    Build pooled verifier-instance timeout observations.

    One row = one (instance, verifier) pair.
    Keeps only solved and timeout outcomes.
    """
    rows = []

    for verifier in verifiers:
        outcome_col = f"{verifier}_outcome"
        status_col = f"{verifier}_status_bucket"

        if outcome_col not in records.columns and status_col not in records.columns:
            continue

        for _, r in records.iterrows():
            raw_outcome = r.get(outcome_col, r.get(status_col, "missing"))
            out = normalize_outcome(raw_outcome)

            if out not in {"solved", "timeout"}:
                continue

            x = safe_float(r.get(xcol))
            y = safe_float(r.get(ycol))
            if x is None or y is None:
                continue

            rows.append({
                "x": x,
                "y": y,
                "timeout": 1 if out == "timeout" else 0,
                "verifier": verifier,
                "benchmark": r.get("benchmark", None),
                "id": r.get("id", r.get("instance_id", None)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if outlier_method != "none":
        keep = display_outlier_mask(
            df,
            ["x", "y"],
            method=outlier_method,
            low_q=outlier_low_q,
            high_q=outlier_high_q,
        )
        df = df.loc[keep].copy()

    return df.reset_index(drop=True)


def two_dim_binned_summary(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    value_col: str,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean value_col in two-dimensional quantile bins.

    B[j, i] = mean value in x-bin i and y-bin j.
    N[j, i] = sample count in x-bin i and y-bin j.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return np.full((1, 1), np.nan), np.zeros((1, 1), dtype=int)

    df = df.loc[finite].copy()
    x = x[finite]
    y = y[finite]

    qx = np.asarray(pd.qcut(x, q=bins, labels=False, duplicates="drop"), dtype=float)
    qy = np.asarray(pd.qcut(y, q=bins, labels=False, duplicates="drop"), dtype=float)

    valid = np.isfinite(qx) & np.isfinite(qy)
    if not np.any(valid):
        return np.full((1, 1), np.nan), np.zeros((1, 1), dtype=int)

    qx = qx[valid].astype(int)
    qy = qy[valid].astype(int)
    df = df.loc[valid].copy()

    nx = int(qx.max()) + 1
    ny = int(qy.max()) + 1
    B = np.full((ny, nx), np.nan)
    N = np.zeros((ny, nx), dtype=int)

    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(float)

    for i in range(nx):
        for j in range(ny):
            m = (qx == i) & (qy == j)
            if int(m.sum()) > 0:
                B[j, i] = float(np.nanmean(values[m]))
                N[j, i] = int(m.sum())

    return B, N


def annotate_heatmap(ax, B: np.ndarray, N: np.ndarray, fontsize: float = 8.5) -> None:
    baseline = np.nanmean(B)

    for jj in range(B.shape[0]):
        for ii in range(B.shape[1]):
            if np.isnan(B[jj, ii]):
                text = "N/A"
                color = "#1f2937"
            else:
                text = f"{B[jj, ii]:.2f}\n(n={N[jj, ii]})"
                color = "white"# if B[jj, ii] > baseline else "#1f2937"

            ax.text(
                ii,
                jj,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight=ANNOT_FONTWEIGHT,
                color=color,
            )


def plot_one_pair_heatmap(
    records: pd.DataFrame,
    verifiers: Sequence[str],
    xlogical: str,
    ylogical: str,
    out_dir: Path,
    bins: int,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
    annotate: bool,
    write_debug_csv: bool,
) -> None:
    xcol = resolve_component_column(records, xlogical)
    ycol = resolve_component_column(records, ylogical)

    df = prepare_pooled_timeout_df(
        records,
        verifiers,
        xcol,
        ycol,
        outlier_method,
        outlier_low_q,
        outlier_high_q,
    )

    if df.empty:
        print(f"Skipping {xlogical}_by_{ylogical}: no solved/timeout observations.")
        return

    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(float)
    B, N = two_dim_binned_summary(df, x, y, "timeout", bins=bins)

    if write_debug_csv:
        rows = []
        for jj in range(B.shape[0]):
            for ii in range(B.shape[1]):
                rows.append({
                    "x_logical": xlogical,
                    "y_logical": ylogical,
                    "x_column": xcol,
                    "y_column": ycol,
                    "x_quantile": ii + 1,
                    "y_quantile": jj + 1,
                    "timeout_rate": B[jj, ii],
                    "n": N[jj, ii],
                })
        pd.DataFrame(rows).to_csv(
            out_dir / f"debug_timeout_heatmap_{xlogical}_by_{ylogical}_bins.csv",
            index=False,
        )

    fig, ax = plt.subplots(figsize=PAIR_FIG_SIZE, constrained_layout=False)
    cax = _set_fixed_heatmap_geometry(fig, ax, PAIR_FIG_SIZE, PAIR_HEATMAP_LEFT_IN)

    im = ax.imshow(
        B,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    # No plot titles.
    ax.set_xlabel(
        PAIR_DISPLAY_LABELS.get(xlogical, f"{xlogical} Quantile"),
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight=AXIS_LABEL_FONTWEIGHT,
    )
    ax.set_ylabel(
        PAIR_DISPLAY_LABELS.get(ylogical, f"{ylogical} Quantile"),
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight=AXIS_LABEL_FONTWEIGHT,
    )

    ax.set_xticks(range(B.shape[1]))
    ax.set_xticklabels([f"Q{i + 1}" for i in range(B.shape[1])])
    ax.set_yticks(range(B.shape[0]))
    ax.set_yticklabels([f"Q{i + 1}" for i in range(B.shape[0])])

    if annotate:
        annotate_heatmap(ax, B, N,fontsize=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Timeout Rate", fontsize=CBAR_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT)
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight(TICK_LABEL_FONTWEIGHT)

    stem = f"fig_timeout_heatmap_{xlogical}_by_{ylogical}"
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote: {out_dir / (stem + '.png')}")
    print(f"Wrote: {out_dir / (stem + '.pdf')}")


# -----------------------------------------------------------------------------
# Oriented Timeout AUC scorecard heatmap
# -----------------------------------------------------------------------------

def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j

    return ranks


def roc_auc_from_scores(scores: Sequence[float], labels01: Sequence[int]) -> Optional[float]:
    s = np.asarray(scores, float)
    y = np.asarray(labels01, int)

    mask = np.isfinite(s) & np.isfinite(y)
    s = s[mask]
    y = y[mask]

    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())

    if n_pos == 0 or n_neg == 0:
        return None

    ranks = rankdata_average(s)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def prepare_component_timeout_df(
    records: pd.DataFrame,
    verifier: str,
    component_col: str,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
) -> pd.DataFrame:
    rows = []

    outcome_col = f"{verifier}_outcome"
    status_col = f"{verifier}_status_bucket"

    if outcome_col not in records.columns and status_col not in records.columns:
        return pd.DataFrame()

    for _, r in records.iterrows():
        raw_outcome = r.get(outcome_col, r.get(status_col, "missing"))
        out = normalize_outcome(raw_outcome)
        if out not in {"solved", "timeout"}:
            continue

        value = safe_float(r.get(component_col))
        if value is None:
            continue

        rows.append({
            "component_value": value,
            "timeout": 1 if out == "timeout" else 0,
            "verifier": verifier,
            "benchmark": r.get("benchmark", None),
            "id": r.get("id", r.get("instance_id", None)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if outlier_method != "none":
        keep = display_outlier_mask(
            df,
            ["component_value"],
            method=outlier_method,
            low_q=outlier_low_q,
            high_q=outlier_high_q,
        )
        df = df.loc[keep].copy()

    return df.reset_index(drop=True)


def compute_oriented_timeout_auc_table(
    records: pd.DataFrame,
    verifiers: Sequence[str],
    logical_components: Sequence[str],
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Compute the timeout AUC panel from the oriented scorecard.

    For a component with direction:
      higher_harder: oriented AUC = AUC(component_value, timeout)
      higher_easier: oriented AUC = 1 - AUC(component_value, timeout)

    This matches the uploaded scorecard code's logic:
      oriented_timeout_effect = (raw_auc - 0.5) * orientation_sign
      plotted_timeout_auc = oriented_timeout_effect + 0.5
    """
    resolved_cols = [resolve_component_column(records, c) for c in logical_components]
    ylabels = [
        component_display_label(logical, resolved)
        for logical, resolved in zip(logical_components, resolved_cols)
    ]

    auc_mat = np.full((len(logical_components), len(verifiers)), np.nan)
    n_mat = np.full((len(logical_components), len(verifiers)), np.nan)
    rows = []

    for i, (logical, component_col) in enumerate(zip(logical_components, resolved_cols)):
        direction = component_direction(logical, component_col)
        sign = orientation_sign(direction)

        for j, verifier in enumerate(verifiers):
            df = prepare_component_timeout_df(
                records,
                verifier,
                component_col,
                outlier_method,
                outlier_low_q,
                outlier_high_q,
            )

            n_used = int(len(df))
            n_timeout = int(df["timeout"].sum()) if n_used else 0
            n_solved = int(n_used - n_timeout)

            raw_auc = None
            oriented_auc = None

            if n_used > 0 and n_timeout > 0 and n_solved > 0:
                raw_auc = roc_auc_from_scores(
                    pd.to_numeric(df["component_value"], errors="coerce").to_numpy(float),
                    pd.to_numeric(df["timeout"], errors="coerce").to_numpy(int),
                )

                if raw_auc is not None and np.isfinite(sign):
                    # Equivalent to 0.5 + sign * (raw_auc - 0.5).
                    oriented_auc = 0.5 + sign * (float(raw_auc) - 0.5)

            if oriented_auc is not None:
                auc_mat[i, j] = float(oriented_auc)
            n_mat[i, j] = float(n_used) if n_used else np.nan

            rows.append({
                "component": logical,
                "component_column": component_col,
                "component_direction": direction,
                "orientation_sign": sign,
                "verifier": verifier,
                "n_used": n_used,
                "n_solved": n_solved,
                "n_timeout": n_timeout,
                "raw_timeout_auc": raw_auc,
                "oriented_timeout_auc": oriented_auc,
            })

    return pd.DataFrame(rows), auc_mat, n_mat, list(logical_components), list(verifiers), ylabels


def annotate_auc_matrix(ax, auc_mat: np.ndarray, n_mat: np.ndarray, annotate_n: bool) -> None:
    for i in range(auc_mat.shape[0]):
        for j in range(auc_mat.shape[1]):
            val = auc_mat[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "#1f2937"
            else:
                if annotate_n and np.isfinite(n_mat[i, j]):
                    text = f"{val:.2f}\n(n={int(round(n_mat[i, j]))})"
                    fontsize = ANNOT_FONTSIZE
                else:
                    text = f"{val:.2f}"
                    fontsize = ANNOT_FONTSIZE
                color = "white" if val >= 0.72 or val <= 0.28 else "#1f2937"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=fontsize if not np.isnan(val) else ANNOT_FONTSIZE,
                fontweight=ANNOT_FONTWEIGHT,
                color=color,
            )


def plot_oriented_timeout_auc_scorecard(
    records: pd.DataFrame,
    verifiers: Sequence[str],
    logical_components: Sequence[str],
    out_dir: Path,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
    annotate_n: bool,
) -> None:
    table, auc_mat, n_mat, components, verifiers_used, ylabels = compute_oriented_timeout_auc_table(
        records,
        verifiers,
        logical_components,
        outlier_method,
        outlier_low_q,
        outlier_high_q,
    )

    table.to_csv(out_dir / "table_pooled_oriented_timeout_auc.csv", index=False)

    fig, ax = plt.subplots(figsize=AUC_FIG_SIZE, constrained_layout=False)
    cax = _set_fixed_heatmap_geometry(fig, ax, AUC_FIG_SIZE, AUC_HEATMAP_LEFT_IN)

    im = ax.imshow(
        auc_mat,
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    # No plot title.
    ax.set_xticks(np.arange(len(verifiers_used)))
    ax.set_xticklabels(
        [verifier_display_label(v) for v in verifiers_used],
        rotation=-15,
        ha="center",
        fontsize=TICK_LABEL_FONTSIZE,
        fontweight=TICK_LABEL_FONTWEIGHT,
    )

    ax.set_yticks(np.arange(len(components)))
    ax.set_yticklabels(
        ylabels,
        fontsize=25,#TICK_LABEL_FONTSIZE,
        fontweight=TICK_LABEL_FONTWEIGHT,
        linespacing=0.90,
    )

    ax.set_xlabel("Verifier", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT, labelpad=8)
    ax.set_ylabel("Component", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT, labelpad=8)
    ax.tick_params(axis="both", which="major", length=0, pad=6)

    for tick in ax.get_yticklabels():
        tick.set_linespacing(0.90)
        tick.set_va("center")

    annotate_auc_matrix(ax, auc_mat, n_mat, annotate_n=annotate_n)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(
        im,
        cax=cax,
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    cbar.set_label("Timeout AUC", fontsize=CBAR_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT, labelpad=8)
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight(TICK_LABEL_FONTWEIGHT)

    stem = "fig_pooled_oriented_timeout_auc_scorecard"
    fig.savefig(out_dir / f"{stem}.png", dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote: {out_dir / (stem + '.png')}")
    print(f"Wrote: {out_dir / (stem + '.pdf')}")
    print(f"Wrote: {out_dir / 'table_pooled_oriented_timeout_auc.csv'}")


# -----------------------------------------------------------------------------
# Style / main
# -----------------------------------------------------------------------------

def set_white_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.size": 11,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "axes.labelweight": AXIS_LABEL_FONTWEIGHT,
        "xtick.labelsize": TICK_LABEL_FONTSIZE,
        "ytick.labelsize": TICK_LABEL_FONTSIZE,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Make two separate pooled timeout 2D heatmaps plus the oriented "
            "Timeout AUC scorecard heatmap."
        )
    )
    parser.add_argument("--merged-records", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--verifiers",
        nargs="+",
        default=None,
        help="Optional verifier subset. Default: infer all *_outcome columns.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=4,
        help="Number of quantile bins per axis for the pairwise timeout heatmaps.",
    )
    parser.add_argument(
        "--outlier-method",
        choices=["none", "quantile", "mad"],
        default="quantile",
    )
    parser.add_argument("--outlier-low-q", type=float, default=0.01)
    parser.add_argument("--outlier-high-q", type=float, default=0.99)
    parser.add_argument(
        "--auc-components",
        nargs="+",
        default=["M_hat_min", "d_eff", "G_IBP", "U", "A_tau"],
        help=(
            "Components to include in the oriented Timeout AUC scorecard. "
            "Default: M_hat_min d_eff G_IBP U A_tau."
        ),
    )
    parser.add_argument(
        "--skip-pair-heatmaps",
        action="store_true",
        help="Skip A_tau×G_IBP and U×d_eff timeout-rate heatmaps.",
    )
    parser.add_argument(
        "--skip-timeout-auc",
        action="store_true",
        help="Skip the oriented Timeout AUC scorecard heatmap.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Do not write timeout-rate/n annotations inside pairwise heatmap cells.",
    )
    parser.add_argument(
        "--auc-annotate-n",
        action="store_true",
        help="Annotate oriented Timeout AUC cells with sample size n.",
    )
    parser.add_argument(
        "--no-debug-csv",
        action="store_true",
        help="Do not write per-bin debug CSVs for pairwise heatmaps.",
    )

    args = parser.parse_args()

    set_white_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(Path(args.merged_records))

    verifiers = infer_verifiers(records)
    if args.verifiers:
        requested = set(args.verifiers)
        verifiers = [v for v in verifiers if v in requested]

    if not verifiers:
        raise ValueError(
            "Could not infer any verifiers. Expected columns like abcrown_outcome, "
            "marabou_outcome, neuralsat_outcome, etc."
        )

    print(f"Rows loaded : {len(records)}")
    print(f"Verifiers   : {verifiers}")
    print(f"Bins        : {args.bins}")

    if not args.skip_pair_heatmaps:
        plot_one_pair_heatmap(
            records=records,
            verifiers=verifiers,
            xlogical="A_tau",
            ylogical="G_IBP",
            out_dir=out_dir,
            bins=args.bins,
            outlier_method=args.outlier_method,
            outlier_low_q=args.outlier_low_q,
            outlier_high_q=args.outlier_high_q,
            annotate=not args.no_annotations,
            write_debug_csv=not args.no_debug_csv,
        )

        plot_one_pair_heatmap(
            records=records,
            verifiers=verifiers,
            xlogical="U",
            ylogical="d_eff",
            out_dir=out_dir,
            bins=args.bins,
            outlier_method=args.outlier_method,
            outlier_low_q=args.outlier_low_q,
            outlier_high_q=args.outlier_high_q,
            annotate=not args.no_annotations,
            write_debug_csv=not args.no_debug_csv,
        )

    if not args.skip_timeout_auc:
        plot_oriented_timeout_auc_scorecard(
            records=records,
            verifiers=verifiers,
            logical_components=args.auc_components,
            out_dir=out_dir,
            outlier_method=args.outlier_method,
            outlier_low_q=args.outlier_low_q,
            outlier_high_q=args.outlier_high_q,
            annotate_n=args.auc_annotate_n,
        )


if __name__ == "__main__":
    main()



