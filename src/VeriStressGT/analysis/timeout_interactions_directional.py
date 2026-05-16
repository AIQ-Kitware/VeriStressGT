#!/usr/bin/env python3
from __future__ import annotations

from matplotlib import cm, colors
import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from profile_viz_utils_v5_fixed import (
    add_figure_title,
    apply_style,
    display_outlier_mask,
    ensure_dir,
    infer_verifiers,
    load_wide_records,
    nice_component_name,
    robust_certification_subset,
    safe_float,
)


# -----------------------------------------------------------------------------
# Basic numeric helpers
# -----------------------------------------------------------------------------

def zscore_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    y = (y - np.mean(y)) / (np.std(y) + 1e-12)
    return x, y


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
    mask = np.isfinite(s)
    s, y = s[mask], y[mask]
    pos = y == 1
    neg = y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata_average(s)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_fit_irls(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
    l2: float = 1e-6,
) -> Dict[str, Any]:
    """Small dependency-free logistic regression fit for binary timeout labels."""
    beta = np.zeros(X.shape[1], float)
    for _ in range(max_iter):
        eta = X @ beta
        mu = sigmoid(eta)
        W = np.clip(mu * (1 - mu), 1e-8, None)
        z = eta + (y - mu) / W
        XT_W = X.T * W
        H = XT_W @ X + l2 * np.eye(X.shape[1])
        beta_new = np.linalg.solve(H, XT_W @ z)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu = sigmoid(X @ beta)
    ll = float(np.sum(y * np.log(mu + 1e-12) + (1 - y) * np.log(1 - mu + 1e-12)))
    pbar = np.clip(y.mean(), 1e-8, 1 - 1e-8)
    ll_null = float(np.sum(y * np.log(pbar) + (1 - y) * np.log(1 - pbar)))
    pseudo_r2 = 0.0 if abs(ll_null) <= 1e-12 else 1.0 - ll / ll_null

    W = np.clip(mu * (1 - mu), 1e-8, None)
    cov = np.linalg.pinv((X.T * W) @ X + l2 * np.eye(X.shape[1]))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    zvals = beta / np.where(se > 0, se, np.nan)
    return {
        "beta": beta,
        "pseudo_r2": float(pseudo_r2),
        "auc": roc_auc_from_scores(mu, y.astype(int)),
        "zvals": zvals,
    }


# -----------------------------------------------------------------------------
# Component inference and expected difficulty directions
# -----------------------------------------------------------------------------

# Convention: +1 means larger component value should make timeout more likely.
#             -1 means larger component value should make timeout less likely.
# These are intentionally broad aliases because different scripts have used
# slightly different column names.
EXPLICIT_HARDNESS_DIRECTIONS: Dict[str, int] = {
    "A_tau_local_log": +1,
    "A_tau": +1,
    "local_affine_cover_log": +1,
    "unstable_frac": +1,
    "U_phi": +1,
    "effective_grad_dim_mean": +1,
    "effective_grad_dim": +1,
    "D_eff": +1,
    "ibp_relative_gap": +1,
    "G_IBP": +1,
    "margin_sample_min": -1,
    "sample_margin_min": -1,
    "margin_sample_mean": -1,
    "sample_margin_mean": -1,
    "B_slk": -1,
    "slack": -1,
}


def hardness_direction(component: str) -> Optional[int]:
    """
    Infer the intended hardness direction for a component.

    Returns:
      +1: higher should be harder, i.e. higher timeout probability.
      -1: higher should be easier, i.e. lower timeout probability.
      None: unknown direction, so do not use for direction-consistency filtering.
    """
    if component in EXPLICIT_HARDNESS_DIRECTIONS:
        return EXPLICIT_HARDNESS_DIRECTIONS[component]

    c = component.lower()
    if "margin" in c or "slack" in c:
        return -1
    if "unstable" in c:
        return +1
    if "effective_grad" in c or "d_eff" in c or "grad_dim" in c:
        return +1
    if "ibp" in c and "gap" in c:
        return +1
    if "a_tau" in c or "local" in c or "cover" in c:
        return +1
    if "complex" in c or "pattern" in c:
        return +1
    return None


def infer_component_columns_from_records(
    df: pd.DataFrame,
    min_count: int = 1,
) -> List[str]:
    """
    Infer candidate component columns directly from merged_records.json instead
    of relying on a hard-coded whitelist.
    """
    exclude_exact = {
        "id",
        "instance_id",
        "benchmark",
        "construction",
        "onnx_path",
        "vnnlib_path",
        "ground_truth_label",
        "label",
        "warnings",
        "component_times",
    }

    exclude_substrings = [
        "_time",
        "_outcome",
        "_status",
        "_result",
        "_error",
        "_path",
        "__na_reason",
        "__na_detail",
    ]

    comps = []
    for c in df.columns:
        if c in exclude_exact:
            continue
        if any(s in c for s in exclude_substrings):
            continue

        vals = pd.to_numeric(df[c], errors="coerce")
        n = int(vals.notna().sum())
        if n < min_count:
            continue

        # Skip columns that are constant or nearly constant.
        uniq = int(vals.dropna().nunique())
        if uniq <= 1:
            continue

        comps.append(c)

    return sorted(comps)


def _standardize_within_group(df: pd.DataFrame, col: str, group_col: str = "benchmark") -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if group_col not in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 2:
            mu = float(vals.mean())
            sd = float(vals.std(ddof=0))
            out.loc[vals.notna()] = 0.0 if sd <= 1e-12 else (vals[vals.notna()] - mu) / sd
        return out

    for _, idx in df.groupby(group_col).groups.items():
        sub = pd.to_numeric(df.loc[idx, col], errors="coerce")
        mask = sub.notna()
        if int(mask.sum()) < 3:
            continue
        mu = float(sub[mask].mean())
        sd = float(sub[mask].std(ddof=0))
        out.loc[sub[mask].index] = 0.0 if sd <= 1e-12 else (sub[mask] - mu) / sd
    return out


def _direction_fields(
    xcomp: str,
    ycomp: str,
    beta_x: float,
    beta_y: float,
) -> Dict[str, Any]:
    """
    Summarize whether main-effect signs agree with intended component directions.

    This does not force any sign on the interaction term beta_xy. It only asks
    whether each component's marginal coefficient points in the expected
    hard/easy direction.
    """
    x_dir = hardness_direction(xcomp)
    y_dir = hardness_direction(ycomp)

    x_signed = None if x_dir is None else float(x_dir * beta_x)
    y_signed = None if y_dir is None else float(y_dir * beta_y)

    x_ok = None if x_signed is None else bool(x_signed >= 0)
    y_ok = None if y_signed is None else bool(y_signed >= 0)
    n_known = int(x_dir is not None) + int(y_dir is not None)
    n_ok = int(x_ok is True) + int(y_ok is True)

    # Lower means less contradictory. Unknown directions add no penalty, but are
    # tracked separately so known-and-consistent rows sort ahead of unknown rows.
    contradiction = 0.0
    if x_signed is not None:
        contradiction += max(0.0, -x_signed)
    if y_signed is not None:
        contradiction += max(0.0, -y_signed)

    return {
        "x_hardness_direction": x_dir,
        "y_hardness_direction": y_dir,
        "x_expected_sign_ok": x_ok,
        "y_expected_sign_ok": y_ok,
        "n_direction_known": n_known,
        "n_direction_ok": n_ok,
        "direction_consistent": bool(n_known > 0 and n_ok == n_known),
        "contradiction_score": float(contradiction),
        "signed_beta_x_expected": x_signed,
        "signed_beta_y_expected": y_signed,
    }


# -----------------------------------------------------------------------------
# Timeout data preparation
# -----------------------------------------------------------------------------

def prepare_timeout_pair_df(
    records: pd.DataFrame,
    verifier: str,
    xcomp: str,
    ycomp: str,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
    standardize_within_benchmark: bool = False,
) -> pd.DataFrame:
    rows = []
    for _, r in records.iterrows():
        out = str(r.get(f"{verifier}_outcome", "missing"))
        if out not in {"solved", "timeout"}:
            continue
        x, y = safe_float(r.get(xcomp)), safe_float(r.get(ycomp))
        if x is None or y is None:
            continue
        rows.append({
            "x": x,
            "y": y,
            "benchmark": r.get("benchmark"),
            "timeout": 1 if out == "timeout" else 0,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    if outlier_method != "none":
        keep, _ = display_outlier_mask(
            df,
            ["x", "y"],
            method=outlier_method,
            low_q=outlier_low_q,
            high_q=outlier_high_q,
        )
        df = df.loc[keep].copy()

    if standardize_within_benchmark:
        df["x_std"] = _standardize_within_group(df, "x", "benchmark")
        df["y_std"] = _standardize_within_group(df, "y", "benchmark")

    return df.reset_index(drop=True)


def prepare_timeout_pair_df_pooled(
    records: pd.DataFrame,
    verifiers: Sequence[str],
    xcomp: str,
    ycomp: str,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
    standardize_within_benchmark: bool = False,
) -> pd.DataFrame:
    """Build one pooled table of (instance, verifier) timeout observations."""
    rows = []
    for verifier in verifiers:
        for _, r in records.iterrows():
            out = str(r.get(f"{verifier}_outcome", "missing"))
            if out not in {"solved", "timeout"}:
                continue
            x, y = safe_float(r.get(xcomp)), safe_float(r.get(ycomp))
            if x is None or y is None:
                continue
            rows.append({
                "x": x,
                "y": y,
                "benchmark": r.get("benchmark"),
                "verifier": verifier,
                "timeout": 1 if out == "timeout" else 0,
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    if outlier_method != "none":
        keep, _ = display_outlier_mask(
            df,
            ["x", "y"],
            method=outlier_method,
            low_q=outlier_low_q,
            high_q=outlier_high_q,
        )
        df = df.loc[keep].copy()

    if standardize_within_benchmark:
        df["x_std"] = _standardize_within_group(df, "x", "benchmark")
        df["y_std"] = _standardize_within_group(df, "y", "benchmark")

    return df.reset_index(drop=True)


def fixed_effect_matrix(labels: Sequence[object]) -> Tuple[np.ndarray, List[str]]:
    """One-hot fixed effects with the first category dropped."""
    vals = np.asarray([str(v) for v in labels], dtype=object)
    categories = sorted(pd.unique(vals).tolist())
    if len(categories) <= 1:
        return np.zeros((len(vals), 0), dtype=float), categories

    kept = categories[1:]
    X = np.zeros((len(vals), len(kept)), dtype=float)
    for j, cat in enumerate(kept):
        X[:, j] = (vals == cat).astype(float)
    return X, categories


# -----------------------------------------------------------------------------
# Timeout interaction models
# -----------------------------------------------------------------------------

def analyze_timeout_interactions(
    records: pd.DataFrame,
    verifiers: Sequence[str],
    components: Sequence[str],
    out_dir: Path,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
    min_n: int,
    standardize_within_benchmark: bool = False,
) -> pd.DataFrame:
    rows = []
    for verifier in verifiers:
        for xcomp, ycomp in itertools.combinations(components, 2):
            df = prepare_timeout_pair_df(
                records,
                verifier,
                xcomp,
                ycomp,
                outlier_method,
                outlier_low_q,
                outlier_high_q,
                standardize_within_benchmark=standardize_within_benchmark,
            )
            if len(df) < min_n or df["timeout"].sum() in {0, len(df)}:
                continue

            xs = pd.to_numeric(df["x_std"] if standardize_within_benchmark else df["x"], errors="coerce")
            ys = pd.to_numeric(df["y_std"] if standardize_within_benchmark else df["y"], errors="coerce")
            tout = pd.to_numeric(df["timeout"], errors="coerce")
            mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(tout)
            if int(mask.sum()) < min_n or int(tout[mask].sum()) in {0, int(mask.sum())}:
                continue

            xz, yz = zscore_pair(xs[mask].to_numpy(), ys[mask].to_numpy())
            inter = xz * yz
            y = tout[mask].to_numpy().astype(float)

            add = logistic_fit_irls(np.column_stack([np.ones(len(y)), xz, yz]), y)
            interfit = logistic_fit_irls(np.column_stack([np.ones(len(y)), xz, yz, inter]), y)
            beta_x = float(interfit["beta"][1])
            beta_y = float(interfit["beta"][2])
            beta_xy = float(interfit["beta"][-1])

            row = {
                "verifier": verifier,
                "x_component": xcomp,
                "y_component": ycomp,
                "n_used": int(mask.sum()),
                "n_timeout": int(tout[mask].sum()),
                "timeout_rate": float(tout[mask].mean()),
                "benchmark_standardized": bool(standardize_within_benchmark),
                "pseudo_r2_additive": add["pseudo_r2"],
                "pseudo_r2_interaction": interfit["pseudo_r2"],
                "delta_pseudo_r2": interfit["pseudo_r2"] - add["pseudo_r2"],
                "auc_additive": add["auc"],
                "auc_interaction": interfit["auc"],
                "delta_auc": (interfit["auc"] - add["auc"])
                if (add["auc"] is not None and interfit["auc"] is not None)
                else None,
                "beta_x": beta_x,
                "beta_y": beta_y,
                "beta_xy": beta_xy,
                "z_xy": float(interfit["zvals"][-1]) if np.isfinite(interfit["zvals"][-1]) else None,
            }
            row.update(_direction_fields(xcomp, ycomp, beta_x, beta_y))
            rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            [
                "benchmark_standardized",
                "verifier",
                "direction_consistent",
                "n_direction_known",
                "delta_pseudo_r2",
                "contradiction_score",
            ],
            ascending=[True, True, False, False, False, True],
        )

    stem = "all_pairs_timeout_interactions"
    if standardize_within_benchmark:
        stem += "_within_benchmark_standardized"
    out.to_csv(out_dir / f"{stem}.csv", index=False)
    return out


def analyze_pooled_timeout_interactions(
    records: pd.DataFrame,
    verifiers: Sequence[str],
    components: Sequence[str],
    out_dir: Path,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
    min_n: int,
    standardize_within_benchmark: bool = False,
) -> pd.DataFrame:
    """Fit one timeout-interaction model per component pair after pooling verifiers."""
    rows = []
    for xcomp, ycomp in itertools.combinations(components, 2):
        df = prepare_timeout_pair_df_pooled(
            records,
            verifiers,
            xcomp,
            ycomp,
            outlier_method,
            outlier_low_q,
            outlier_high_q,
            standardize_within_benchmark=standardize_within_benchmark,
        )
        if len(df) < min_n or df["timeout"].sum() in {0, len(df)}:
            continue

        xs = pd.to_numeric(df["x_std"] if standardize_within_benchmark else df["x"], errors="coerce")
        ys = pd.to_numeric(df["y_std"] if standardize_within_benchmark else df["y"], errors="coerce")
        tout = pd.to_numeric(df["timeout"], errors="coerce")
        mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(tout)
        if int(mask.sum()) < min_n or int(tout[mask].sum()) in {0, int(mask.sum())}:
            continue

        xz, yz = zscore_pair(xs[mask].to_numpy(), ys[mask].to_numpy())
        inter = xz * yz
        y = tout[mask].to_numpy().astype(float)
        fe, fe_categories = fixed_effect_matrix(df.loc[mask, "verifier"].to_numpy())

        add = logistic_fit_irls(np.column_stack([np.ones(len(y)), xz, yz, fe]), y)
        interfit = logistic_fit_irls(np.column_stack([np.ones(len(y)), xz, yz, fe, inter]), y)
        beta_x = float(interfit["beta"][1])
        beta_y = float(interfit["beta"][2])
        beta_xy = float(interfit["beta"][-1])

        row = {
            "verifier": "pooled_all",
            "x_component": xcomp,
            "y_component": ycomp,
            "n_used": int(mask.sum()),
            "n_timeout": int(tout[mask].sum()),
            "timeout_rate": float(tout[mask].mean()),
            "n_verifiers": int(df.loc[mask, "verifier"].nunique()),
            "pooled_verifiers": ",".join(fe_categories),
            "benchmark_standardized": bool(standardize_within_benchmark),
            "verifier_fixed_effects": True,
            "pseudo_r2_additive": add["pseudo_r2"],
            "pseudo_r2_interaction": interfit["pseudo_r2"],
            "delta_pseudo_r2": interfit["pseudo_r2"] - add["pseudo_r2"],
            "auc_additive": add["auc"],
            "auc_interaction": interfit["auc"],
            "delta_auc": (interfit["auc"] - add["auc"])
            if (add["auc"] is not None and interfit["auc"] is not None)
            else None,
            "beta_x": beta_x,
            "beta_y": beta_y,
            "beta_xy": beta_xy,
            "z_xy": float(interfit["zvals"][-1]) if np.isfinite(interfit["zvals"][-1]) else None,
        }
        row.update(_direction_fields(xcomp, ycomp, beta_x, beta_y))
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            [
                "benchmark_standardized",
                "direction_consistent",
                "n_direction_known",
                "delta_pseudo_r2",
                "contradiction_score",
            ],
            ascending=[True, False, False, False, True],
        )

    stem = "all_pairs_timeout_interactions_pooled_verifiers"
    if standardize_within_benchmark:
        stem += "_within_benchmark_standardized"
    out.to_csv(out_dir / f"{stem}.csv", index=False)
    return out


# -----------------------------------------------------------------------------
# Ranking helpers: top == direction-consistent by default
# -----------------------------------------------------------------------------

def _rank_for_top(
    df: pd.DataFrame,
    score_col: str,
    top_k: int,
    direction_consistent_only: bool = True,
    min_auc: Optional[float] = None,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    ranked = df.copy()
    if direction_consistent_only and "direction_consistent" in ranked.columns:
        filt = ranked[ranked["direction_consistent"].fillna(False).astype(bool)].copy()
        if len(filt) > 0:
            ranked = filt

    if min_auc is not None and "auc_interaction" in ranked.columns:
        filt = ranked[pd.to_numeric(ranked["auc_interaction"], errors="coerce") >= float(min_auc)].copy()
        if len(filt) > 0:
            ranked = filt

    sort_cols = []
    ascending = []
    if "direction_consistent" in ranked.columns:
        sort_cols.append("direction_consistent")
        ascending.append(False)
    if "n_direction_known" in ranked.columns:
        sort_cols.append("n_direction_known")
        ascending.append(False)
    if score_col in ranked.columns:
        sort_cols.append(score_col)
        ascending.append(False)
    if "auc_interaction" in ranked.columns:
        sort_cols.append("auc_interaction")
        ascending.append(False)
    if "contradiction_score" in ranked.columns:
        sort_cols.append("contradiction_score")
        ascending.append(True)

    if sort_cols:
        ranked = ranked.sort_values(sort_cols, ascending=ascending)
    return ranked.head(int(top_k))


def _top_rows_overall(
    df: pd.DataFrame,
    score_col: str,
    top_k: int,
    direction_consistent_only: bool = True,
    min_auc: Optional[float] = None,
) -> List[Dict[str, Any]]:
    ranked = _rank_for_top(df, score_col, top_k, direction_consistent_only, min_auc)
    return ranked.to_dict("records") if len(ranked) else []


def _top_rows_per_verifier(
    df: pd.DataFrame,
    score_col: str,
    verifier_col: str,
    top_k_each: int,
    direction_consistent_only: bool = True,
    min_auc: Optional[float] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df is None or len(df) == 0:
        return rows
    for verifier, g in df.groupby(verifier_col):
        ranked = _rank_for_top(g, score_col, top_k_each, direction_consistent_only, min_auc)
        rows.extend(ranked.to_dict("records"))
    return rows


# -----------------------------------------------------------------------------
# Heatmap plots
# -----------------------------------------------------------------------------

def two_dim_binned_summary(df: pd.DataFrame, x: np.ndarray, y: np.ndarray, value_col: str, bins: int = 4):
    qx = np.asarray(pd.qcut(x, q=bins, labels=False, duplicates="drop"))
    qy = np.asarray(pd.qcut(y, q=bins, labels=False, duplicates="drop"))
    nx = int(qx.max()) + 1 if len(qx) else 0
    ny = int(qy.max()) + 1 if len(qy) else 0
    B = np.full((max(ny, 1), max(nx, 1)), np.nan)
    N = np.zeros((max(ny, 1), max(nx, 1)), dtype=int)
    for i in range(nx):
        for j in range(ny):
            m = (qx == i) & (qy == j)
            if m.sum():
                B[j, i] = float(df.loc[m, value_col].mean())
                N[j, i] = int(m.sum())
    return B, N


def _plot_timeout_heatmap_panel(
    ax,
    fig,
    records: pd.DataFrame,
    verifiers: Sequence[str],
    row: Dict[str, Any],
    rank_scope: str,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
    standardize_within_benchmark: bool,
    bins: int,
):
    xcomp = row["x_component"]
    ycomp = row["y_component"]

    if rank_scope == "pooled-verifiers":
        df = prepare_timeout_pair_df_pooled(
            records,
            verifiers,
            xcomp,
            ycomp,
            outlier_method,
            outlier_low_q,
            outlier_high_q,
            standardize_within_benchmark=standardize_within_benchmark,
        )
        verifier_label = "pooled verifiers"
        title_prefix = "Pooled timeout top"
    else:
        df = prepare_timeout_pair_df(
            records,
            row["verifier"],
            xcomp,
            ycomp,
            outlier_method,
            outlier_low_q,
            outlier_high_q,
            standardize_within_benchmark=standardize_within_benchmark,
        )
        verifier_label = str(row["verifier"])
        title_prefix = "Verifier timeout top" if rank_scope == "overall" else "Per-verifier timeout top"

    if len(df) == 0:
        ax.axis("off")
        return

    xs = pd.to_numeric(df["x_std"] if standardize_within_benchmark else df["x"], errors="coerce")
    ys = pd.to_numeric(df["y_std"] if standardize_within_benchmark else df["y"], errors="coerce")
    mask = np.isfinite(xs) & np.isfinite(ys)
    df = df.loc[mask].copy()
    x = xs[mask].to_numpy()
    y = ys[mask].to_numpy()
    if len(df) == 0:
        ax.axis("off")
        return

    B, N = two_dim_binned_summary(df, x, y, "timeout", bins=bins)
    im = ax.imshow(B, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    score = safe_float(row.get("delta_pseudo_r2"))
    auc = safe_float(row.get("auc_interaction"))
    contradiction = safe_float(row.get("contradiction_score"))
    score_text = f"ΔR²={score:.3f}" if score is not None else "ΔR²=N/A"
    auc_text = f"AUC={auc:.2f}" if auc is not None else "AUC=N/A"
    contradiction_text = f"contr={contradiction:.2g}" if contradiction is not None else "contr=N/A"

    ax.set_title(
        f"{title_prefix}: {verifier_label}\n"
        f"{nice_component_name(xcomp)} × {nice_component_name(ycomp)}\n"
        f"{score_text}, {auc_text}, {contradiction_text}",
        loc="left",
        pad=7,
    )
    ax.set_xlabel(f"{nice_component_name(xcomp)} quantile")
    ax.set_ylabel(f"{nice_component_name(ycomp)} quantile")
    ax.set_xticks(range(B.shape[1]))
    ax.set_xticklabels([f"Q{i + 1}" for i in range(B.shape[1])])
    ax.set_yticks(range(B.shape[0]))
    ax.set_yticklabels([f"Q{i + 1}" for i in range(B.shape[0])])

    baseline = np.nanmean(B)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if np.isnan(B[i, j]):
                txt, color = "N/A", "#1f2937"
            else:
                txt = f"{B[i, j]:.2f}\n(n={N[i, j]})"
                color = "white" if B[i, j] > baseline else "#1f2937"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.1, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)
    cbar.set_label("timeout rate", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_top_timeout_heatmaps(
    records: pd.DataFrame,
    timeout_df: pd.DataFrame,
    pooled_timeout_df: pd.DataFrame,
    verifiers: Sequence[str],
    out_dir: Path,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
    top_k_each: int = 3,
    top_k_pooled: int = 12,
    heatmap_bins: int = 5,
    standardize_within_benchmark: bool = False,
    direction_consistent_top: bool = True,
    top_min_auc: Optional[float] = None,
):
    panels: List[Tuple[Dict[str, Any], str]] = []

    # Pooled-all-verifiers rows first. This is the part that used to be capped
    # at only 3 when --top-k-pooled defaulted to --top-k-each.
    for r in _top_rows_overall(
        pooled_timeout_df,
        "delta_pseudo_r2",
        top_k_pooled,
        direction_consistent_only=direction_consistent_top,
        min_auc=top_min_auc,
    ):
        panels.append((r, "pooled-verifiers"))

    # Then the best verifier-specific rows globally.
    for r in _top_rows_overall(
        timeout_df,
        "delta_pseudo_r2",
        top_k_each,
        direction_consistent_only=direction_consistent_top,
        min_auc=top_min_auc,
    ):
        panels.append((r, "overall"))

    # Finally, a small per-verifier breakdown.
    for r in _top_rows_per_verifier(
        timeout_df,
        "delta_pseudo_r2",
        "verifier",
        top_k_each,
        direction_consistent_only=direction_consistent_top,
        min_auc=top_min_auc,
    ):
        panels.append((r, "per-verifier"))

    if not panels:
        return

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.9 * nrows), constrained_layout=False)
    axes = np.array(axes).reshape(nrows, ncols)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.90, hspace=0.48, wspace=0.30)

    subtitle_bits = []
    if standardize_within_benchmark:
        subtitle_bits.append("components standardized within benchmark")
    if direction_consistent_top:
        subtitle_bits.append("top rows prefer component signs matching expected hard/easy directions")
    if top_min_auc is not None:
        subtitle_bits.append(f"AUC ≥ {top_min_auc:.2f} when possible")
    subtitle = "; ".join(subtitle_bits) if subtitle_bits else "raw top timeout interaction rows"
    add_figure_title(fig, "Top timeout interaction regions", subtitle, top=0.945)

    for ax, (row, rank_scope) in zip(axes.flat, panels):
        _plot_timeout_heatmap_panel(
            ax,
            fig,
            records,
            verifiers,
            row,
            rank_scope,
            outlier_method,
            outlier_low_q,
            outlier_high_q,
            standardize_within_benchmark,
            heatmap_bins,
        )

    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    stem = "fig_top_timeout_interaction_heatmaps"
    if direction_consistent_top:
        stem += "_direction_consistent"
    if standardize_within_benchmark:
        stem += "_within_benchmark_standardized"
    fig.savefig(out_dir / f"{stem}.png", dpi=250)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3D timeout surfaces: pooled first, timeout only
# -----------------------------------------------------------------------------

def _filename_token(text: object, max_len: int = 80) -> str:
    raw = str(text)
    chars = []
    last_sep = False
    for ch in raw:
        if ch.isalnum():
            chars.append(ch.lower())
            last_sep = False
        else:
            if not last_sep:
                chars.append("_")
                last_sep = True
    token = "".join(chars).strip("_")
    return (token[:max_len].strip("_") or "surface")


def _write_3d_index_html(out_dir: Path, stem: str, rows: List[Dict[str, Any]], title: str) -> None:
    if not rows:
        return

    def esc(x: object) -> str:
        text = str(x)
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    items = []
    for r in rows:
        filename = esc(r.get("html_file", ""))
        label = (
            f"#{r.get('rank')} {r.get('rank_scope')} — {r.get('verifier')} — "
            f"{r.get('x_component')} × {r.get('y_component')} "
            f"(Δpseudo-R²={float(r.get('delta_pseudo_r2')):.4f}, "
            f"AUC={float(r.get('auc_interaction')):.4f})"
        )
        items.append(f'<li><a href="{filename}">{esc(label)}</a></li>')

    html = "\n".join([
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8" />',
        f"<title>{esc(title)}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45;}",
        "li{margin:0.5rem 0;}",
        "code{background:#f3f4f6;padding:0.1rem 0.25rem;border-radius:4px;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{esc(title)}</h1>",
        "<p>Each link opens one standalone, rotatable Plotly 3D timeout surface. The files are self-contained and do not require the Plotly CDN.</p>",
        "<ol>",
        *items,
        "</ol>",
        "</body>",
        "</html>",
    ])
    (out_dir / f"{stem}_index.html").write_text(html, encoding="utf-8")


def _collect_timeout_3d_candidates(
    timeout_df: pd.DataFrame,
    pooled_timeout_df: pd.DataFrame,
    top_k_pooled_3d: int,
    top_k_verifier_3d: int,
    direction_consistent_top: bool,
    top_min_auc: Optional[float],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for r in _top_rows_overall(
        pooled_timeout_df,
        "delta_pseudo_r2",
        top_k_pooled_3d,
        direction_consistent_only=direction_consistent_top,
        min_auc=top_min_auc,
    ):
        rr = dict(r)
        rr["rank_scope"] = "pooled-verifiers"
        candidates.append(rr)

    for r in _top_rows_overall(
        timeout_df,
        "delta_pseudo_r2",
        top_k_verifier_3d,
        direction_consistent_only=direction_consistent_top,
        min_auc=top_min_auc,
    ):
        rr = dict(r)
        rr["rank_scope"] = "verifier-specific"
        candidates.append(rr)

    return candidates


def plot_top_timeout_3d_meshes(
    records: pd.DataFrame,
    timeout_df: pd.DataFrame,
    pooled_timeout_df: pd.DataFrame,
    verifiers: Sequence[str],
    out_dir: Path,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
    top_k_pooled_3d: int = 8,
    top_k_verifier_3d: int = 4,
    bins: int = 10,
    standardize_within_benchmark: bool = False,
    direction_consistent_top: bool = True,
    top_min_auc: Optional[float] = None,
):
    candidates = _collect_timeout_3d_candidates(
        timeout_df,
        pooled_timeout_df,
        top_k_pooled_3d,
        top_k_verifier_3d,
        direction_consistent_top,
        top_min_auc,
    )
    if not candidates:
        return

    suffix = "_within_benchmark_standardized" if standardize_within_benchmark else ""
    stem = f"fig_top_timeout_3d_meshes{suffix}"
    if direction_consistent_top:
        stem += "_direction_consistent"

    candidate_rows: List[Dict[str, Any]] = []
    surfaces: List[Dict[str, Any]] = []

    for rank, candidate in enumerate(candidates, start=1):
        xcomp = candidate["x_component"]
        ycomp = candidate["y_component"]
        rank_scope = candidate["rank_scope"]

        if rank_scope == "pooled-verifiers":
            df = prepare_timeout_pair_df_pooled(
                records,
                verifiers,
                xcomp,
                ycomp,
                outlier_method,
                outlier_low_q,
                outlier_high_q,
                standardize_within_benchmark=standardize_within_benchmark,
            )
            verifier_label = "pooled_all"
        else:
            df = prepare_timeout_pair_df(
                records,
                candidate["verifier"],
                xcomp,
                ycomp,
                outlier_method,
                outlier_low_q,
                outlier_high_q,
                standardize_within_benchmark=standardize_within_benchmark,
            )
            verifier_label = str(candidate["verifier"])

        if len(df) == 0:
            continue

        x_col = "x_std" if standardize_within_benchmark else "x"
        y_col = "y_std" if standardize_within_benchmark else "y"
        xs = pd.to_numeric(df[x_col], errors="coerce")
        ys = pd.to_numeric(df[y_col], errors="coerce")
        mask = np.isfinite(xs) & np.isfinite(ys)
        df = df.loc[mask].copy()
        x = xs[mask].to_numpy()
        y = ys[mask].to_numpy()
        if len(df) == 0:
            continue

        B, N = two_dim_binned_summary(df, x, y, "timeout", bins=bins)
        if not np.isfinite(B).any():
            continue

        slug = _filename_token(f"{rank_scope}_{verifier_label}_timeout_{xcomp}_x_{ycomp}", max_len=95)
        html_file = f"{stem}_rank{rank:02d}_{slug}.html"
        png_file = f"{stem}_rank{rank:02d}_{slug}.png"
        pdf_file = f"{stem}_rank{rank:02d}_{slug}.pdf"

        row = {
            "rank": rank,
            "rank_scope": rank_scope,
            "verifier": verifier_label,
            "x_component": xcomp,
            "y_component": ycomp,
            "delta_pseudo_r2": candidate.get("delta_pseudo_r2"),
            "auc_interaction": candidate.get("auc_interaction"),
            "contradiction_score": candidate.get("contradiction_score"),
            "direction_consistent": candidate.get("direction_consistent"),
            "n_used_surface": int(N.sum()),
            "bins_requested": int(bins),
            "bins_x_actual": int(B.shape[1]),
            "bins_y_actual": int(B.shape[0]),
            "benchmark_standardized": bool(standardize_within_benchmark),
            "html_file": html_file,
            "static_png_file": png_file,
            "static_pdf_file": pdf_file,
        }
        candidate_rows.append(row)
        surfaces.append({
            "row": row,
            "rank": rank,
            "candidate": candidate,
            "B": B,
            "N": N,
            "x_label": f"{nice_component_name(xcomp)} quantile bin",
            "y_label": f"{nice_component_name(ycomp)} quantile bin",
            "z_label": "timeout rate",
            "html_file": html_file,
            "png_file": png_file,
            "pdf_file": pdf_file,
        })

    if not surfaces:
        return

    wrote_plotly = False
    try:
        import plotly.graph_objects as go

        for s in surfaces:
            c = s["candidate"]
            B = s["B"]
            N = s["N"]
            xq = np.arange(1, B.shape[1] + 1)
            yq = np.arange(1, B.shape[0] + 1)
            custom = np.dstack([N])
            title_suffix = " after within-benchmark standardization" if standardize_within_benchmark else ""
            title = (
                f"#{s['rank']} Timeout Interaction Surface{title_suffix}<br>"
                f"{c['rank_scope']} — {c['verifier']}<br>"
                f"{nice_component_name(c['x_component'])} × {nice_component_name(c['y_component'])} "
                f"(Δpseudo-R²={float(c['delta_pseudo_r2']):.4f}, "
                f"AUC={float(c['auc_interaction']):.4f})"
            )

            fig = go.Figure(
                data=[
                    go.Surface(
                        x=xq,
                        y=yq,
                        z=B,
                        customdata=custom,
                        colorscale="Viridis",
                        cmin=0.0,
                        cmax=1.0,
                        colorbar={"title": "timeout rate"},
                        hovertemplate=(
                            "x quantile bin: Q%{x}<br>"
                            "y quantile bin: Q%{y}<br>"
                            "timeout rate: %{z:.3f}<br>"
                            "n: %{customdata[0]}<extra></extra>"
                        ),
                        showscale=True,
                    )
                ]
            )
            fig.update_layout(
                title=title,
                width=980,
                height=780,
                margin={"l": 20, "r": 20, "t": 125, "b": 20},
                scene={
                    "xaxis_title": s["x_label"],
                    "yaxis_title": s["y_label"],
                    "zaxis_title": "timeout rate",
                    "zaxis": {"range": [0.0, 1.0]},
                    "camera": {"eye": {"x": 1.5, "y": -1.6, "z": 1.1}},
                },
            )
            fig.write_html(
                out_dir / s["html_file"],
                include_plotlyjs=True,
                full_html=True,
                config={"responsive": True, "displaylogo": False},
            )
            wrote_plotly = True
    except Exception as e:
        print(f"Plotly 3D timeout mesh output failed; writing static fallback instead. Error: {e}")

    if wrote_plotly:
        _write_3d_index_html(
            out_dir,
            stem,
            candidate_rows,
            title=(
                "Top Timeout Interaction Surfaces"
                + (" — Within-Benchmark Standardized" if standardize_within_benchmark else "")
            ),
        )
        pd.DataFrame(candidate_rows).to_csv(out_dir / f"{stem}_candidates.csv", index=False)
        return

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    for s in surfaces:
        fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=False)
        fig.subplots_adjust(left=0.04, right=0.92, bottom=0.08, top=0.82)
        c = s["candidate"]
        title_suffix = " after within-benchmark standardization" if standardize_within_benchmark else ""
        add_figure_title(
            fig,
            f"#{s['rank']} Timeout Interaction Surface",
            (
                f"{c['rank_scope']} — {c['verifier']}{title_suffix}; "
                f"{nice_component_name(c['x_component'])} × {nice_component_name(c['y_component'])}; "
                f"Δpseudo-R²={float(c['delta_pseudo_r2']):.4f}, AUC={float(c['auc_interaction']):.4f}"
            ),
            top=0.96,
        )
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        B = s["B"]
        X, Y = np.meshgrid(np.arange(1, B.shape[1] + 1), np.arange(1, B.shape[0] + 1))
        surf = ax.plot_surface(X, Y, B, cmap="viridis", linewidth=0, antialiased=True, vmin=0.0, vmax=1.0)
        ax.set_xlabel(s["x_label"])
        ax.set_ylabel(s["y_label"])
        ax.set_zlabel("timeout rate")
        ax.set_zlim(0.0, 1.0)
        ax.set_xticks(np.arange(1, B.shape[1] + 1))
        ax.set_yticks(np.arange(1, B.shape[0] + 1))
        fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.10)
        fig.savefig(out_dir / s["png_file"], dpi=250)
        fig.savefig(out_dir / s["pdf_file"])
        plt.close(fig)

    pd.DataFrame(candidate_rows).to_csv(out_dir / f"{stem}_candidates.csv", index=False)


# -----------------------------------------------------------------------------
# CSV summaries
# -----------------------------------------------------------------------------

def write_component_coverage(records: pd.DataFrame, components: Sequence[str], out_dir: Path) -> None:
    rows = []
    for c in components:
        n = int(pd.to_numeric(records[c], errors="coerce").notna().sum()) if c in records.columns else 0
        rows.append({
            "component": c,
            "n_nonnull": n,
            "hardness_direction": hardness_direction(c),
            "direction_label": (
                "higher_harder" if hardness_direction(c) == +1 else
                "higher_easier" if hardness_direction(c) == -1 else
                "unknown"
            ),
        })
    pd.DataFrame(rows).sort_values("n_nonnull", ascending=False).to_csv(
        out_dir / "component_coverage.csv",
        index=False,
    )


def _append_summary(
    summary_rows: List[Dict[str, Any]],
    df: pd.DataFrame,
    analysis_name: str,
    score_col: str,
    top_k: int,
    rank_scope: str,
    verifier_col: Optional[str] = None,
    direction_consistent_top: bool = True,
    top_min_auc: Optional[float] = None,
) -> None:
    if df is None or len(df) == 0:
        return

    if rank_scope == "per_verifier":
        assert verifier_col is not None
        groups = df.groupby(verifier_col)
        for verifier, g in groups:
            ranked = _rank_for_top(g, score_col, top_k, direction_consistent_top, top_min_auc)
            for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
                summary_rows.append({
                    "analysis": analysis_name,
                    "rank_scope": rank_scope,
                    "rank": rank,
                    "verifier": verifier,
                    "x_component": r["x_component"],
                    "y_component": r["y_component"],
                    "interaction_score": r[score_col],
                    "auc_interaction": r.get("auc_interaction", None),
                    "beta_x": r.get("beta_x", None),
                    "beta_y": r.get("beta_y", None),
                    "beta_xy": r.get("beta_xy", None),
                    "direction_consistent": r.get("direction_consistent", None),
                    "contradiction_score": r.get("contradiction_score", None),
                    "n_used": r["n_used"],
                })
    else:
        ranked = _rank_for_top(df, score_col, top_k, direction_consistent_top, top_min_auc)
        for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
            summary_rows.append({
                "analysis": analysis_name,
                "rank_scope": rank_scope,
                "rank": rank,
                "verifier": r.get(verifier_col, "pooled_all") if verifier_col else "pooled_all",
                "x_component": r["x_component"],
                "y_component": r["y_component"],
                "interaction_score": r[score_col],
                "auc_interaction": r.get("auc_interaction", None),
                "beta_x": r.get("beta_x", None),
                "beta_y": r.get("beta_y", None),
                "beta_xy": r.get("beta_xy", None),
                "direction_consistent": r.get("direction_consistent", None),
                "contradiction_score": r.get("contradiction_score", None),
                "n_used": r["n_used"],
                "n_verifiers": r.get("n_verifiers", None),
            })


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

































# -----------------------------------------------------------------------------
# Targeted pooled 3D timeout mesh panels
# -----------------------------------------------------------------------------

_COMPONENT_ALIASES = {
    "U": ["unstable_frac", "U_phi", "U"],
    "d_eff": ["effective_grad_dim_mean", "effective_grad_dim", "D_eff", "d_eff"],
    "A_tau": ["A_tau_local_log", "A_tau", "local_affine_cover_log"],
    "G_IBP": ["ibp_relative_gap", "G_IBP"],
}


def _resolve_component_alias(records: pd.DataFrame, name: str) -> Optional[str]:
    for c in _COMPONENT_ALIASES.get(name, [name]):
        if c in records.columns:
            return c
    return None


def _target_axis_label(name: str, component: str) -> str:
    labels = {
        "U": r"$U$ quantile bin",
        "d_eff": r"$d_{\mathrm{eff}}$ quantile bin",
        "A_tau": r"$A_{\tau}$ quantile bin",
        "G_IBP": r"$G_{\mathrm{IBP}}$ quantile bin",
    }
    return labels.get(name, f"{nice_component_name(component)} quantile bin")


def plot_targeted_pooled_timeout_3d_mesh_panels(
    records: pd.DataFrame,
    verifiers: Sequence[str],
    out_dir: Path,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
    bins: int = 4,
) -> None:
    """
    Make one compact raw-component 3D mesh figure:
      left:  pooled U x d_eff -> timeout rate
      right: pooled A_tau x G_IBP -> timeout rate

    Each observation is an (instance, verifier) pair.
    Components are NOT standardized.
    Axes are quantile bins of the raw component values.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    targets = [
        ("U", "d_eff", r"$U \times d_{\mathrm{eff}}$"),
        ("G_IBP", "A_tau", r"$G_{\mathrm{IBP}} \times A_{\tau}$"),
        # ("d_eff", "U", r"$d_{\mathrm{eff}} \times U$"),
        # ("A_tau", "G_IBP", r"$A_{\tau} \times G_{\mathrm{IBP}}$"),
    ]

    resolved = []
    for xname, yname, title in targets:
        xcomp = _resolve_component_alias(records, xname)
        ycomp = _resolve_component_alias(records, yname)
        if xcomp is None or ycomp is None:
            print(
                f"Skipping targeted 3D mesh {xname} x {yname}: "
                f"could not resolve columns ({xcomp=}, {ycomp=})."
            )
            continue
        resolved.append((xname, yname, xcomp, ycomp, title))

    if not resolved:
        return

    fig = plt.figure(figsize=(8.8, 3.7), constrained_layout=False)
    fig.subplots_adjust(left=0.04, right=0.92, bottom=0.08, top=0.80, wspace=0.12)

    add_figure_title(
        fig,
        "Pooled timeout interaction meshes",
        "pooled over verifier-instance observations; raw components binned by quantile",
        top=0.94,
    )

    axes = []
    last_surf = None

    for panel_idx, (xname, yname, xcomp, ycomp, panel_title) in enumerate(resolved, start=1):
        ax = fig.add_subplot(1, 2, panel_idx, projection="3d")
        axes.append(ax)

        df = prepare_timeout_pair_df_pooled(
            records,
            verifiers,
            xcomp,
            ycomp,
            outlier_method,
            outlier_low_q,
            outlier_high_q,
            standardize_within_benchmark=False,
        )

        if len(df) == 0:
            ax.set_axis_off()
            ax.set_title(f"{panel_title}\nno data", loc="left")
            continue

        xs = pd.to_numeric(df["x"], errors="coerce")
        ys = pd.to_numeric(df["y"], errors="coerce")
        ts = pd.to_numeric(df["timeout"], errors="coerce")
        mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ts)

        df = df.loc[mask].copy()
        x = xs[mask].to_numpy(float)
        y = ys[mask].to_numpy(float)

        if len(df) == 0:
            ax.set_axis_off()
            ax.set_title(f"{panel_title}\nno finite data", loc="left")
            continue

        B, N = two_dim_binned_summary(df, x, y, "timeout", bins=bins)

        X, Y = np.meshgrid(
            np.arange(1, B.shape[1] + 1),
            np.arange(1, B.shape[0] + 1),
        )

        norm = colors.Normalize(vmin=0.0, vmax=1.0)
        cmap = cm.get_cmap("viridis")

        last_surf = ax.plot_surface(
            X,
            Y,
            B,
            facecolors=cmap(norm(B)),
            linewidth=0.35,
            edgecolor="black",
            antialiased=False,
            shade=False,
            alpha=0.96,
        )

        timeout_rate = float(pd.to_numeric(df["timeout"], errors="coerce").mean())

        ax.set_title(
            f"{panel_title}\n"
            f"n={int(N.sum())}, timeout={timeout_rate:.1%}",
            loc="left",
            pad=0,
            fontsize=10.5,
        )

        ax.set_xlabel(_target_axis_label(xname, xcomp), labelpad=4)
        ax.set_ylabel(_target_axis_label(yname, ycomp), labelpad=4)
        ax.set_zlabel("timeout rate", labelpad=4)

        ax.set_zlim(0.0, 1.0)
        ax.set_xticks(np.arange(1, B.shape[1] + 1))
        ax.set_yticks(np.arange(1, B.shape[0] + 1))
        ax.set_zticks([0.0, 0.5, 1.0])

        ax.tick_params(axis="both", which="major", labelsize=7.5)
        ax.tick_params(axis="z", which="major", labelsize=7.5)

        ax.view_init(elev=60, azim=-20)
        ax.set_box_aspect((1.2, 1.0, 0.55))

    if last_surf is not None:
        sm = cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap=cm.get_cmap("viridis"))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.62, pad=0.04)
        cbar.set_label("timeout rate", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.savefig(out_dir / "fig_pooled_timeout_targeted_3d_mesh_panels.png", dpi=300)
    fig.savefig(out_dir / "fig_pooled_timeout_targeted_3d_mesh_panels.pdf")
    plt.close(fig)






def main() -> None:
    ap = argparse.ArgumentParser(
        description="Timeout-only pairwise interaction analysis over profile components."
    )
    ap.add_argument("--merged-records", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--components", nargs="+", default=None)
    ap.add_argument("--verifiers", nargs="+", default=None)
    ap.add_argument("--outlier-method", choices=["none", "quantile", "mad"], default="quantile")
    ap.add_argument("--outlier-low-q", type=float, default=0.01)
    ap.add_argument("--outlier-high-q", type=float, default=0.99)
    ap.add_argument("--min-n-timeout", type=int, default=20)
    ap.add_argument("--min-count", type=int, default=100)
    ap.add_argument(
        "--top-k-each",
        type=int,
        default=3,
        help="Top verifier-specific rows overall and per verifier to include in figures/summaries.",
    )
    ap.add_argument(
        "--top-k-pooled",
        type=int,
        default=12,
        help="Number of pooled-all-verifiers timeout top pairs to include in heatmaps. Default is 12.",
    )
    ap.add_argument(
        "--heatmap-bins",
        type=int,
        default=4,
        help="Number of quantile bins per axis for 2D timeout heatmaps.",
    )
    ap.add_argument(
        "--top-k-pooled-3d",
        type=int,
        default=8,
        help="Number of pooled-all-verifiers timeout surfaces to create for 3D HTML plots.",
    )
    ap.add_argument(
        "--top-k-verifier-3d",
        type=int,
        default=4,
        help="Number of verifier-specific timeout surfaces to create for 3D HTML plots.",
    )
    ap.add_argument(
        "--mesh-bins",
        type=int,
        default=4,
        help="Number of quantile bins per axis for 3D timeout surfaces.",
    )
    ap.add_argument(
        "--skip-3d-mesh",
        action="store_true",
        help="Skip the separate top timeout 3D mesh surface plots.",
    )
    ap.add_argument(
        "--raw-top",
        action="store_true",
        help="Rank top rows purely by score. By default, top rows prefer expected hard/easy component signs.",
    )
    ap.add_argument(
        "--top-min-auc",
        type=float,
        default=0.55,
        help=(
            "Prefer top rows with timeout AUC at least this value, when such rows exist. "
            "Use --top-min-auc 0 to disable."
        ),
    )
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    records = robust_certification_subset(load_wide_records(Path(args.merged_records)))
    if args.benchmark and "benchmark" in records.columns:
        records = records[records["benchmark"] == args.benchmark].copy()

    verifiers = infer_verifiers(records)
    if args.verifiers:
        requested = set(args.verifiers)
        verifiers = [v for v in verifiers if v in requested]

    components = infer_component_columns_from_records(records, min_count=args.min_count)
    if args.components:
        requested_components = set(args.components)
        components = [c for c in components if c in requested_components]

    direction_consistent_top = not args.raw_top
    top_min_auc = None if args.top_min_auc is None or args.top_min_auc <= 0 else float(args.top_min_auc)

    print(f"Rows used      : {len(records)}")
    print(f"Verifiers      : {verifiers}")
    print(f"Components     : {len(components)}")
    print(f"Pairs scanned  : {len(list(itertools.combinations(components, 2)))}")
    print(f"Top mode       : {'direction-consistent' if direction_consistent_top else 'raw-score'}")
    print(f"Top min AUC    : {top_min_auc if top_min_auc is not None else 'disabled'}")

    write_component_coverage(records, components, out_dir)

    timeout_df = analyze_timeout_interactions(
        records,
        verifiers,
        components,
        out_dir,
        args.outlier_method,
        args.outlier_low_q,
        args.outlier_high_q,
        args.min_n_timeout,
        standardize_within_benchmark=False,
    )
    pooled_timeout_df = analyze_pooled_timeout_interactions(
        records,
        verifiers,
        components,
        out_dir,
        args.outlier_method,
        args.outlier_low_q,
        args.outlier_high_q,
        args.min_n_timeout,
        standardize_within_benchmark=False,
    )

    timeout_df_std = analyze_timeout_interactions(
        records,
        verifiers,
        components,
        out_dir,
        args.outlier_method,
        args.outlier_low_q,
        args.outlier_high_q,
        args.min_n_timeout,
        standardize_within_benchmark=True,
    )
    pooled_timeout_df_std = analyze_pooled_timeout_interactions(
        records,
        verifiers,
        components,
        out_dir,
        args.outlier_method,
        args.outlier_low_q,
        args.outlier_high_q,
        args.min_n_timeout,
        standardize_within_benchmark=True,
    )

    summary_rows: List[Dict[str, Any]] = []
    summary_k_pooled = max(args.top_k_pooled, 20)
    summary_k_each = max(args.top_k_each, 10)

    _append_summary(
        summary_rows,
        pooled_timeout_df,
        "timeout_pooled_verifiers",
        "delta_pseudo_r2",
        summary_k_pooled,
        "pooled_verifiers",
        verifier_col=None,
        direction_consistent_top=direction_consistent_top,
        top_min_auc=top_min_auc,
    )
    _append_summary(
        summary_rows,
        timeout_df,
        "timeout",
        "delta_pseudo_r2",
        summary_k_each,
        "overall",
        verifier_col="verifier",
        direction_consistent_top=direction_consistent_top,
        top_min_auc=top_min_auc,
    )
    _append_summary(
        summary_rows,
        timeout_df,
        "timeout",
        "delta_pseudo_r2",
        args.top_k_each,
        "per_verifier",
        verifier_col="verifier",
        direction_consistent_top=direction_consistent_top,
        top_min_auc=top_min_auc,
    )
    _append_summary(
        summary_rows,
        pooled_timeout_df_std,
        "timeout_pooled_verifiers_within_benchmark_standardized",
        "delta_pseudo_r2",
        summary_k_pooled,
        "pooled_verifiers",
        verifier_col=None,
        direction_consistent_top=direction_consistent_top,
        top_min_auc=top_min_auc,
    )
    _append_summary(
        summary_rows,
        timeout_df_std,
        "timeout_within_benchmark_standardized",
        "delta_pseudo_r2",
        summary_k_each,
        "overall",
        verifier_col="verifier",
        direction_consistent_top=direction_consistent_top,
        top_min_auc=top_min_auc,
    )
    _append_summary(
        summary_rows,
        timeout_df_std,
        "timeout_within_benchmark_standardized",
        "delta_pseudo_r2",
        args.top_k_each,
        "per_verifier",
        verifier_col="verifier",
        direction_consistent_top=direction_consistent_top,
        top_min_auc=top_min_auc,
    )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "top_timeout_interaction_summary.csv", index=False)

    plot_top_timeout_heatmaps(
        records,
        timeout_df,
        pooled_timeout_df,
        verifiers,
        out_dir,
        args.outlier_method,
        args.outlier_low_q,
        args.outlier_high_q,
        top_k_each=args.top_k_each,
        top_k_pooled=args.top_k_pooled,
        heatmap_bins=args.heatmap_bins,
        standardize_within_benchmark=False,
        direction_consistent_top=direction_consistent_top,
        top_min_auc=top_min_auc,
    )
    plot_top_timeout_heatmaps(
        records,
        timeout_df_std,
        pooled_timeout_df_std,
        verifiers,
        out_dir,
        args.outlier_method,
        args.outlier_low_q,
        args.outlier_high_q,
        top_k_each=args.top_k_each,
        top_k_pooled=args.top_k_pooled,
        heatmap_bins=args.heatmap_bins,
        standardize_within_benchmark=True,
        direction_consistent_top=direction_consistent_top,
        top_min_auc=top_min_auc,
    )

    if not args.skip_3d_mesh:
        plot_top_timeout_3d_meshes(
            records,
            timeout_df,
            pooled_timeout_df,
            verifiers,
            out_dir,
            args.outlier_method,
            args.outlier_low_q,
            args.outlier_high_q,
            top_k_pooled_3d=args.top_k_pooled_3d,
            top_k_verifier_3d=args.top_k_verifier_3d,
            bins=args.mesh_bins,
            standardize_within_benchmark=False,
            direction_consistent_top=direction_consistent_top,
            top_min_auc=top_min_auc,
        )
        plot_top_timeout_3d_meshes(
            records,
            timeout_df_std,
            pooled_timeout_df_std,
            verifiers,
            out_dir,
            args.outlier_method,
            args.outlier_low_q,
            args.outlier_high_q,
            top_k_pooled_3d=args.top_k_pooled_3d,
            top_k_verifier_3d=args.top_k_verifier_3d,
            bins=args.mesh_bins,
            standardize_within_benchmark=True,
            direction_consistent_top=direction_consistent_top,
            top_min_auc=top_min_auc,
        )

    print(f"Wrote: {out_dir/'component_coverage.csv'}")
    print(f"Wrote: {out_dir/'all_pairs_timeout_interactions.csv'}")
    print(f"Wrote: {out_dir/'all_pairs_timeout_interactions_pooled_verifiers.csv'}")
    print(f"Wrote: {out_dir/'all_pairs_timeout_interactions_within_benchmark_standardized.csv'}")
    print(f"Wrote: {out_dir/'all_pairs_timeout_interactions_pooled_verifiers_within_benchmark_standardized.csv'}")
    if summary_rows:
        print(f"Wrote: {out_dir/'top_timeout_interaction_summary.csv'}")
    heatmap_stem = "fig_top_timeout_interaction_heatmaps"
    if direction_consistent_top:
        heatmap_stem += "_direction_consistent"
    print(f"Wrote: {out_dir/(heatmap_stem + '.png')}")
    print(f"Wrote: {out_dir/(heatmap_stem + '_within_benchmark_standardized.png')}")



    plot_targeted_pooled_timeout_3d_mesh_panels(
        records,
        verifiers,
        out_dir,
        args.outlier_method,args.outlier_low_q,args.outlier_high_q,
        bins=4#args.heatmap_bins,
    )

    if not args.skip_3d_mesh:
        mesh_stem = "fig_top_timeout_3d_meshes"
        if direction_consistent_top:
            mesh_stem += "_direction_consistent"
        print(f"Wrote: {out_dir/(mesh_stem + '_index.html')}")
        print(f"Wrote: {out_dir/(mesh_stem + '_candidates.csv')}")
        print(f"Wrote: separate per-rank HTML files matching {out_dir/(mesh_stem + '_rank*.html')}")
        print(f"Wrote: {out_dir/('fig_top_timeout_3d_meshes_within_benchmark_standardized' + ('_direction_consistent' if direction_consistent_top else '') + '_index.html')}")
        print(f"Wrote: {out_dir/('fig_top_timeout_3d_meshes_within_benchmark_standardized' + ('_direction_consistent' if direction_consistent_top else '') + '_candidates.csv')}")


if __name__ == "__main__":
    main()
