#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence

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
    effective_time,
    ensure_dir,
    infer_verifiers,
    load_wide_records,
    nice_component_name,
    normalize_outcome,
    robust_certification_subset,
    safe_float,
    VERIFIER_COLORS,
    get_construction_colors,
)


def sanitize_component_name(c: str) -> str:
    return c.replace("/", "_").replace(" ", "_")


def build_component_records(
    records: pd.DataFrame,
    component: str,
    verifiers: Sequence[str],
    timeout_cap: float,
    outlier_method: str,
    outlier_low_q: float,
    outlier_high_q: float,
) -> pd.DataFrame:
    rows = []
    for _, r in records.iterrows():
        x = safe_float(r.get(component))
        if x is None:
            continue
        benchmark = r.get("benchmark", "all")
        construction = r.get("construction", "unknown")
        for v in verifiers:
            outcome = normalize_outcome(r.get(f"{v}_outcome"))
            if outcome not in {"solved", "timeout"}:
                continue
            t_eff = effective_time(r, v, timeout_cap)
            t_raw = safe_float(r.get(f"{v}_time"))
            if t_eff is None:
                continue
            rows.append({
                "id": r.get("id"),
                "benchmark": benchmark,
                "construction": construction,
                "verifier": v,
                "component": component,
                "x": x,
                "runtime_censored": float(t_eff),
                "runtime_raw": t_raw,
                "log_runtime_censored": math.log1p(float(t_eff)),
                "timeout": 1 if outcome == "timeout" else 0,
                "outcome": outcome,
            })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # component-wise x trimming only, matching the philosophy of the other scripts
    if outlier_method != "none":
        kept_parts = []
        for v in df["verifier"].unique():
            for b in df["benchmark"].unique():
                sub = df[(df["verifier"] == v) & (df["benchmark"] == b)].copy()
                if len(sub) == 0:
                    continue
                keep_mask, _ = display_outlier_mask(
                    sub, ["x"],
                    method=outlier_method,
                    low_q=outlier_low_q,
                    high_q=outlier_high_q,
                )
                kept_parts.append(sub.loc[keep_mask].copy())
        if kept_parts:
            df = pd.concat(kept_parts, ignore_index=True)
        else:
            df = df.iloc[0:0].copy()
    return df

def pooled_benchmark_spearman(comp_records: pd.DataFrame) -> pd.DataFrame:
    """
    For one component's scatter dataframe, compute a single pooled-within-benchmark
    Spearman correlation across all verifiers, treating timeout as censored runtime.
    """
    rows = []

    def _rho(df: pd.DataFrame):
        if len(df) < 3:
            return None, len(df)
        x = pd.to_numeric(df["x"], errors="coerce")
        y = pd.to_numeric(df["log_runtime_censored"], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 3:
            return None, len(x)
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        return (None if pd.isna(rho) else float(rho)), len(x)

    rho_all, n_all = _rho(comp_records)
    rows.append({"benchmark": "all", "pooled_runtime_spearman": rho_all, "n_used": n_all})

    if "benchmark" in comp_records.columns:
        for bench, g in comp_records.groupby("benchmark"):
            rho, n = _rho(g)
            rows.append({"benchmark": str(bench), "pooled_runtime_spearman": rho, "n_used": n})

    return pd.DataFrame(rows)

def metric_heatmaps_for_component(
    metrics_all: pd.DataFrame,
    metrics_bench: pd.DataFrame,
    component: str,
    verifiers: Sequence[str],
    comp_records: pd.DataFrame,
    out_dir: Path,
) -> None:
    benches = list(dict.fromkeys(metrics_bench["benchmark"])) if len(metrics_bench) else []
    row_names = ["all"] + benches

    def get_value(row_name: str, verifier: str, col: str):
        if row_name == "all":
            sub = metrics_all[(metrics_all["component"] == component) & (metrics_all["verifier"] == verifier)]
        else:
            sub = metrics_bench[
                (metrics_bench["component"] == component)
                & (metrics_bench["verifier"] == verifier)
                & (metrics_bench["benchmark"] == row_name)
            ]
        if sub.empty:
            return np.nan
        val = safe_float(sub.iloc[0][col])
        return np.nan if val is None else val

    mats = {
        "runtime": np.array([[get_value(r, v, "spearman_runtime") for v in verifiers] for r in row_names], dtype=float),
        "timeout": np.array([[get_value(r, v, "timeout_effect") for v in verifiers] for r in row_names], dtype=float),
        "n": np.array([[get_value(r, v, "n_used") for v in verifiers] for r in row_names], dtype=float),
    }

    pooled_df = pooled_benchmark_spearman(comp_records)
    pooled_map = {str(r["benchmark"]): r for _, r in pooled_df.iterrows()}
    mats["pooled_runtime"] = np.array(
        [[safe_float(pooled_map.get(r, {}).get("pooled_runtime_spearman"))] for r in row_names],
        dtype=float,
    )
    mats["pooled_n"] = np.array(
        [[safe_float(pooled_map.get(r, {}).get("n_used"))] for r in row_names],
        dtype=float,
    )

    fig, axes = plt.subplots(
        1, 4,
        figsize=(22.5, max(4.2, 0.65 * len(row_names) + 2.5)),
        constrained_layout=False
    )
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.10, top=0.88, wspace=0.34)
    add_figure_title(
        fig,
        f"{nice_component_name(component)} — correlations and sample sizes",
        "Panels: verifier-specific runtime, verifier-specific timeout effect, verifier-specific sample size, and pooled-within-benchmark runtime correlation across all verifiers.",
        top=0.94,
    )

    runtime_abs = np.nanmax(np.abs(mats["runtime"])) if np.isfinite(mats["runtime"]).any() else 0.55
    timeout_abs = np.nanmax(np.abs(mats["timeout"])) if np.isfinite(mats["timeout"]).any() else 0.22
    pooled_abs = np.nanmax(np.abs(mats["pooled_runtime"])) if np.isfinite(mats["pooled_runtime"]).any() else 0.55

    specs = [
        ("runtime", "Runtime Spearman", "RdBu_r", -max(0.55, runtime_abs), max(0.55, runtime_abs), verifiers),
        ("timeout", "Timeout effect (AUC - 0.5)", "RdBu_r", -max(0.22, timeout_abs), max(0.22, timeout_abs), verifiers),
        ("n", "Sample size", "Blues", 0.0, max(1.0, np.nanmax(mats["n"]) if np.isfinite(mats["n"]).any() else 1.0), verifiers),
        ("pooled_runtime", "Pooled runtime Spearman", "RdBu_r", -max(0.55, pooled_abs), max(0.55, pooled_abs), ["pooled"]),
    ]

    for ax, (k, title, cmap, vmin, vmax, xticklabels) in zip(axes, specs):
        im = ax.imshow(mats[k], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, loc="left", pad=8)
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, fontsize=9.5)
        ax.set_yticks(np.arange(len(row_names)))
        ax.set_yticklabels(row_names, fontsize=9.5)

        for i in range(mats[k].shape[0]):
            for j in range(mats[k].shape[1]):
                val = mats[k][i, j]
                if np.isnan(val):
                    txt = "N/A"
                elif k == "n":
                    txt = f"n={int(round(val))}"
                elif k == "pooled_runtime":
                    n_here = mats["pooled_n"][i, 0]
                    txt = f"{val:+.2f}\n(n={int(round(n_here))})" if np.isfinite(n_here) else f"{val:+.2f}"
                else:
                    txt = f"{val:+.2f}"
                color = "white" if (not np.isnan(val) and k not in {"n"} and abs(val) > 0.28) else "#1f2937"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.1, color=color)

        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)
        cbar.ax.tick_params(labelsize=8.5)

    fig.savefig(out_dir / f"fig_{sanitize_component_name(component)}_correlation_heatmaps.png", dpi=250)
    fig.savefig(out_dir / f"fig_{sanitize_component_name(component)}_correlation_heatmaps.pdf")
    plt.close(fig)

def scatter_all_verifiers(df: pd.DataFrame, component: str, out_dir: Path) -> None:
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.87)
    add_figure_title(
        fig,
        f"{nice_component_name(component)} vs runtime — all verifiers",
        "Solved = circles, timeouts = triangles; y-axis uses censored runtime.",
        top=0.93,
    )

    for verifier, sub in df.groupby("verifier"):
        solved = sub[sub["timeout"] == 0]
        timeout = sub[sub["timeout"] == 1]
        color = VERIFIER_COLORS.get(verifier, None)
        if len(solved):
            ax.scatter(solved["x"], solved["runtime_censored"], s=22, alpha=0.72, linewidth=0.35,
                       edgecolors="white", marker="o", label=verifier if verifier not in ax.get_legend_handles_labels()[1] else None,
                       color=color)
        if len(timeout):
            ax.scatter(timeout["x"], timeout["runtime_censored"], s=30, alpha=0.88, linewidth=0.35,
                       edgecolors="white", marker="^", color=color)

    ax.set_xlabel(nice_component_name(component))
    ax.set_ylabel("runtime (s; timeout censored)")
    ax.set_yscale("log")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(ncol=2)
    fig.savefig(out_dir / f"fig_{sanitize_component_name(component)}_scatter_all_verifiers.png", dpi=250)
    fig.savefig(out_dir / f"fig_{sanitize_component_name(component)}_scatter_all_verifiers.pdf")
    plt.close(fig)


def scatter_by_verifier(df: pd.DataFrame, component: str, out_dir: Path) -> None:
    verifiers = list(dict.fromkeys(df["verifier"])) if len(df) else []
    if not verifiers:
        return
    ncols = 2
    nrows = int(np.ceil(len(verifiers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5 * ncols / 2, 4.7 * nrows), constrained_layout=False)
    axes = np.array(axes).reshape(nrows, ncols)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.90, hspace=0.34, wspace=0.24)
    add_figure_title(
        fig,
        f"{nice_component_name(component)} vs runtime — broken down by verifier",
        "Points are pooled across benchmarks; solved = circles, timeouts = triangles.",
        top=0.94,
    )
    bench_colors = get_construction_colors(sorted(df["benchmark"].unique()))
    for ax, verifier in zip(axes.flat, verifiers):
        sub = df[df["verifier"] == verifier].copy()
        for bench, sb in sub.groupby("benchmark"):
            solved = sb[sb["timeout"] == 0]
            timeout = sb[sb["timeout"] == 1]
            color = bench_colors.get(bench)
            if len(solved):
                ax.scatter(solved["x"], solved["runtime_censored"], s=20, alpha=0.72, linewidth=0.35,
                           edgecolors="white", marker="o", color=color, label=bench if bench not in ax.get_legend_handles_labels()[1] else None)
            if len(timeout):
                ax.scatter(timeout["x"], timeout["runtime_censored"], s=28, alpha=0.88, linewidth=0.35,
                           edgecolors="white", marker="^", color=color)
        ax.set_title(verifier, loc="left", pad=7)
        ax.set_xlabel(nice_component_name(component))
        ax.set_ylabel("runtime (s)")
        ax.set_yscale("log")
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.flat[len(verifiers):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.985), ncol=min(4, len(labels)))
    fig.savefig(out_dir / f"fig_{sanitize_component_name(component)}_scatter_by_verifier.png", dpi=250)
    fig.savefig(out_dir / f"fig_{sanitize_component_name(component)}_scatter_by_verifier.pdf")
    plt.close(fig)


def scatter_by_benchmark_and_verifier(df: pd.DataFrame, component: str, out_dir: Path) -> None:
    benches = list(dict.fromkeys(df["benchmark"])) if len(df) else []
    verifiers = list(dict.fromkeys(df["verifier"])) if len(df) else []
    if not benches or not verifiers:
        return
    fig, axes = plt.subplots(len(benches), len(verifiers),
                             figsize=(4.4 * len(verifiers), 3.6 * len(benches)),
                             constrained_layout=False)
    axes = np.array(axes).reshape(len(benches), len(verifiers))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, hspace=0.34, wspace=0.22)
    add_figure_title(
        fig,
        f"{nice_component_name(component)} vs runtime — benchmark × verifier",
        "Each panel is one benchmark/verifier slice; solved = circles, timeouts = triangles.",
        top=0.94,
    )
    construction_colors = get_construction_colors(sorted(df["construction"].unique()))
    for i, bench in enumerate(benches):
        for j, verifier in enumerate(verifiers):
            ax = axes[i, j]
            sub = df[(df["benchmark"] == bench) & (df["verifier"] == verifier)].copy()
            for cons, sc in sub.groupby("construction"):
                solved = sc[sc["timeout"] == 0]
                timeout = sc[sc["timeout"] == 1]
                color = construction_colors.get(cons)
                if len(solved):
                    ax.scatter(solved["x"], solved["runtime_censored"], s=18, alpha=0.72, linewidth=0.3,
                               edgecolors="white", marker="o", color=color)
                if len(timeout):
                    ax.scatter(timeout["x"], timeout["runtime_censored"], s=26, alpha=0.88, linewidth=0.3,
                               edgecolors="white", marker="^", color=color)
            if i == 0:
                ax.set_title(verifier, pad=7)
            if j == 0:
                ax.set_ylabel(f"{bench}\nruntime (s)")
            else:
                ax.set_ylabel("")
            if i == len(benches) - 1:
                ax.set_xlabel(nice_component_name(component))
            else:
                ax.set_xlabel("")
            ax.set_yscale("log")
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.savefig(out_dir / f"fig_{sanitize_component_name(component)}_scatter_by_benchmark_verifier.png", dpi=250)
    fig.savefig(out_dir / f"fig_{sanitize_component_name(component)}_scatter_by_benchmark_verifier.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Per-component visualization factory for the generic difficulty profile.")
    parser.add_argument("--merged-records", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--components", nargs="+", default=None, help="Optional explicit component subset.")
    parser.add_argument("--outlier-method", choices=["none", "quantile", "mad"], default="quantile")
    parser.add_argument("--outlier-low-q", type=float, default=0.01)
    parser.add_argument("--outlier-high-q", type=float, default=0.99)
    parser.add_argument("--min-count", type=int, default=1, help="Minimum non-null count to include component.")
    args = parser.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    records = load_wide_records(Path(args.merged_records))
    records = robust_certification_subset(records)
    verifiers = infer_verifiers(records)

    components = available_candidate_components(records)
    if args.components:
        wanted = set(args.components)
        components = [c for c in components if c in wanted]
    if args.min_count > 1:
        keep = []
        for c in components:
            n = int(pd.to_numeric(records[c], errors="coerce").notna().sum()) if c in records.columns else 0
            if n >= args.min_count:
                keep.append(c)
        components = keep

    timeout_cap = choose_timeout_cap(records, verifiers)

    pooled_metrics, _ = compute_metrics_for_subset(
        records, verifiers, components, benchmark_name="all",
        outlier_method=args.outlier_method,
        outlier_low_q=args.outlier_low_q,
        outlier_high_q=args.outlier_high_q,
    )

    bench_rows = []
    if "benchmark" in records.columns:
        for bench, g in records.groupby("benchmark"):
            m, _ = compute_metrics_for_subset(
                g.copy(), verifiers, components, benchmark_name=str(bench),
                outlier_method=args.outlier_method,
                outlier_low_q=args.outlier_low_q,
                outlier_high_q=args.outlier_high_q,
            )
            bench_rows.append(m)
    by_benchmark = pd.concat(bench_rows, ignore_index=True) if bench_rows else pd.DataFrame(columns=pooled_metrics.columns)

    for component in components:
        comp_dir = out_dir / sanitize_component_name(component)
        ensure_dir(comp_dir)

        comp_records = build_component_records(
            records, component, verifiers, timeout_cap,
            args.outlier_method, args.outlier_low_q, args.outlier_high_q,
        )

        pooled_metrics[pooled_metrics["component"] == component].to_csv(
            comp_dir / f"{sanitize_component_name(component)}_correlations_overall.csv", index=False
        )
        by_benchmark[by_benchmark["component"] == component].to_csv(
            comp_dir / f"{sanitize_component_name(component)}_correlations_by_benchmark.csv", index=False
        )
        comp_records.to_csv(
            comp_dir / f"{sanitize_component_name(component)}_scatter_data.csv", index=False
        )

        metric_heatmaps_for_component(
            pooled_metrics, by_benchmark, component, verifiers, comp_records, comp_dir
        )
        scatter_all_verifiers(comp_records, component, comp_dir)
        scatter_by_verifier(comp_records, component, comp_dir)
        scatter_by_benchmark_and_verifier(comp_records, component, comp_dir)

    summary = pd.DataFrame({
        "component": components,
        "nonnull_count": [
            int(pd.to_numeric(records[c], errors="coerce").notna().sum()) if c in records.columns else 0
            for c in components
        ],
    })
    summary.to_csv(out_dir / "component_summary.csv", index=False)
    print(f"Wrote component subfolders to {out_dir}")


if __name__ == "__main__":
    main()
