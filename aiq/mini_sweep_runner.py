"""MAGNET pipeline-node runner for the mini sweep card.

Builds a 50-instance benchmark spanning 8 construction types, then runs
every requested verifier on it sequentially. Verifiers whose binary or
environment is missing are skipped gracefully (logged but not fatal).

All instances are UNSAT by construction, so the expected verdict is always UNSAT.
A verifier returning SAT is counted as wrong; TIMEOUT/UNKNOWN is not wrong.

After verification, difficulty profiles are estimated for every instance and
two analysis heatmaps (timeout AUC, runtime correlation) are saved alongside
results.json.

Writes {"per_verifier": {...}, "summary": {...}} JSON to --results_fpath so
MAGNET's GenericPipelineProcessor lifts each top-level key into a card symbol.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import scriptconfig as scfg
import ubelt as ub


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VERIFIERS = [
    "abcrown",
    "pyrat",
    "nnenum",
]

_UNSUPPORTED_KEYWORDS = (
    "unsupported",
    "not supported",
    "not implement",
    "notimplementederror",
    "unsupportedop",
    "layer type",
    "no support",
)

CONSTRUCTION_DISPLAY = {
    "mlp_relu.meap":              "MEAP",
    "mlp_relu.milp.exact_radius": "MILP",
    "mlp_relu.corners":           "Corners",
    "mlp_relu.embedded_projection": "Emb. Proj.",
    "cnn.deep_contractive_cnn":   "Contractive",
    "cnn.cnn_paired_bias":        "Paired-Bias",
    "attention.linear_dominance": "Linear-Attn",
    "attention.fixed_pattern":    "Fixed-Attn",
}

# ── Difficulty analysis constants ─────────────────────────────────────────────

ANALYSIS_COMPONENTS = [
    "margin_sample_min",
    "ibp_relative_gap",
    "unstable_frac",
    "A_tau_effective_log",
    "effective_grad_dim_mean",
]

COMPONENT_DISPLAY_NAMES = {
    "margin_sample_min":      r"$\widehat{M}_{\min}$",
    "ibp_relative_gap":       r"$G_{\mathrm{IBP}}$",
    "unstable_frac":          r"$U$",
    "A_tau_effective_log":    r"$A_{\tau}$",
    "effective_grad_dim_mean": r"$d_{\mathrm{eff}}$",
}

# True → larger value predicts timeout (score used directly for AUROC).
# False → larger value predicts easy (score is negated before AUROC).
COMPONENT_HARDER_IS_LARGER = {
    "margin_sample_min":      False,
    "ibp_relative_gap":       True,
    "unstable_frac":          True,
    "A_tau_effective_log":    True,
    "effective_grad_dim_mean": True,
}

VERIFIER_DISPLAY_NAMES = {
    "abcrown":   r"$\alpha,\beta$-CROWN",
    "pyrat":     "PyRAT",
    "nnenum":    "nnenum",
    "neuralsat": "NeuralSAT",
    "marabou":   "Marabou",
}


# ── Utility ───────────────────────────────────────────────────────────────────

def _resolve_under_repo(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (REPO_ROOT / pp).resolve()


def _run_subprocess(cmd: List[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def _load_results_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    verdicts: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return verdicts
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        verdicts[rec["instance_id"]] = rec
    return verdicts


def _classify(status: str, rec: Optional[Dict[str, Any]]) -> str:
    """Map a raw status + record into one of: correct, wrong, timeout, unsupported, error, missing."""
    s = status.lower()
    if s == "unsat":
        return "correct"
    if s == "sat":
        return "wrong"
    if s == "timeout":
        return "timeout"
    if s == "missing":
        return "missing"
    if s == "error":
        preview = ""
        if rec:
            preview = (rec.get("error_preview") or "").lower()
            if not preview:
                preview = (rec.get("stdout_preview") or "").lower()
        if any(kw in preview for kw in _UNSUPPORTED_KEYWORDS):
            return "unsupported"
        return "error"
    return "error"


def _grade_verifier(
    verifier: str,
    instance_ids: List[str],
    run_subdir: Path,
) -> Dict[str, Any]:
    verdicts = _load_results_jsonl(run_subdir / "results.jsonl")
    per_instance: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {
        "correct": 0, "wrong": 0, "timeout": 0, "unsupported": 0, "error": 0, "missing": 0,
    }

    for inst_id in instance_ids:
        rec = verdicts.get(inst_id)
        raw_status = str(rec["status"]).lower() if rec else "missing"
        wall = float(rec.get("wall_time_s") or 0.0) if rec else None
        category = _classify(raw_status, rec)
        counts[category] += 1
        per_instance.append({
            "instance_id": inst_id,
            "expected": "unsat",
            "actual": raw_status,
            "category": category,
            "wall_time_s": wall,
        })

    total = len(per_instance)
    correct_fraction = counts["correct"] / total if total else 0.0
    print(
        f"  [{verifier}] correct={counts['correct']}/{total} ({correct_fraction:.0%}) "
        f"wrong={counts['wrong']} timeout={counts['timeout']} "
        f"error={counts['error']}",
        flush=True,
    )
    return {
        "correct": counts["correct"],
        "total": total,
        "correct_fraction": correct_fraction,
        "wrong": counts["wrong"],
        "timeout": counts["timeout"],
        "unsupported": counts["unsupported"],
        "error": counts["error"],
        "skipped": False,
        "per_instance": per_instance,
    }


# ── Difficulty analysis ───────────────────────────────────────────────────────

def _auroc(labels: List[int], scores: List[float]) -> Optional[float]:
    """Area under the ROC curve via Mann-Whitney U (no external deps)."""
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    if not pos or not neg or len(pos) + len(neg) < 3:
        return None
    pairs = sum(1.0 for p in pos for n in neg if p > n) + sum(0.5 for p in pos for n in neg if p == n)
    return pairs / (len(pos) * len(neg))


def _estimate_profiles(
    bench_dir: Path,
    instance_ids: List[str],
    manifest_instances: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Estimate difficulty profiles for all instances. Returns {instance_id: profile_dict}."""
    try:
        from VeriStressGT.difficulty_profile.profile import estimate_profile
    except ImportError as exc:
        print(f"  [profile] cannot import estimate_profile: {exc}", flush=True)
        return {}

    inst_paths: Dict[str, Tuple[str, str]] = {}
    for inst in manifest_instances:
        iid = str(inst["id"])
        paths = inst.get("paths", {})
        onnx = paths.get("onnx")
        vnnlib = paths.get("vnnlib")
        if onnx and vnnlib:
            inst_paths[iid] = (
                str((bench_dir / onnx).resolve()),
                str((bench_dir / vnnlib).resolve()),
            )

    profiles: Dict[str, Dict[str, Any]] = {}
    for iid in instance_ids:
        if iid not in inst_paths:
            continue
        onnx_path, vnnlib_path = inst_paths[iid]
        print(f"  [profile] {iid} ...", flush=True)
        try:
            p = estimate_profile(onnx_path, vnnlib_path, verbose=False)
            profiles[iid] = p.to_dict()
        except Exception as exc:
            print(f"  [profile] {iid} failed: {exc}", flush=True)
            profiles[iid] = {}
    return profiles


def _cell_color_auc(val: Optional[float]) -> str:
    if val is None:
        return "#EEEEEE"
    if val >= 0.70:
        return "#C8E6C9"
    if val >= 0.50:
        return "#FFF9C4"
    return "#FFCDD2"



def _plot_component_heatmap(
    matrix: List[List[Optional[float]]],
    col_labels: List[str],
    row_labels: List[str],
    cell_fmt,
    cell_color,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(row_labels), len(col_labels)
    fig, ax = plt.subplots(figsize=(max(4, 2.8 * n_cols), max(3, 0.9 * n_rows)))

    for ri in range(n_rows):
        for ci in range(n_cols):
            val = matrix[ri][ci]
            ax.add_patch(
                plt.Rectangle(
                    (ci, ri), 1, 1,
                    facecolor=cell_color(val),
                    edgecolor="white",
                    linewidth=2,
                )
            )
            ax.text(
                ci + 0.5, ri + 0.5, cell_fmt(val),
                ha="center", va="center",
                fontsize=14, fontweight="bold",
            )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    ax.set_xticklabels(col_labels, fontsize=16, fontweight="bold")
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(row_labels, fontsize=16, fontweight="bold")
    ax.set_xlabel("Verifier", fontsize=18, fontweight="bold", labelpad=12)
    ax.set_ylabel("Difficulty Component", fontsize=18, fontweight="bold", labelpad=14)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def _plot_construction_heatmap(
    instance_ids: List[str],
    manifest: Dict[str, Any],
    per_verifier: Dict[str, Dict[str, Any]],
    active_verifiers: List[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Map instance → construction
    inst_construction: Dict[str, str] = {}
    for inst in manifest.get("instances", []):
        inst_construction[str(inst["id"])] = inst.get("construction", "unknown")

    # Ordered list of constructions as they appear
    seen: Dict[str, None] = {}
    for iid in instance_ids:
        c = inst_construction.get(iid, "unknown")
        seen[c] = None
    constructions = list(seen)

    c_labels = [CONSTRUCTION_DISPLAY.get(c, c) for c in constructions]
    v_labels = [VERIFIER_DISPLAY_NAMES.get(v, v) for v in active_verifiers]

    # Build per-instance lookup
    per_inst: Dict[str, Dict[str, Any]] = {}
    for v in active_verifiers:
        for rec in per_verifier.get(v, {}).get("per_instance", []):
            per_inst.setdefault(rec["instance_id"], {})[v] = rec

    # Aggregate per (construction, verifier)
    data: Dict[tuple, Dict[str, Any]] = {}
    for ci, cname in enumerate(constructions):
        for vi, vname in enumerate(active_verifiers):
            solved, total, times = 0, 0, []
            for iid in instance_ids:
                if inst_construction.get(iid) != cname:
                    continue
                total += 1
                rec = per_inst.get(iid, {}).get(vname)
                if rec and rec.get("category") == "correct":
                    solved += 1
                    w = rec.get("wall_time_s")
                    if w:
                        times.append(float(w))
            avg = sum(times) / len(times) if times else None
            data[(ci, vi)] = {"solved": solved, "total": total, "avg": avg}

    n_rows, n_cols = len(constructions), len(active_verifiers)
    fig, ax = plt.subplots(figsize=(max(4, 2.8 * n_cols), max(3, 0.7 * n_rows)))

    for ci in range(n_rows):
        for vi in range(n_cols):
            d = data[(ci, vi)]
            solved, total, avg = d["solved"], d["total"], d["avg"]
            if total == 0:
                color, text = "#EEEEEE", "—"
            elif solved == total:
                color = "#C8E6C9"
                text = f"{solved}/{total}\nAvg: {avg:.1f}s" if avg else f"{solved}/{total}"
            elif solved == 0:
                color = "#FFCDD2"
                text = f"{solved}/{total}"
            else:
                color = "#FFF9C4"
                text = f"{solved}/{total}\nAvg: {avg:.1f}s" if avg else f"{solved}/{total}"

            ax.add_patch(plt.Rectangle((vi, ci), 1, 1, facecolor=color, edgecolor="white", linewidth=2))
            ax.text(vi + 0.5, ci + 0.5, text, ha="center", va="center",
                    fontsize=13, fontweight="bold", linespacing=1.3)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    ax.set_xticklabels(v_labels, fontsize=16, fontweight="bold")
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(c_labels, fontsize=16, fontweight="bold")
    ax.set_xlabel("Verifier", fontsize=18, fontweight="bold", labelpad=12)
    ax.set_ylabel("Construction", fontsize=18, fontweight="bold", labelpad=14)
    ax.invert_yaxis()

    legend_patches = [
        mpatches.Patch(facecolor="#C8E6C9", edgecolor="gray", label="All UNSAT"),
        mpatches.Patch(facecolor="#FFF9C4", edgecolor="gray", label="Mixed"),
        mpatches.Patch(facecolor="#FFCDD2", edgecolor="gray", label="None solved"),
    ]
    ax.legend(handles=legend_patches, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def _run_difficulty_analysis(
    bench_dir: Path,
    manifest: Dict[str, Any],
    instance_ids: List[str],
    per_verifier: Dict[str, Dict[str, Any]],
    active_verifiers: List[str],
    out_dir: Path,
) -> None:
    """Compute difficulty profiles then save timeout-AUC and runtime-correlation heatmaps + CSVs."""
    print("\n=== Difficulty Profile Analysis ===", flush=True)

    profiles = _estimate_profiles(bench_dir, instance_ids, manifest.get("instances", []))
    if not profiles:
        print("  [profile] no profiles computed — skipping analysis", flush=True)
        return

    # Build fast lookup: instance_id → verifier → per_instance record
    per_inst: Dict[str, Dict[str, Any]] = {}
    for v in active_verifiers:
        for rec in per_verifier.get(v, {}).get("per_instance", []):
            iid = rec["instance_id"]
            per_inst.setdefault(iid, {})[v] = rec

    v_labels = [VERIFIER_DISPLAY_NAMES.get(v, v) for v in active_verifiers]
    c_labels = [COMPONENT_DISPLAY_NAMES[c] for c in ANALYSIS_COMPONENTS]

    auc_matrix: List[List[Optional[float]]] = []

    for comp in ANALYSIS_COMPONENTS:
        harder_is_larger = COMPONENT_HARDER_IS_LARGER[comp]
        auc_row: List[Optional[float]] = []

        for v in active_verifiers:
            y_labels: List[int]   = []
            x_scores: List[float] = []

            for iid in instance_ids:
                cval = profiles.get(iid, {}).get(comp)
                if cval is None:
                    continue
                try:
                    cval = float(cval)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(cval):
                    continue

                rec = per_inst.get(iid, {}).get(v)
                if rec is None:
                    continue
                category = rec.get("category")
                if category not in {"correct", "timeout"}:
                    continue

                y_labels.append(1 if category == "timeout" else 0)
                x_scores.append(cval if harder_is_larger else -cval)

            auc_row.append(_auroc(y_labels, x_scores))

        auc_matrix.append(auc_row)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "timeout_auc.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component"] + list(active_verifiers))
        for comp_key, row in zip(ANALYSIS_COMPONENTS, auc_matrix):
            w.writerow([comp_key] + [f"{v:.4f}" if v is not None else "" for v in row])
    print(f"  saved {csv_path}", flush=True)

    # ── Save plot ─────────────────────────────────────────────────────────────
    _plot_component_heatmap(
        matrix=auc_matrix,
        col_labels=v_labels,
        row_labels=c_labels,
        cell_fmt=lambda v: f"{v:.2f}" if v is not None else "N/A",
        cell_color=_cell_color_auc,
        title="Timeout AUC (component predicts timeout)",
        out_path=out_dir / "timeout_auc_heatmap.png",
    )


# ── scriptconfig CLI ──────────────────────────────────────────────────────────

class MiniSweepRunnerCLI(scfg.DataConfig):
    """Build a 50-instance multi-construction benchmark, run all requested verifiers, report results.

    All instances are UNSAT by construction; a verifier returning SAT is a
    soundness error. TIMEOUT/UNKNOWN is reported but not counted as wrong.
    """

    spec_path = scfg.Value(
        "src/VeriStressGT/configs/mini_sweep.yaml",
        help="Build spec passed to VeriStressGT.cli.create_benchmark.",
        tags=["algo_param"],
    )

    bench_dir = scfg.Value(
        "mini_sweep_bench",
        help="Where create_benchmark writes ONNX/VNNLIB instances.",
        tags=["algo_param"],
    )

    run_dir = scfg.Value(
        "mini_sweep_run",
        help="Root dir for per-verifier verify_benchmark output subdirs.",
        tags=["algo_param"],
    )

    timeout = scfg.Value(
        60.0,
        type=float,
        help="Per-instance verifier wall-clock timeout (seconds).",
        tags=["algo_param"],
    )

    verifiers = scfg.Value(
        DEFAULT_VERIFIERS,
        help="Ordered list of verifier keys to run. Missing verifiers are skipped.",
        tags=["algo_param"],
    )

    max_instances = scfg.Value(
        None,
        type=int,
        help="Cap the number of instances verified (takes the first N). None = all.",
        tags=["algo_param"],
    )

    rebuild = scfg.Value(
        False,
        help="Force re-generation of the benchmark even if bench_dir/manifest.json exists.",
        tags=["algo_param"],
    )

    abcrown_config = scfg.Value(
        "src/VeriStressGT/configs/abcrown_basic.yaml",
        help="α-β-CROWN config YAML (required by the abcrown adapter).",
        tags=["algo_param"],
    )

    max_memory_gb = scfg.Value(
        None,
        type=float,
        help="Per-instance memory cap in GB passed to verify_benchmark (RLIMIT_AS). "
             "Processes that exceed the limit are killed and recorded as ERROR rather "
             "than crashing the host.",
        tags=["algo_param"],
    )

    results_fpath = scfg.Value(
        "results.json",
        help="Output JSON file consumed by MAGNET.",
        tags=["out_path", "primary"],
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        os.environ["PYTHONUNBUFFERED"] = "1"
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        spec_path = _resolve_under_repo(config.spec_path)
        if not spec_path.exists():
            raise FileNotFoundError(f"build spec not found: {spec_path}")

        bench_dir = _resolve_under_repo(config.bench_dir)
        run_dir   = _resolve_under_repo(config.run_dir)

        verifiers: List[str] = (
            config.verifiers
            if isinstance(config.verifiers, list)
            else [v.strip() for v in str(config.verifiers).split(",") if v.strip()]
        )

        # ── 1. Build benchmark ────────────────────────────────────────────────
        manifest_path = bench_dir / "manifest.json"
        if manifest_path.exists() and not config.rebuild:
            print(
                f"\n=== Reusing existing benchmark at {bench_dir} "
                "(pass --rebuild to regenerate) ===",
                flush=True,
            )
        else:
            print("\n=== Building benchmark ===", flush=True)
            _run_subprocess([
                sys.executable, "-m", "VeriStressGT.cli.create_benchmark",
                "--spec", str(spec_path),
                "--out_dir", str(bench_dir),
                "--overwrite",
            ])

        manifest = json.loads(manifest_path.read_text())
        all_instance_ids = [inst["id"] for inst in manifest["instances"]]

        max_n: Optional[int] = config.max_instances
        instance_ids = all_instance_ids[:max_n] if max_n is not None else all_instance_ids
        print(f"Using {len(instance_ids)}/{len(all_instance_ids)} instances.", flush=True)

        verifier_extra: Dict[str, List[str]] = {
            "abcrown": [
                "--abcrown_config",
                str(_resolve_under_repo(config.abcrown_config)),
            ],
            "pyrat": [
                "--pyrat_domains", "con_z",
                "--pyrat_device", "cpu",
                "--pyrat_library", "torch",
                "--pyrat_split_relu",
                "--no-pyrat_split",
                "--pyrat_split_heuristic", "better",
            ],
        }

        # ── 2. Run each verifier ──────────────────────────────────────────────
        per_verifier: Dict[str, Dict[str, Any]] = {}
        verifiers_run: List[str] = []
        verifiers_skipped: List[str] = []

        for verifier in verifiers:
            print(f"\n=== Verifier: {verifier} ===", flush=True)
            run_subdir = run_dir / verifier
            cmd = [
                sys.executable, "-u", "-m", "VeriStressGT.cli.verify_benchmark",
                "--benchmark", str(bench_dir),
                "--verifier", verifier,
                "--out_dir", str(run_subdir),
                "--timeout", str(float(config.timeout)),
                "--instances", *instance_ids,
                "--overwrite",
                *verifier_extra.get(verifier, []),
                *(["--max_memory_gb", str(config.max_memory_gb)] if config.max_memory_gb else []),
            ]
            try:
                _run_subprocess(cmd)
                verifiers_run.append(verifier)
                per_verifier[verifier] = _grade_verifier(verifier, instance_ids, run_subdir)
            except subprocess.CalledProcessError as exc:
                print(
                    f"  [SKIP] {verifier} exited with code {exc.returncode} "
                    "(missing binary/env or setup error — skipping)",
                    flush=True,
                )
                verifiers_skipped.append(verifier)
                per_verifier[verifier] = {
                    "correct": 0,
                    "total": len(instance_ids),
                    "correct_fraction": None,
                    "wrong": 0,
                    "timeout": 0,
                    "unsupported": 0,
                    "error": 0,
                    "skipped": True,
                    "per_instance": [],
                }

        # ── 3. Write results ──────────────────────────────────────────────────
        summary: Dict[str, Any] = {
            "total_instances": len(instance_ids),
            "verifiers_run": verifiers_run,
            "verifiers_skipped": verifiers_skipped,
            "per_verifier_fraction": {
                v: per_verifier[v]["correct_fraction"]
                for v in verifiers_run
            },
        }

        def _vf(v: str) -> Dict[str, Any]:
            return per_verifier.get(v, {})

        out = {
            "result": {
                "per_verifier": per_verifier,
                "summary": summary,
                # Flat scalars surfaced by MAGNET's simple_view() on the dashboard
                "abcrown_correct_fraction": _vf("abcrown").get("correct_fraction"),
                "abcrown_correct":          _vf("abcrown").get("correct", 0),
                "abcrown_timeout":          _vf("abcrown").get("timeout", 0),
                "pyrat_correct_fraction":   _vf("pyrat").get("correct_fraction"),
                "pyrat_correct":            _vf("pyrat").get("correct", 0),
                "pyrat_timeout":            _vf("pyrat").get("timeout", 0),
                "nnenum_correct_fraction":  _vf("nnenum").get("correct_fraction"),
                "nnenum_correct":           _vf("nnenum").get("correct", 0),
                "nnenum_timeout":           _vf("nnenum").get("timeout", 0),
                "total_instances":          len(instance_ids),
                "any_sat":                  any(
                    per_verifier[v]["wrong"] > 0
                    for v in verifiers_run
                ),
            }
        }

        out_fpath = ub.Path(config.results_fpath)
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps(out, indent=2))
        print(f"\nWrote results to: {out_fpath}", flush=True)

        # ── Print summary table ───────────────────────────────────────────────
        header = f"  {'verifier':<20s} {'ok':>4}  {'wrong':>5}  {'timeout':>7}  {'error':>5}  {'total':>5}"
        print(f"\n── Summary {'─' * (len(header) - 10)}", flush=True)
        print(header, flush=True)
        print(f"  {'─'*20}  {'─'*4}  {'─'*5}  {'─'*7}  {'─'*5}  {'─'*5}", flush=True)
        for v in verifiers:
            info = per_verifier[v]
            if info["skipped"]:
                print(f"  {v:<20s} SKIPPED", flush=True)
            else:
                print(
                    f"  {v:<20s} {info['correct']:>4}  {info['wrong']:>5}  "
                    f"{info['timeout']:>7}  {info['error']:>5}  {info['total']:>5}",
                    flush=True,
                )

        csv_fpath = out_fpath.parent / "mini_sweep_summary.csv"
        with open(csv_fpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["verifier", "correct", "wrong", "timeout", "unsupported", "error", "skipped", "total", "correct_fraction"])
            for v in verifiers:
                info = per_verifier[v]
                w.writerow([
                    v,
                    info["correct"],
                    info["wrong"],
                    info["timeout"],
                    info["unsupported"],
                    info["error"],
                    info["skipped"],
                    info["total"],
                    info["correct_fraction"],
                ])
        print(f"Summary CSV: {csv_fpath}", flush=True)

        # ── 4. Construction × verifier solved/total heatmap ──────────────────
        _plot_construction_heatmap(
            instance_ids=instance_ids,
            manifest=manifest,
            per_verifier=per_verifier,
            active_verifiers=verifiers_run,
            out_path=Path(out_fpath.parent) / "construction_heatmap.png",
        )

        # ── 5. Difficulty profile analysis ────────────────────────────────────
        _run_difficulty_analysis(
            bench_dir=bench_dir,
            manifest=manifest,
            instance_ids=instance_ids,
            per_verifier=per_verifier,
            active_verifiers=verifiers_run,
            out_dir=Path(out_fpath.parent),
        )


__cli__ = MiniSweepRunnerCLI

if __name__ == "__main__":
    __cli__.main()

    r"""
    CommandLine:
        python aiq/mini_sweep_runner.py \
            --spec_path src/VeriStressGT/configs/mini_sweep.yaml \
            --bench_dir ./mini_sweep_bench \
            --run_dir ./mini_sweep_run \
            --timeout 300 \
            --verifiers abcrown pyrat nnenum \
            --results_fpath ./results.json
    """
