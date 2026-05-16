from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# =============================================================================
# Display metadata
# =============================================================================

CONSTRUCTION_SHORT = {
    "mlp_relu.embedded_projection": "Emb. Proj.",
    "mlp_relu.meap": "MEAP",
    "mlp_relu.corners": "Corners",
    "mlp_relu.milp.exact_radius": "MILP",
    "cnn.cnn_paired_bias": "Paired-Bias",
    "cnn.deep_contractive_cnn": "Contractive",
    "attention.fixed_pattern": "Fixed-Attn",
    "attention.linear_dominance": "Linear-Attn",
    "polynomial.network": "PolyNet",
}

# Paper-facing construction labels. Keep this synchronized with the
# construction-by-verifier heatmap labels and reuse it for trajectory plots.
CONSTRUCTION_DISPLAY_LABELS = {
    "mlp_relu.meap": "MEAP",
    "mlp_relu.milp.exact_radius": "MILP",
    "mlp_relu.corners": "Input-Corner Stress",
    "cnn.deep_contractive_cnn": "Deep Contractive",
    "cnn.cnn_paired_bias": "Paired-Biases",
    "attention.linear_dominance": "Dominant Key Attn",
    "attention.fixed_pattern": "Fixed Order Attn",
    "mlp_relu.embedded_projection": "Constant-on-Box",
    "polynomial.algebraic_boundary": "Polynomial Net",
    "polynomial.network": "Polynomial Net",
    "vnncomp.ingested": "MNIST_fc",
    "vnncomp.oval21": "oval21",
}

CONSTRUCTION_COLORS = {
    "mlp_relu.embedded_projection": "#2196F3",
    "mlp_relu.meap": "#F44336",
    "mlp_relu.corners": "#4CAF50",
    "mlp_relu.milp.exact_radius": "#9C27B0",
    "cnn.cnn_paired_bias": "#FF9800",
    "cnn.deep_contractive_cnn": "#00BCD4",
    "attention.fixed_pattern": "#E91E63",
    "attention.linear_dominance": "#795548",
    "polynomial.network": "#607D8B",
}

CONSTRUCTION_MARKERS = {
    "mlp_relu.embedded_projection": "o",
    "mlp_relu.meap": "s",
    "mlp_relu.corners": "^",
    "mlp_relu.milp.exact_radius": "D",
    "cnn.cnn_paired_bias": "P",
    "cnn.deep_contractive_cnn": "X",
    "attention.fixed_pattern": "v",
    "attention.linear_dominance": "*",
    "polynomial.network": "h",
}

VERIFIER_STYLE = {
    "abcrown": "partition",
    "alpha-beta-crown": "partition",
    "ab-crown": "partition",
    "neuralsat": "sat",
    "marabou": "sat_partition",
    "nnenum": "reachability",
    "nnv": "reachability",
    "pyrat": "other",
}

SOLVED_STATUSES = {"verified", "holds", "unsat"}
TIMEOUT_STATUSES = {"timeout", "timed_out", "time_out"}
COUNTEREXAMPLE_STATUSES = {"violated", "sat", "counterexample", "cex"}
ERROR_STATUSES = {"error", "failed", "crash", "exception"}
MISSING_STATUSES = {"missing"}

# Outcomes that represent a definitive answer with runtime worth plotting.
COMPLETED_OUTCOMES = {"solved", "counterexample"}


# =============================================================================
# Updated profile components
# =============================================================================

PROFILE_COMPONENTS: List[Tuple[str, str]] = [
    ("M_hat_min", "M̂_min"),
    ("G_IBP", "G_IBP"),
    ("U", "U"),
    ("A_tau", "Aτ"),
    ("d_eff", "d_eff"),
]

PROFILE_KEYS = [k for k, _ in PROFILE_COMPONENTS]

# Aliases are ordered by preference. The first present finite value is used.
# The first key in each list is the current profiler key name.
PROFILE_KEY_ALIASES: Dict[str, List[str]] = {
    "M_hat_min": [
        "margin_sample_min",
        "M_hat_min",
        "Mhat_min",
        "M_min_hat",
        "M_min_sampled",
        "M_sample_min",
        "M_emp_min",
        "margin_min",
        "sampled_min_margin",
        "sample_min_margin",
        "min_margin",
        "empirical_min_margin",
        "mhat_min",
        "m_min_hat",
    ],
    "G_IBP": [
        "G_IBP",
        "G_ibp",
        "g_ibp",
        "IBP_relative_gap",
        "ibp_relative_gap",
        "ibp_rel_gap",
        "relative_ibp_gap",
        "IBP_gap",
        "ibp_gap",
    ],
    "U": [
        "U",
        "unstable_fraction",
        "unstable_frac",
        "nonlinear_instability_fraction",
        "nonlinear_exposed_fraction",
        "U_unstable",
        "U_phi",       # legacy name; now interpreted as U if this is all that exists
    ],
    "A_tau": [
        "A_tau_local_log",
        "A_tau",
        "A_tau_empirical",
        "A_tau_hat",
        "A_tau_star_hat",
        "A_tau_upper", # legacy upper-bound/surrogate key
        "local_region_complexity",
        "log_affine_cover",
        "log_local_regions",
        "local_region_log_count",
        "local_linearization_log_count",
    ],
    "d_eff": [
        "effective_grad_dim_mean",
        "d_eff",
        "d_eff_grad",
        "effective_gradient_dim",
        "effective_gradient_dimension",
        "effective_gradient_dimension_mean",
        "D_eff",
        "D_eff_grad",
        "D_eff_input",
        "D_eff_soft",  # legacy fallback only; prefer emitting d_eff directly
    ],
}

IBP_LOWER_ALIASES = [
    "L_IBP",
    "L_ibp",
    "ibp_lower",
    "ibp_margin_lower",
    "ibp_margin_lb",
    "margin_ibp_lower",
    "lower_bound_ibp",
]

COMPONENT_COLORS = {
    "M_hat_min": "#1f77b4",
    "G_IBP": "#d62728",
    "U": "#FF9800",
    "A_tau": "#2ca02c",
    "d_eff": "#9467bd",
}

PAIRWISE_PLOTS = [
    (x, y, f"{xl} vs {yl}")
    for (x, xl), (y, yl) in itertools.combinations(PROFILE_COMPONENTS, 2)
]

INTERACTION_PLOTS = [
    ("G_IBP", "M_hat_min", "timeout_rate_G_IBP_vs_M_hat_min"),
    ("G_IBP", "A_tau", "timeout_rate_G_IBP_vs_A_tau"),
    ("G_IBP", "U", "timeout_rate_G_IBP_vs_U"),
    ("G_IBP", "d_eff", "timeout_rate_G_IBP_vs_d_eff"),
    ("U", "A_tau", "timeout_rate_U_vs_A_tau"),
    ("U", "d_eff", "timeout_rate_U_vs_d_eff"),
    ("A_tau", "d_eff", "timeout_rate_A_tau_vs_d_eff"),
]


# =============================================================================
# Utilities
# =============================================================================

def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def canonical_outcome(status: str) -> str:
    s = (status or "").strip().lower()
    if s in SOLVED_STATUSES:
        return "solved"
    if s in TIMEOUT_STATUSES:
        return "timeout"
    if s in COUNTEREXAMPLE_STATUSES:
        return "counterexample"
    if s in ERROR_STATUSES:
        return "error"
    if s in MISSING_STATUSES or s == "":
        return "missing"
    return "other"


def status_bucket(status: str) -> str:
    """
    Human-facing raw status bucket for SAT/UNSAT/TIMEOUT tables.
    Checks UNSAT before SAT so that 'unsat' is not misclassified as SAT.
    """
    s = (status or "").strip().lower()
    if s == "" or s in MISSING_STATUSES:
        return "MISSING"
    if "unsat" in s or s in {"verified", "holds"}:
        return "UNSAT"
    if "timeout" in s or "timed_out" in s or "time_out" in s:
        return "TIMEOUT"
    if s == "sat" or "counterexample" in s or "violated" in s or s == "cex":
        return "SAT"
    if s in ERROR_STATUSES or "error" in s or "exception" in s or "crash" in s:
        return "ERROR"
    return "OTHER"


def verifier_style(verifier_name: str) -> str:
    return VERIFIER_STYLE.get(verifier_name.lower(), "unknown")


def infer_verifier_name(run_dir: Path) -> str:
    name = run_dir.name
    # Usual conventions: profile_test_abcrown, abcrown_pt2, runs/abcrown.
    known = sorted(VERIFIER_STYLE.keys(), key=len, reverse=True)
    lowered = name.lower()
    for k in known:
        if lowered == k or lowered.endswith(f"_{k}") or lowered.startswith(f"{k}_") or f"_{k}_" in lowered:
            return k
    if "_" in name:
        return name.split("_")[-1]
    return name


def construction_short(name: str) -> str:
    return CONSTRUCTION_DISPLAY_LABELS.get(name, CONSTRUCTION_SHORT.get(name, name[:18]))


def construction_color(name: str) -> str:
    return CONSTRUCTION_COLORS.get(name, "#777777")


def construction_marker(name: str) -> str:
    return CONSTRUCTION_MARKERS.get(name, "o")


def parse_instance_id_from_onnx_path(onnx_path: str) -> Optional[str]:
    parts = Path(onnx_path).parts
    for i, part in enumerate(parts):
        if part == "instances" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def profile_instance_id(profile: Dict[str, Any]) -> Optional[str]:
    for key in ["instance_id", "id", "instance", "name"]:
        val = profile.get(key)
        if val not in {None, ""}:
            return str(val)
    return parse_instance_id_from_onnx_path(str(profile.get("onnx_path", "")))


def median_or_none(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    return float(np.median(xs)) if xs else None


def mean_or_none(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_first_float(d: Dict[str, Any], aliases: List[str]) -> Tuple[Optional[float], Optional[str]]:
    for key in aliases:
        if key in d:
            val = safe_float(d.get(key))
            if val is not None:
                return val, key
    return None, None


def extract_updated_profile(p: Dict[str, Any], gap_eta: float = 1e-8) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[str]]]:
    """
    Extract the final five profile components from a raw difficulty profile dict.
    Returns:
      values: canonical component name -> value
      sources: canonical component name -> source key or derivation description
    """
    values: Dict[str, Optional[float]] = {}
    sources: Dict[str, Optional[str]] = {}

    for canonical, aliases in PROFILE_KEY_ALIASES.items():
        val, src = get_first_float(p, aliases)
        values[canonical] = val
        sources[canonical] = src

    # Compute G_IBP from M_hat_min and L_IBP if not explicitly emitted.
    if values["G_IBP"] is None:
        libp, lsrc = get_first_float(p, IBP_LOWER_ALIASES)
        m = values["M_hat_min"]
        if m is not None and libp is not None:
            values["G_IBP"] = (m - libp) / (abs(m) + gap_eta)
            sources["G_IBP"] = f"computed_from:{sources.get('M_hat_min')}:{lsrc}"

    # Compute U from unstable count if no direct fraction exists.
    if values["U"] is None:
        n_unstable, usrc = get_first_float(p, ["n_unstable", "num_unstable", "unstable_count"])
        n_total, tsrc = get_first_float(p, ["n_total_neurons", "num_neurons", "total_neurons", "n_nonlinear"])
        if n_unstable is not None and n_total is not None and n_total > 0:
            values["U"] = n_unstable / n_total
            sources["U"] = f"computed_from:{usrc}:{tsrc}"

    return values, sources


def format_num(x: Optional[float], width: int = 9, precision: int = 3) -> str:
    if x is None:
        return "N/A".rjust(width)
    if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
        return f"{x:{width}.{precision}e}"
    return f"{x:{width}.{precision}f}"


# =============================================================================
# Data loading
# =============================================================================

def load_profiles(benchmark_dir: Path) -> Dict[str, Dict[str, Any]]:
    pf = benchmark_dir / "difficulty_profiles.json"
    if not pf.exists():
        print(f"[warn] No difficulty_profiles.json in {benchmark_dir}")
        return {}

    data = json.loads(pf.read_text())
    profiles = data.get("instances", data.get("profiles", []))
    if isinstance(profiles, dict):
        profiles = list(profiles.values())
    if not isinstance(profiles, list):
        profiles = [data]

    by_id: Dict[str, Dict[str, Any]] = {}
    for p in profiles:
        if not isinstance(p, dict):
            continue
        inst_id = profile_instance_id(p)
        if inst_id:
            by_id[inst_id] = p
    return by_id


def load_manifest(benchmark_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load manifest.json while preserving original instance order from the spec.
    """
    mf = benchmark_dir / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"manifest.json not found in {benchmark_dir}")

    manifest = json.loads(mf.read_text())
    by_id: Dict[str, Dict[str, Any]] = {}
    for idx, inst in enumerate(manifest.get("instances", [])):
        inst_copy = dict(inst)
        inst_copy["_yaml_order"] = idx
        by_id[str(inst["id"])] = inst_copy
    return by_id


def load_verifier_results(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load either results.jsonl or summary.csv keyed by instance_id.
    """
    results: Dict[str, Dict[str, Any]] = {}

    rfile = run_dir / "results.jsonl"
    if rfile.exists():
        with open(rfile) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                iid = str(r.get("instance_id", r.get("id", "")))
                if iid:
                    results[iid] = r
        return results

    csvf = run_dir / "summary.csv"
    if csvf.exists():
        with open(csvf) as f:
            reader = csv.DictReader(f)
            for row in reader:
                iid = str(row.get("instance_id", row.get("id", "")))
                if not iid:
                    continue
                results[iid] = {
                    "status": row.get("status", "unknown"),
                    "wall_time_s": safe_float(row.get("wall_time_s", row.get("time", row.get("runtime", row.get("runtime_s"))))),
                }
        return results

    print(f"[warn] No results.jsonl or summary.csv found in {run_dir}")
    return results


def merge_data(benchmark_dir: Path, run_dirs: List[Path], gap_eta: float) -> Tuple[List[Dict[str, Any]], List[str]]:
    profiles = load_profiles(benchmark_dir)
    manifest = load_manifest(benchmark_dir)

    verifier_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rd in run_dirs:
        vname = infer_verifier_name(rd)
        verifier_results[vname] = load_verifier_results(rd)

    records: List[Dict[str, Any]] = []

    # Iterate in manifest/YAML order, not sorted(manifest.keys()).
    for inst_id, m in manifest.items():
        p = profiles.get(inst_id, {})
        profile_error = p.get("error") if isinstance(p, dict) else None

        values, sources = extract_updated_profile(p if isinstance(p, dict) else {}, gap_eta=gap_eta)

        rec: Dict[str, Any] = {
            "id": inst_id,
            "construction": m.get("construction", "unknown"),
            "construction_short": construction_short(m.get("construction", "unknown")),
            "args": m.get("args", {}),
            "_yaml_order": m.get("_yaml_order"),
            "profile_error": profile_error,
        }
        for key in PROFILE_KEYS:
            rec[key] = values.get(key)
            rec[f"{key}_source"] = sources.get(key)

        # Optional raw fields that are useful for debugging or interpreting derived values.
        for raw_key in [
            "L_IBP",
            "L_ibp",
            "ibp_lower",
            "n_unstable",
            "n_total_neurons",
            "epsilon",
            "Lc",
            "margin_at_x0",
        ]:
            rec[raw_key] = safe_float(p.get(raw_key)) if isinstance(p, dict) else None

        for vname, vres in verifier_results.items():
            r = vres.get(inst_id, {})
            status = r.get("status", "missing")
            t = safe_float(r.get("wall_time_s", r.get("time", r.get("runtime", r.get("runtime_s")))))
            rec[f"{vname}_status"] = status
            rec[f"{vname}_status_bucket"] = status_bucket(status)
            rec[f"{vname}_time"] = t
            rec[f"{vname}_outcome"] = canonical_outcome(status)
            rec[f"{vname}_style"] = verifier_style(vname)

        records.append(rec)

    return records, list(verifier_results.keys())


def _result_ids_for_run_dir(run_dir: Path) -> set[str]:
    """
    Return instance ids present in a run directory without changing their spelling.
    This is used only to match run directories to benchmark manifests in the
    multi-benchmark mode.
    """
    results = load_verifier_results(run_dir)
    return set(results.keys())


def _strip_benchmark_prefix(instance_id: str, benchmark_name: str) -> str:
    """
    Accept either raw instance ids, e.g. meap1, or already-prefixed ids,
    e.g. sweep_all/meap1.
    """
    prefix = f"{benchmark_name}/"
    if instance_id.startswith(prefix):
        return instance_id[len(prefix):]
    return instance_id


def merge_data_multi(
    benchmark_dirs: List[Path],
    run_dirs: List[Path],
    gap_eta: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Merge records across multiple benchmark directories.

    Each benchmark contributes its own manifest/profile records. Run directories
    are matched to a benchmark when their result ids overlap that benchmark's
    manifest ids. This keeps the old single-benchmark behavior while allowing a
    single merged_records.json across benchmark suites.
    """
    all_records: List[Dict[str, Any]] = []
    all_verifiers: List[str] = []

    run_dir_ids: Dict[Path, set[str]] = {}
    for rd in run_dirs:
        run_dir_ids[rd] = _result_ids_for_run_dir(rd)

    for bdir in benchmark_dirs:
        benchmark_name = bdir.name
        try:
            manifest = load_manifest(bdir)
        except FileNotFoundError as e:
            print(f"[warn] skipping benchmark {bdir}: {e}")
            continue

        manifest_ids = set(manifest.keys())
        prefixed_manifest_ids = {f"{benchmark_name}/{iid}" for iid in manifest_ids}

        matching_run_dirs: List[Path] = []
        for rd, ids in run_dir_ids.items():
            if ids & manifest_ids or ids & prefixed_manifest_ids:
                matching_run_dirs.append(rd)

        if not matching_run_dirs:
            print(f"[warn] no matching run dirs found for benchmark {bdir}")
            continue

        records, verifiers = merge_data(bdir, matching_run_dirs, gap_eta=gap_eta)

        for r in records:
            raw_id = str(r["id"])
            r["benchmark"] = benchmark_name
            r["instance_id"] = raw_id
            r["id"] = f"{benchmark_name}/{raw_id}"

        all_records.extend(records)

        for v in verifiers:
            if v not in all_verifiers:
                all_verifiers.append(v)

    return all_records, all_verifiers


# =============================================================================
# CSV and summary tables
# =============================================================================

def write_merged_json(records: List[Dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "merged_records.json"
    path.write_text(json.dumps(sanitize_for_json(records), indent=2))
    print(f"  saved {path}")


def write_longform_csv(records: List[Dict[str, Any]], verifiers: List[str], out_dir: Path) -> None:
    path = out_dir / "records_longform.csv"
    profile_fields: List[str] = []
    for k in PROFILE_KEYS:
        profile_fields.extend([k, f"{k}_source"])

    fieldnames = [
        "benchmark",
        "id",
        "instance_id",
        "construction",
        "construction_short",
        "verifier",
        "style",
        "status",
        "status_bucket",
        "outcome",
        "time",
        *profile_fields,
        "profile_error",
        "_yaml_order",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            for v in verifiers:
                row = {
                    "benchmark": r.get("benchmark"),
                    "id": r["id"],
                    "instance_id": r.get("instance_id", r["id"]),
                    "construction": r["construction"],
                    "construction_short": r["construction_short"],
                    "verifier": v,
                    "style": verifier_style(v),
                    "status": r.get(f"{v}_status", "missing"),
                    "status_bucket": r.get(f"{v}_status_bucket", "MISSING"),
                    "outcome": r.get(f"{v}_outcome", "missing"),
                    "time": r.get(f"{v}_time"),
                    "profile_error": r.get("profile_error"),
                    "_yaml_order": r.get("_yaml_order"),
                }
                for k in PROFILE_KEYS:
                    row[k] = r.get(k)
                    row[f"{k}_source"] = r.get(f"{k}_source")
                writer.writerow(row)
    print(f"  saved {path}")


def write_outcome_summary_tables(records: List[Dict[str, Any]], verifiers: List[str], out_dir: Path) -> None:
    statuses = ["UNSAT", "SAT", "TIMEOUT", "ERROR", "MISSING", "OTHER"]

    path1 = out_dir / "outcome_summary_by_verifier.csv"
    with open(path1, "w", newline="") as f:
        fieldnames = ["verifier", *statuses, "total", "unsat_rate", "sat_rate", "timeout_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in verifiers:
            counts = defaultdict(int)
            for r in records:
                counts[r.get(f"{v}_status_bucket", "MISSING")] += 1
            total = sum(counts[s] for s in statuses)
            nonmissing = total - counts["MISSING"]
            writer.writerow({
                "verifier": v,
                **{s: counts[s] for s in statuses},
                "total": total,
                "unsat_rate": counts["UNSAT"] / nonmissing if nonmissing else None,
                "sat_rate": counts["SAT"] / nonmissing if nonmissing else None,
                "timeout_rate": counts["TIMEOUT"] / nonmissing if nonmissing else None,
            })
    print(f"  saved {path1}")

    path2 = out_dir / "outcome_summary_by_construction_verifier.csv"
    with open(path2, "w", newline="") as f:
        fieldnames = ["construction", "construction_short", "verifier", *statuses, "total", "unsat_rate", "sat_rate", "timeout_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in sorted({r["construction"] for r in records}):
            crecs = [r for r in records if r["construction"] == c]
            for v in verifiers:
                counts = defaultdict(int)
                for r in crecs:
                    counts[r.get(f"{v}_status_bucket", "MISSING")] += 1
                total = sum(counts[s] for s in statuses)
                nonmissing = total - counts["MISSING"]
                writer.writerow({
                    "construction": c,
                    "construction_short": construction_short(c),
                    "verifier": v,
                    **{s: counts[s] for s in statuses},
                    "total": total,
                    "unsat_rate": counts["UNSAT"] / nonmissing if nonmissing else None,
                    "sat_rate": counts["SAT"] / nonmissing if nonmissing else None,
                    "timeout_rate": counts["TIMEOUT"] / nonmissing if nonmissing else None,
                })
    print(f"  saved {path2}")


def write_summary_tables(records: List[Dict[str, Any]], verifiers: List[str], out_dir: Path) -> None:
    path1 = out_dir / "summary_by_construction_verifier.csv"
    mean_fields = [f"mean_{k}" for k in PROFILE_KEYS]
    with open(path1, "w", newline="") as f:
        fieldnames = [
            "construction",
            "construction_short",
            "verifier",
            "style",
            "n_total",
            "n_solved",
            "n_timeout",
            "n_counterexample",
            "n_error",
            "solve_rate",
            "timeout_rate",
            "median_completed_time",
            "mean_completed_time",
            *mean_fields,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        constructions = sorted({r["construction"] for r in records})
        for c in constructions:
            crecs = [r for r in records if r["construction"] == c]
            for v in verifiers:
                outs = [r.get(f"{v}_outcome", "missing") for r in crecs]
                times = [
                    r.get(f"{v}_time")
                    for r in crecs
                    if r.get(f"{v}_outcome") in COMPLETED_OUTCOMES and r.get(f"{v}_time") is not None
                ]
                n_total = sum(o != "missing" for o in outs)
                n_solved = sum(o == "solved" for o in outs)
                n_timeout = sum(o == "timeout" for o in outs)
                n_counter = sum(o == "counterexample" for o in outs)
                n_error = sum(o == "error" for o in outs)

                row = {
                    "construction": c,
                    "construction_short": construction_short(c),
                    "verifier": v,
                    "style": verifier_style(v),
                    "n_total": n_total,
                    "n_solved": n_solved,
                    "n_timeout": n_timeout,
                    "n_counterexample": n_counter,
                    "n_error": n_error,
                    "solve_rate": (n_solved / n_total) if n_total else None,
                    "timeout_rate": (n_timeout / n_total) if n_total else None,
                    "median_completed_time": median_or_none(times),
                    "mean_completed_time": mean_or_none(times),
                }
                for k in PROFILE_KEYS:
                    row[f"mean_{k}"] = mean_or_none(r[k] for r in crecs if r.get(k) is not None)
                writer.writerow(row)
    print(f"  saved {path1}")

    path2 = out_dir / "summary_by_style.csv"
    with open(path2, "w", newline="") as f:
        fieldnames = [
            "style",
            "n_records",
            "n_solved",
            "n_timeout",
            "n_counterexample",
            "solve_rate",
            "timeout_rate",
            "median_completed_time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        styles = sorted({verifier_style(v) for v in verifiers})
        for style in styles:
            style_verifiers = [v for v in verifiers if verifier_style(v) == style]
            rows = []
            for r in records:
                for v in style_verifiers:
                    out = r.get(f"{v}_outcome", "missing")
                    t = r.get(f"{v}_time")
                    if out == "missing":
                        continue
                    rows.append((out, t))

            n_total = len(rows)
            n_solved = sum(o == "solved" for o, _ in rows)
            n_timeout = sum(o == "timeout" for o, _ in rows)
            n_counter = sum(o == "counterexample" for o, _ in rows)
            completed_times = [t for o, t in rows if o in COMPLETED_OUTCOMES and t is not None]

            writer.writerow({
                "style": style,
                "n_records": n_total,
                "n_solved": n_solved,
                "n_timeout": n_timeout,
                "n_counterexample": n_counter,
                "solve_rate": (n_solved / n_total) if n_total else None,
                "timeout_rate": (n_timeout / n_total) if n_total else None,
                "median_completed_time": median_or_none(completed_times),
            })
    print(f"  saved {path2}")


# =============================================================================
# Console tables
# =============================================================================

def print_results_table(records: List[Dict[str, Any]], verifiers: List[str]) -> None:
    v_headers = "".join(f"  {v:>13s}" for v in verifiers)
    print(
        f"\n{'ID':<20} {'Type':<18} {'Mhat_min':>10} {'G_IBP':>10} {'U':>8} "
        f"{'Aτ':>8} {'d_eff':>10}{v_headers}"
    )
    print("-" * (80 + 15 * len(verifiers)))

    for r in records:
        vals = [
            format_num(r.get("M_hat_min"), width=10, precision=3),
            format_num(r.get("G_IBP"), width=10, precision=3),
            format_num(r.get("U"), width=8, precision=3),
            format_num(r.get("A_tau"), width=8, precision=3),
            format_num(r.get("d_eff"), width=10, precision=3),
        ]

        v_cols = ""
        for v in verifiers:
            out = r.get(f"{v}_outcome", "missing")
            t = r.get(f"{v}_time")
            if out == "solved":
                cell = f"UNSAT {t:.1f}s" if t is not None else "UNSAT"
            elif out == "timeout":
                cell = "TIMEOUT"
            elif out == "counterexample":
                cell = f"SAT {t:.1f}s" if t is not None else "SAT"
            elif out == "error":
                cell = "ERROR"
            elif out == "missing":
                cell = "—"
            else:
                cell = str(r.get(f"{v}_status", out))[:13]
            v_cols += f"  {cell:>13s}"

        print(
            f"{r['id']:<20} {r['construction_short']:<18} "
            f"{vals[0]} {vals[1]} {vals[2]} {vals[3]} {vals[4]}{v_cols}"
        )


def print_outcome_table(records: List[Dict[str, Any]], verifiers: List[str]) -> None:
    statuses = ["UNSAT", "SAT", "TIMEOUT", "ERROR", "MISSING", "OTHER"]
    print("\nVerifier outcome summary")
    print("-" * 92)
    print(f"{'Verifier':<22}" + "".join(f"{s:>10}" for s in statuses) + f"{'Total':>10}")
    print("-" * 92)
    for v in verifiers:
        counts = defaultdict(int)
        for r in records:
            counts[r.get(f"{v}_status_bucket", "MISSING")] += 1
        total = sum(counts[s] for s in statuses)
        print(f"{v:<22}" + "".join(f"{counts[s]:>10}" for s in statuses) + f"{total:>10}")


def print_outcome_table_by_construction(records: List[Dict[str, Any]], verifiers: List[str]) -> None:
    statuses = ["UNSAT", "SAT", "TIMEOUT", "ERROR", "MISSING", "OTHER"]
    print("\nOutcome summary by construction and verifier")
    print("-" * 118)
    print(f"{'Construction':<24} {'Verifier':<18}" + "".join(f"{s:>9}" for s in statuses) + f"{'Total':>9}")
    print("-" * 118)
    for c in sorted({r["construction"] for r in records}):
        crecs = [r for r in records if r["construction"] == c]
        for v in verifiers:
            counts = defaultdict(int)
            for r in crecs:
                counts[r.get(f"{v}_status_bucket", "MISSING")] += 1
            total = sum(counts[s] for s in statuses)
            print(
                f"{construction_short(c):<24} {v:<18}"
                + "".join(f"{counts[s]:>9}" for s in statuses)
                + f"{total:>9}"
            )


def print_profile_source_warnings(records: List[Dict[str, Any]]) -> None:
    print("\nProfile component source keys")
    print("-" * 72)
    for k in PROFILE_KEYS:
        counts = defaultdict(int)
        for r in records:
            src = r.get(f"{k}_source") or "missing"
            counts[src] += 1
        summary = ", ".join(f"{src}: {n}" for src, n in sorted(counts.items(), key=lambda kv: str(kv[0])))
        print(f"{k:<12} {summary}")

    legacy_deff = sum(1 for r in records if r.get("d_eff_source") == "D_eff_soft")
    if legacy_deff:
        print(
            "\n[warn] d_eff used legacy key D_eff_soft for "
            f"{legacy_deff} records. If your profiler now computes effective gradient "
            "dimensionality, have it emit d_eff directly."
        )


# =============================================================================
# Plots
# =============================================================================

def timeout_penalty_for_record(r: Dict[str, Any]) -> float:
    return 600.0 if r.get("benchmark") == "sweep_all" else 360.0

def plot_construction_heatmap(records: List[Dict[str, Any]], verifiers: List[str], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    heatmap_labels = CONSTRUCTION_DISPLAY_LABELS

    verifier_labels = {
        "abcrown": r"$\alpha,\beta$-CROWN",
        "alpha-beta-crown": r"$\alpha,\beta$-CROWN",
        "ab-crown": r"$\alpha,\beta$-CROWN",
        "marabou": "Marabou",
        "neuralsat": "NeuralSAT",
        "nnenum": "nnenum",
        "pyrat": "PyRAT",
    }

    desired_order = [
        "MEAP",
        "MILP",
        "Input-Corner Stress",
        "Exponential Contraction",
        "Paired-Biases",
        "Linear-Attn",
        "Fixed-Attn",
    ]
    order_rank = {name: i for i, name in enumerate(desired_order)}

    # Keep only constructions that actually appear, but impose the requested order
    # for the benchmark constructions. Any unexpected constructions are appended.
    constructions_seen = list(dict.fromkeys(r["construction"] for r in records))
    constructions = sorted(
        constructions_seen,
        key=lambda c: (
            order_rank.get(heatmap_labels.get(c, construction_short(c)), len(order_rank)),
            constructions_seen.index(c),
        ),
    )
    print(constructions)
    c_labels = [heatmap_labels.get(c, construction_short(c)) for c in constructions]
    v_labels = [verifier_labels.get(v.lower(), v) for v in verifiers]

    data = {}
    for ci, cname in enumerate(constructions):
        crecs = [r for r in records if r["construction"] == cname]
        for vi, vname in enumerate(verifiers):
            counts = {
                "solved": 0,
                "timeout": 0,
                "counterexample": 0,
                "error": 0,
                "other": 0,
                "total": 0,
            }
            unsat_times = []
            for r in crecs:
                outcome = r.get(f"{vname}_outcome", "missing")
                t = r.get(f"{vname}_time")

                # For the construction-by-verifier heatmap, missing run results
                # are treated as errors so they count in the denominator and
                # appear as failures rather than disappearing from the cell.
                if outcome == "missing":
                    outcome = "error"

                counts["total"] += 1
                counts[outcome] = counts.get(outcome, 0) + 1
                if outcome == "solved" and t is not None:
                    unsat_times.append(t)
                elif outcome == "timeout":
                    unsat_times.append(timeout_penalty_for_record(r))
            counts["mean_unsat_time"] = mean_or_none(unsat_times)
            data[(ci, vi)] = counts

    fig, ax = plt.subplots(
        figsize=(max(4, 3 * len(verifiers)), max(3, 0.52 * len(constructions)))
    )

    for ci in range(len(constructions)):
        for vi in range(len(verifiers)):
            c = data.get(
                (ci, vi),
                {"solved": 0, "timeout": 0, "counterexample": 0, "error": 0, "other": 0, "total": 0},
            )
            total = c["total"]
            solved = c["solved"]

            if total == 0:
                color = "#EEEEEE"
                text = "—"
            else:
                # Preserve the original coloring logic.
                if solved == total:
                    color = "#C8E6C9"
                elif c.get("counterexample", 0) == total:
                    color = "#BBDEFB"
                elif c["timeout"] == total:
                    color = "#FFCDD2"
                elif solved + c.get("counterexample", 0) > 0:
                    color = "#FFF9C4"
                else:
                    color = "#FFCDD2"

                mt = c.get("mean_unsat_time")
                avg_text = f"Avg: {mt:.1f}s" if mt is not None else "avg —"
                text = f"{solved}/{total} \n{avg_text}"

            ax.add_patch(
                plt.Rectangle((vi, ci), 1, 1, facecolor=color, edgecolor="white", linewidth=2)
            )
            ax.text(
                vi + 0.5,
                ci + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                linespacing=1.25,
            )

    ax.set_xlim(0, len(verifiers))
    ax.set_ylim(0, len(constructions))
    ax.set_xticks([i + 0.5 for i in range(len(verifiers))])
    ax.set_xticklabels(v_labels, fontsize=18, fontweight="bold")
    ax.set_yticks([i + 0.5 for i in range(len(constructions))])
    ax.set_yticklabels(c_labels, fontsize=18, fontweight="bold")
    ax.set_xlabel("Verifier", fontsize=22, fontweight="bold", labelpad=14)
    ax.set_ylabel("Veristress-GT Constructor", fontsize=22, fontweight="bold", labelpad=18)
    ax.invert_yaxis()

    # Title intentionally omitted for the heatmap.

    legend_patches = [
        mpatches.Patch(facecolor="#C8E6C9", edgecolor="gray", label="All UNSAT"),
        mpatches.Patch(facecolor="#FFF9C4", edgecolor="gray", label="Mixed"),
        mpatches.Patch(facecolor="#FFCDD2", edgecolor="gray", label="All Timeout/Error"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=22
    )

    fig.subplots_adjust(left=0.26, right=0.84, top=0.94, bottom=0.25)
    fig.tight_layout()
    path = out_dir / "construction_verifier_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"  saved {path}")


def _runtime_color_values(recs: List[Dict[str, Any]], verifier: str) -> Tuple[List[float], float, float]:
    vals = []
    for r in recs:
        t = r.get(f"{verifier}_time")
        out = r.get(f"{verifier}_outcome")
        if out in COMPLETED_OUTCOMES and t is not None and t > 0:
            vals.append(t)
    if not vals:
        return [], 0.0, 1.0
    return vals, float(min(vals)), float(max(vals))

def plot_construction_trajectories(records: List[Dict[str, Any]], verifiers: List[str], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    constructions = sorted(
        {r["construction"] for r in records},
        key=lambda c: construction_short(c),
    )

    for cname in constructions:
        crecs = sorted(
            [r for r in records if r["construction"] == cname],
            key=lambda r: (r.get("_yaml_order", float("inf")), r["id"]),
        )
        if len(crecs) < 3:
            continue

        component_specs = [(key, label, COMPONENT_COLORS.get(key, "#333333")) for key, label in PROFILE_COMPONENTS]
        x = np.arange(len(crecs))

        # Only plot profile-component trajectories plus the categorical outcome strip.
        # Do not allocate or draw a runtime panel.
        fig, axes = plt.subplots(
            len(component_specs) + 1,
            1,
            figsize=(12, 1.7 * len(component_specs) + 2.6),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0] * len(component_specs) + [1.45], "hspace": 0.12},
        )
        axes = np.atleast_1d(axes)

        for ax, (comp, label, color) in zip(axes[:len(component_specs)], component_specs):
            ys = [r.get(comp) for r in crecs]
            ys_plot = [np.nan if y is None else y for y in ys]
            ax.plot(x, ys_plot, marker="o", linewidth=2, color=color, markersize=5)
            ax.set_ylabel(label, rotation=0, labelpad=25, fontsize=15, fontweight="bold")
            ax.grid(True, alpha=0.3)

        display_name = construction_short(cname)
        axes[0].set_title(
            f"{display_name}: Component Trajectories",
            fontsize=20,
            fontweight="bold",
            pad=10,
        )

        ax_outcome = axes[-1]
        for yi, verifier in enumerate(verifiers):
            for xi, r in enumerate(crecs):
                out = r.get(f"{verifier}_outcome", "missing")
                if out == "solved":
                    color = "#4CAF50"
                elif out == "timeout":
                    color = "#000000"
                elif out == "counterexample":
                    color = "#D32F2F"
                elif out == "error":
                    color = "#9E9E9E"
                else:
                    color = "#EEEEEE"

                ax_outcome.scatter(
                    xi,
                    yi,
                    c=color,
                    marker="s",
                    s=120,
                    edgecolors="white",
                    linewidth=0.6,
                )

        ax_outcome.set_yticks(np.arange(len(verifiers)))
        ax_outcome.set_yticklabels(verifiers)
        ax_outcome.set_xlabel("Instance Number in Config")
        ax_outcome.set_ylabel("Verifier")
        ax_outcome.set_xlim(-0.5, len(crecs) - 0.5)
        ax_outcome.grid(True, alpha=0.2, axis="x")

        legend_handles = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#4CAF50",
                   markeredgecolor="white", markersize=8, label="UNSAT/verified"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#000000",
                   markeredgecolor="white", markersize=8, label="Timeout"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#D32F2F",
                   markeredgecolor="white", markersize=8, label="SAT/cex"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#9E9E9E",
                   markeredgecolor="white", markersize=8, label="Error"),
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, .07),
            ncol=4,
            fontsize=15,
            title="Outcome",
        )

        fig.tight_layout()
        path = out_dir / f"trajectory_{display_name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


# =============================================================================
# Modeling
# =============================================================================

def _build_feature_matrix(rows: List[Dict[str, Any]], add_interactions: bool = True) -> Tuple[np.ndarray, List[str], List[int]]:
    X_rows = []
    kept = []
    names: List[str] = list(PROFILE_KEYS)

    for i, r in enumerate(rows):
        vals = [r.get(f) for f in PROFILE_KEYS]
        if any(v is None for v in vals):
            continue
        vals = [float(v) for v in vals]
        row = list(vals)

        if add_interactions:
            m, g, u, a, d = vals
            row.extend([
                g * a,        # relaxation gap × realized local-region complexity
                u * a,        # nonlinear exposure × realized local-region complexity
                g * d,        # relaxation gap × distributed sensitivity
                a * d,        # local regions × effective input dimension
                u * d,        # nonlinear exposure × effective input dimension
                m * g,        # margin scale × relaxation loss
            ])
            names = list(PROFILE_KEYS) + [
                "G_IBP_x_A_tau",
                "U_x_A_tau",
                "G_IBP_x_d_eff",
                "A_tau_x_d_eff",
                "U_x_d_eff",
                "M_hat_min_x_G_IBP",
            ]

        X_rows.append(row)
        kept.append(i)

    if not X_rows:
        return np.empty((0, 0)), [], []
    return np.array(X_rows, dtype=float), names, kept


def fit_models(records: List[Dict[str, Any]], verifiers: List[str], out_dir: Path) -> None:
    try:
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import r2_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        print(f"[warn] sklearn unavailable; skipping model fitting ({e})")
        return

    model_rows = []
    model_json: Dict[str, Any] = {}

    for verifier in verifiers:
        class_rows = []
        y_class = []
        for r in records:
            out = r.get(f"{verifier}_outcome", "missing")
            if out in COMPLETED_OUTCOMES:
                class_rows.append(r)
                y_class.append(1)
            elif out == "timeout":
                class_rows.append(r)
                y_class.append(0)

        verifier_result: Dict[str, Any] = {"verifier": verifier, "style": verifier_style(verifier)}

        if len(class_rows) >= 12 and len(set(y_class)) >= 2:
            X, X_names, kept = _build_feature_matrix(class_rows, add_interactions=True)
            y = np.array([y_class[i] for i in kept], dtype=int)
            if len(y) >= 12 and len(set(y)) >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=0, stratify=y
                )
                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=500)),
                ])
                clf.fit(X_train, y_train)
                probs = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, probs)

                coef = clf.named_steps["model"].coef_[0]
                intercept = float(clf.named_steps["model"].intercept_[0])
                verifier_result["logistic_auc"] = float(auc)
                verifier_result["logistic_n"] = int(len(y))
                verifier_result["logistic_intercept"] = intercept
                verifier_result["logistic_coefficients"] = {name: float(val) for name, val in zip(X_names, coef)}

        reg_rows = []
        y_reg = []
        for r in records:
            out = r.get(f"{verifier}_outcome", "missing")
            t = r.get(f"{verifier}_time")
            if out in COMPLETED_OUTCOMES and t is not None and t > 0:
                reg_rows.append(r)
                y_reg.append(math.log1p(t))

        if len(reg_rows) >= 12:
            X, X_names, kept = _build_feature_matrix(reg_rows, add_interactions=True)
            y = np.array([y_reg[i] for i in kept], dtype=float)
            if len(y) >= 12:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
                reg = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ])
                reg.fit(X_train, y_train)
                preds = reg.predict(X_test)
                r2 = r2_score(y_test, preds)

                coef = reg.named_steps["model"].coef_
                intercept = float(reg.named_steps["model"].intercept_)
                verifier_result["runtime_r2"] = float(r2)
                verifier_result["runtime_n"] = int(len(y))
                verifier_result["runtime_intercept"] = intercept
                verifier_result["runtime_coefficients"] = {name: float(val) for name, val in zip(X_names, coef)}

        model_rows.append({
            "verifier": verifier,
            "style": verifier_style(verifier),
            "logistic_auc": verifier_result.get("logistic_auc"),
            "logistic_n": verifier_result.get("logistic_n"),
            "runtime_r2": verifier_result.get("runtime_r2"),
            "runtime_n": verifier_result.get("runtime_n"),
        })
        model_json[verifier] = verifier_result

    json_path = out_dir / "model_summaries.json"
    json_path.write_text(json.dumps(sanitize_for_json(model_json), indent=2))
    print(f"  saved {json_path}")

    csv_path = out_dir / "model_summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["verifier", "style", "logistic_auc", "logistic_n", "runtime_r2", "runtime_n"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in model_rows:
            writer.writerow(row)
    print(f"  saved {csv_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze verifier behavior vs updated five-component difficulty profile.")
    ap.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help=(
            "Benchmark directory containing manifest.json and optional "
            "difficulty_profiles.json. May be supplied multiple times."
        ),
    )
    ap.add_argument(
        "--benchmarks",
        nargs="+",
        default=[],
        help="Multiple benchmark directories. Equivalent to repeating --benchmark.",
    )
    ap.add_argument("--runs", nargs="+", required=True, help="Verifier run directories")
    ap.add_argument("--out-dir", default=None, help="Output directory")
    ap.add_argument("--gap-eta", type=float, default=1e-8, help="Numerical eta used when computing G_IBP from M_hat_min and L_IBP")
    ap.add_argument("--bins", type=int, default=5, help="Quantile bins for interaction heatmaps")
    ap.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    ap.add_argument("--skip-models", action="store_true", help="Skip sklearn model fitting")
    args = ap.parse_args()

    benchmark_dirs = [Path(p) for p in (args.benchmark + args.benchmarks)]
    if not benchmark_dirs:
        ap.error("Provide at least one --benchmark or --benchmarks entry.")

    run_dirs = [Path(r) for r in args.runs]

    if len(benchmark_dirs) == 1:
        out_dir = Path(args.out_dir) if args.out_dir else benchmark_dirs[0] / "analysis"
    else:
        out_dir = Path(args.out_dir) if args.out_dir else Path("analysis_multi_benchmark")
    ensure_dir(out_dir)

    print("Merging data...")
    if len(benchmark_dirs) == 1:
        records, verifiers = merge_data(benchmark_dirs[0], run_dirs, gap_eta=args.gap_eta)
        for r in records:
            r["benchmark"] = benchmark_dirs[0].name
            r["instance_id"] = r["id"]
    else:
        records, verifiers = merge_data_multi(benchmark_dirs, run_dirs, gap_eta=args.gap_eta)
    print(
        f"  {len(records)} instances, {len(verifiers)} verifiers: {verifiers} "
        f"across {len(benchmark_dirs)} benchmark(s)"
    )

    if not records:
        print("No merged records found.")
        sys.exit(1)

    print_results_table(records, verifiers)
    print_outcome_table(records, verifiers)
    print_outcome_table_by_construction(records, verifiers)
    print_profile_source_warnings(records)

    print("\nWriting merged data and summary tables...")
    write_merged_json(records, out_dir)
    write_longform_csv(records, verifiers, out_dir)
    write_summary_tables(records, verifiers, out_dir)
    write_outcome_summary_tables(records, verifiers, out_dir)

    if not args.skip_plots:
        print("\nGenerating plots...")
        plot_construction_heatmap(records, verifiers, out_dir)
        plot_construction_trajectories(records, verifiers, out_dir)

    if not args.skip_models:
        print("\nFitting simple interpretable models...")
        fit_models(records, verifiers, out_dir)

    print(f"\nDone. Analysis saved to {out_dir}")


if __name__ == "__main__":
    main()
