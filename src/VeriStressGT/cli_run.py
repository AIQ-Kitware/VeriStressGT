"""
VeriStressGT CLI runner — with GPU support for α-β-CROWN and NeuralSAT.

Usage examples:
    # CPU (default)
    VeriStressGT-verify --method abcrown-vnncomp2024 --onnx model.onnx --vnnlib spec.vnnlib --config_path cfg.yaml

    # GPU
    VeriStressGT-verify --method abcrown-vnncomp2024 --device cuda --onnx model.onnx --vnnlib spec.vnnlib --config_path cfg.yaml
    VeriStressGT-verify --method neuralsat --device cuda --onnx model.onnx --vnnlib spec.vnnlib

GPU-capable verifiers:   abcrown-vnncomp2024, neuralsat
CPU-only verifiers:      nnenum, marabou, nnv  (--device flag ignored)
"""

import argparse
import subprocess
import tempfile
import os
from pathlib import Path

import yaml  # PyYAML — already a transitive dep via abcrown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_yaml_device(src_config_path: str, device: str) -> str:
    """
    Load a YAML config, override general.device, write to a temp file.

    Returns the path of the patched temp file.  The caller is responsible for
    cleanup (or just let the OS handle it on process exit — fine for short-lived
    verify runs).
    """
    with open(src_config_path) as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    if "general" not in cfg:
        cfg["general"] = {}
    cfg["general"]["device"] = device

    # Write next to the original so relative paths inside the yaml still resolve
    src = Path(src_config_path)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        dir=src.parent,
        prefix=f"_gpu_patched_{src.stem}_",
        delete=False,
    )
    yaml.dump(cfg, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main_verify():
    ap = argparse.ArgumentParser(
        description="Run a neural-network verifier on an ONNX+VNNLIB instance."
    )

    ap.add_argument("--method", required=True,
                    choices=["abcrown-vnncomp2024", "nnenum", "neuralsat", "marabou", "nnv"],
                    help="Which verifier backend to use")
    ap.add_argument("--vnnlib", required=True, help="Path to .vnnlib specification file")
    ap.add_argument("--timeout", type=float, default=300, help="Timeout in seconds")
    ap.add_argument("--output_file_name", default=None, help="Path to output file (nnenum only)")

    # ── α-β-CROWN specific ──────────────────────────────────────────────────
    ap.add_argument("--onnx", default=None, help="Path to .onnx model (abcrown / nnenum / neuralsat)")
    ap.add_argument("--config_path", default=None, help="Path to YAML config (abcrown only)")
    ap.add_argument("--abcrown-repo", default=None,
                    help="Path to alpha-beta-CROWN_vnncomp2024 repo "
                         "(optional; else uses $ABCROWN_VNNCOMP2024_DIR)")
    ap.add_argument("--abcrown-env", default=None,
                    help="Conda env name for abcrown (optional; else uses $ABCROWN_CONDA_ENV)")

    # ── GPU flag ─────────────────────────────────────────────────────────────
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Device for GPU-capable verifiers (abcrown-vnncomp2024, neuralsat). "
                         "Ignored by nnenum, marabou, nnv which are CPU-only.")

    args = ap.parse_args()

    # =========================================================================
    # α-β-CROWN  (GPU-capable)
    # =========================================================================
    if args.method == "abcrown-vnncomp2024":
        if args.onnx is None:
            raise SystemExit("For --method abcrown-vnncomp2024 you must pass --onnx path/to/model.onnx")
        if args.config_path is None:
            raise SystemExit("For --method abcrown-vnncomp2024 you must pass --config_path path/to/config.yaml")

        from VeriStressGT.verifiers.abcrown_vnncomp2024 import run_abcrown_vnncomp2024

        # Patch the YAML to override device if the user asked for GPU
        effective_config = args.config_path
        patched_tmp = None
        if args.device != "cpu":
            print(f"[VeriStressGT] Patching abcrown config: general.device → {args.device}")
            patched_tmp = _patch_yaml_device(args.config_path, args.device)
            effective_config = patched_tmp

        try:
            res = run_abcrown_vnncomp2024(
                onnx_path=args.onnx,
                vnnlib_path=args.vnnlib,
                timeout_s=args.timeout,
                abcrown_repo_dir=args.abcrown_repo,
                conda_env=args.abcrown_env,
                config_path=effective_config,
            )
        finally:
            # Clean up the temp patched config
            if patched_tmp and os.path.exists(patched_tmp):
                os.unlink(patched_tmp)

        print(res.raw_output)

        if res.status == "robust":
            raise SystemExit(0)
        if res.status == "violated":
            raise SystemExit(1)
        raise SystemExit(2)

    # =========================================================================
    # NeuralSAT  (GPU-capable)
    # =========================================================================
    elif args.method == "neuralsat":
        if args.onnx is None:
            raise SystemExit("For --method neuralsat you must pass --onnx path/to/model.onnx")

        neuralsat_repo = os.environ.get("NEURALSAT_DIR", "neuralsat")

        cmd = [
            "python3", "-m", "neuralsat.main",
            "--net", str(args.onnx),
            "--spec", str(args.vnnlib),
            "--timeout", str(int(args.timeout)),
            "--device", args.device,   # NeuralSAT accepts --device cpu|cuda
        ]

        print(f"[VeriStressGT] Running NeuralSAT on {args.device}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=neuralsat_repo if os.path.isdir(neuralsat_repo) else None,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    # =========================================================================
    # nnenum  (CPU-only — --device flag silently ignored)
    # =========================================================================
    elif args.method == "nnenum":
        if args.device == "cuda":
            print("[VeriStressGT] Warning: nnenum is CPU-only; --device cuda ignored.")

        output_file_name = args.output_file_name or "out.txt"

        cmd = [
            "python3", "-m", "nnenum.nnenum",
            str(args.onnx),
            str(args.vnnlib),
            str(args.timeout),
            output_file_name,
            "1",              # number of processes
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    # =========================================================================
    # Marabou  (CPU-only — --device flag silently ignored)
    # =========================================================================
    elif args.method == "marabou":
        if args.device == "cuda":
            print("[VeriStressGT] Warning: Marabou is CPU-only; --device cuda ignored.")

        # Marabou is typically invoked via its Python API or maraboupy.
        # Adjust the command below to match your local Marabou installation.
        marabou_dir = os.environ.get("MARABOU_DIR", "Marabou")
        cmd = [
            "python3",
            os.path.join(marabou_dir, "resources", "runMarabou.py"),
            "--onnx", str(args.onnx),
            "--property", str(args.vnnlib),
            "--timeout", str(int(args.timeout)),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    # =========================================================================
    # NNV  (CPU-only / MATLAB — --device flag silently ignored)
    # =========================================================================
    elif args.method == "nnv":
        if args.device == "cuda":
            print("[VeriStressGT] Warning: NNV is CPU/MATLAB-based; --device cuda ignored.")

        nnv_script = Path(__file__).parent / "verifiers" / "run_nnv_verify.m"
        cmd = [
            "matlab", "-nodisplay", "-nosplash", "-r",
            f"run_nnv_verify('{args.onnx}', '{args.vnnlib}', {int(args.timeout)}); exit;",
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)


# ---------------------------------------------------------------------------
# ONNX export entry point (unchanged)
# ---------------------------------------------------------------------------

def main_export_onnx():
    """Placeholder — keep existing implementation."""
    raise NotImplementedError("main_export_onnx: wire up your existing export logic here.")


if __name__ == "__main__":
    main_verify()
