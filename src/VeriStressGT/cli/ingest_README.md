# Ingesting VNN-COMP benchmarks

`VeriStressGT` can run its verifier pipeline on external benchmarks from
VNN-COMP. Instances from VNN-COMP lack a-priori ground-truth robustness labels
(that is precisely the gap this project targets), so ingested instances are
flagged `is_robust: null, ground_truth_source: "vnncomp_unknown"` in their
per-instance `meta.json`. The difficulty-profile pipeline does not depend on
the label, so profiling and runtime-correlation analysis still work.

## Quick start (MNIST-FC from VNN-COMP 2022)

```bash
# 1. Clone the VNN-COMP benchmark repo and expand compressed files.
git clone https://github.com/ChristopherBrix/vnncomp2022_benchmarks.git ~/vnncomp2022_benchmarks
cd ~/vnncomp2022_benchmarks && ./setup.sh   # extracts .gz files in place
cd -

# 2. Ingest mnist_fc into VeriStressGT format.
python -m VeriStressGT.cli.ingest_vnncomp \
  --src ~/vnncomp2022_benchmarks/benchmarks/mnist_fc \
  --out_dir ./benchmarks/vnncomp_mnist_fc \
  --overwrite

# 3. Run verifiers the same way you would on native constructions.
python -m VeriStressGT.cli.verify_benchmark \
  --benchmark ./benchmarks/vnncomp_mnist_fc \
  --verifier abcrown \
  --out_dir ./runs/abcrown_vnncomp_mnist_fc \
  --timeout 300 \
  --abcrown_config configs/abcrown_basic.yaml \
  --overwrite

python -m VeriStressGT.cli.verify_benchmark \
  --benchmark ./benchmarks/vnncomp_mnist_fc \
  --verifier neuralsat \
  --out_dir ./runs/neuralsat_vnncomp_mnist_fc \
  --timeout 300 \
  --overwrite
```

## Subsetting

MNIST-FC has three network sizes (256x2, 256x4, 256x6) and many properties.
For faster iteration, cap instances or skip the hardest ones:

```bash
# Keep only instances with suggested timeout ≤ 120s, up to 20 instances.
python -m VeriStressGT.cli.ingest_vnncomp \
  --src ~/vnncomp2022_benchmarks/benchmarks/mnist_fc \
  --out_dir ./benchmarks/vnncomp_mnist_fc_small \
  --max_timeout 120 \
  --max_instances 20 \
  --overwrite
```

## What gets copied vs. recorded

For each row `(onnx_rel, vnnlib_rel, timeout_s)` in `instances.csv`, the tool:

- Copies the ONNX file to `instances/<id>/model.onnx`.
- Copies the VNNLIB file to `instances/<id>/spec.vnnlib`.
- Writes `instances/<id>/meta.json` with:
  - `construction: "vnncomp.ingested"` (sentinel so analyses can filter on it),
  - `is_robust: null`, `ground_truth_source: "vnncomp_unknown"`,
  - `sha256` of both files (for reproducibility),
  - `vnncomp: {onnx_rel, vnnlib_rel, timeout_s, source_lineno}` (provenance).
- Writes a top-level `manifest.json` with `mode: "vnncomp_ingest"` and a
  `source` block recording the source `instances.csv` path, its sha256, and
  applied filters.

The VNN-COMP suggested timeout is stored in the meta but **not** used to gate
verifier runs — `verify_benchmark --timeout` is the single source of truth for
per-instance wall-clock limits, as with native instances.

## Downstream analysis

Because `construction == "vnncomp.ingested"` is a stable string, you can carve
ingested instances out of runtime analyses when comparing against native
constructions, e.g. in `correlate_profile_runtime.py`:

```python
# In your analysis script
if inst["construction"] == "vnncomp.ingested":
    group = "vnncomp_mnist_fc"
else:
    group = inst["construction"]
```

The Spearman correlations between profile components and verifier runtime can
be computed on the VNN-COMP cohort exactly as for native constructions, since
profile components are defined purely in terms of `(f, x0, Bε)` and make no
assumption about where those came from.

## Adding other benchmarks

Any benchmark folder with an `instances.csv` in the canonical VNN-COMP format
(`onnx_rel_path, vnnlib_rel_path, timeout_seconds`) is supported as-is. For
ACAS Xu, CIFAR10-ResNet, NN4Sys, etc., the same command works — just point
`--src` at `<vnncomp_repo>/benchmarks/<name>`.