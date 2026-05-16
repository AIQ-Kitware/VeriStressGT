# VeriStress-GT

Benchmark framework for neural network verification. Generates provably robust (ONNX, VNNLIB) pairs with known ground-truth properties, then runs third-party verifiers on them in a standardized way.

## Setup

**Requires:** Python 3.9+, git. Optional: CUDA GPU, [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) (for MILP construction), MATLAB (for NNV verifier).

```bash
git clone --recursive https://github.com/<YOUR_ORG>/VeriStressGT.git
cd VeriStressGT
bash scripts/bootstrap.sh
conda activate VeriStressGT
source .env
cd src/VeriStressGT
```

The bootstrap script handles everything: installs Miniconda if needed, pulls submodules, installs Python packages, creates verifier conda environments, and writes your `.env`. If you just want instance generation without verifier setup: `pip install -e '.[generate]'`

**Run `VeriStressGT-doctor` at any time to check what's working and how to fix installation**

## Usage

**Generate a benchmark:**
Define instances in a YAML spec (see `src/VeriStressGT/configs/` for examples). Then generate:

```bash
python3 -m VeriStressGT.cli.create_benchmark \
  --spec configs/test_benchmark.yaml \
  --out_dir ./benchmarks/test_benchmark \
  --overwrite
```

**Run a verifier on it** (no env switching needed — conda isolation is automatic):

```bash
python3 -m VeriStressGT.cli.verify_benchmark \
  --benchmark ./benchmarks/test_benchmark \
  --verifier abcrown \
  --out_dir ./runs/abcrown_run \
  --timeout 300 \
  --abcrown_config configs/abcrown_basic.yaml \
  --overwrite
```

Results are written to `results.jsonl` and `summary.json`/`summary.csv`. Each instance gets stdout/stderr logs in the `logs/` subdirectory.

To run a different verifier on the same benchmark:

```bash
python -m VeriStressGT.cli.verify_benchmark \
  --benchmark ./benchmarks/my_benchmark \
  --verifier neuralsat \
  --out_dir ./runs/neuralsat_run \
  --timeout 300
```

Results go to `results.jsonl` and `summary.csv`.

## Available constructions

| Name | Type | What it stresses |
|------|------|-----------------|
| `mlp_relu.embedded_projection` | MLP | Baseline provably-robust via input projection |
| `mlp_relu.meap` | MLP | Mutually exclusive activation patterns |
| `mlp_relu.corners` | MLP | Convex corner-based certificate |
| `mlp_relu.activation_test` | MLP | Activation pattern testing |
| `mlp_relu.milp.exact_radius` | MLP | MILP-computed exact robustness radius |
| `cnn.cnn_paired_bias` | CNN | Exploits CROWN's independent neuron relaxation |
| `cnn.deep_contractive_cnn` | CNN | Easy for CROWN, hard for CDCL verifiers |
| `attention.fixed_pattern` | Attention | Softmax attention with pattern stability certificate |
| `attention.linear_dominance` | Attention | Linear attention with key-dominance certificate |
| `polynomial.algebraic_boundary` | Polynomial Net | Small Margin |

## Verifier setup

All handled by `bootstrap.sh`, or individually via `make install-<verifier>`:

| Verifier | GPU | Setup |
|----------|-----|-------|
| α-β-CROWN | ✓ | `make install-abcrown` |
| NeuralSAT | ✓ | `make install-neuralsat` |
| nnenum | ✗ | `make install-nnenum` |
| Marabou | ✗ | `make install-marabou` (requires cmake) |
| NNV | ✗ | `make install-nnv` (requires MATLAB) |
| PyRAT | ✗ | `make install-pyrat` |

## Adding a new constructor

Place a module under `src/VeriStressGT/robust_constructions/` with:

```python
CONSTRUCTION_NAME = "category.my_method"

def add_args(parser):
    parser.add_argument("--my_param", type=float, default=1.0)

def run(args):
    # Build model, export to args.onnx_path and args.vnnlib_path
    ...
```

Discovery is automatic. Subdirectories need an `__init__.py`.

## Repo structure

```
src/VeriStressGT/
    cli/                    create_benchmark, verify_benchmark, doctor
    robust_constructions/   mlp_relu/, cnn/, attention/
    verifier_adapters/      per-verifier CLI adapters
    verifiers/              git submodules for each verifier
    utils/                  ONNX export + VNNLIB generation
```