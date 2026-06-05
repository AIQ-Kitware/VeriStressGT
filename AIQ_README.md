## AIQ Flow
To setup/run, complete the following:

### Benchmark Setup
```bash
git clone --recursive git@github.com:dtroxell19/VeriStressGT.git
cd VeriStressGT
git fetch origin
git checkout -b evaluation-card origin/evaluation-card
bash scripts/bootstrap.sh
conda activate VeriStressGT
```

### MAGNET Setup
```bash
pip install git+https://github.com/AIQ-Kitware/aiq-magnet.git

# Additional deps needed by Gurobi-based constructions
pip install sortedcontainers coloredlogs termcolor beartype gurobipy
```

### α-β-CROWN Environment Variables
From root directory of repo:
```bash
export ABCROWN_VNNCOMP2024_DIR="$(pwd)/src/VeriStressGT/verifiers/alpha-beta-CROWN"
export ABCROWN_CONDA_ENV="alpha-beta-crown"
```

### Run
```bash
magnet evaluate cards/evaluation.yaml
```

### Expected Results

**Output files** written alongside `results.json`:

| File | Description |
|------|-------------|
| `mini_sweep_summary.csv` | Per-verifier solved/total counts and correct fraction |
| `construction_heatmap.png` | Solved/total per construction × verifier, with avg wall time |
| `timeout_auc_heatmap.png` | Timeout AUC per difficulty component × verifier |
| `timeout_auc.csv` | Numeric timeout AUC values |

**1. Correctness invariant:** Every instance is UNSAT by construction (provably robust). No verifier should ever return SAT — a SAT result indicates a soundness error in that verifier. Some verifiers will timeout or return error (i.e. unsupported) results.

**2. Timeout AUC Values Typically Above 0.5:** For each (component, verifier) pair, timeout AUC is the probability that a randomly chosen timed-out instance has a higher component value than a randomly chosen solved instance (AUROC of the component as a timeout predictor). A value > 0.5 means harder instances — as measured by that component — are more likely to cause a timeout, which is the expected direction. Values near 0.5 indicate the component doesn't predict difficulty for that verifier. We expect the majority of verifier, component combinations to have a value over 0.5.

---

### Sample Run Results

**Machine:** MacBook Pro, Apple Silicon, 8 GB RAM, macOS Sonoma 14.7.5  
**Config:** 50 instances × {pyrat, nnenum}, 60s timeout per instance

**Verifier summary:**

| Verifier | UNSAT | Timeout | Error | SAT |
|----------|------:|--------:|------:|----:|
| alpha,beta-crown    | 49   | 1      | 0     | 0   |
| pyrat    | 45    | 4       | 1     | 0   |
| nnenum   | 13     | 10       | 27    | 0   |

No verifier returned SAT. nnenum errors are structural (unsupported op types on meap, corners, and attention constructions) rather than difficulty-related. pyrat timeouts occur on the hardest `pb_cnn` (large `num_pairs`) and `milp`/`meap` instances at the top of the difficulty range.

**Timeout AUC heatmap** (pyrat + nnenum, 60s timeout):

![Timeout AUC Heatmap](assets/timeout_auc_heatmap.png)
