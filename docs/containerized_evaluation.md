# Running the evaluation card in a container

## How Kitware runs this card

`python -m magnet.evaluation_new` reads the card, turns its `kwdagger:` block
into a DAG, and schedules the DAG through kwdagger. For
`cards/evaluation_kwdagger.yaml` that is a single node, `mini_sweep`, whose
command is `bash aiq/run_mini_sweep_env.sh`.

The node runs as one `docker run` of the image built from the `Dockerfile` at
the repository root. The checkout, called `$REPO` below, is bind-mounted at its
own absolute path and the node's working directory is set there. `PYTHONPATH`
is forwarded into the container, so the node runs the mounted checkout, not the
copy baked into the image. Results land under `--output_path`.

The backend is tmux on a workstation and Slurm on a cluster. Per-node leasing
does not apply: this card does no live inference.

This is the one card in the programme whose answer depends on the hardware.
The claim gates on a per-instance timeout, and a timeout scores as INCORRECT,
so run it on a quiet machine. abcrown is CPU-bound, so the contention that
matters is CPU. Never compare counts across machines.

## Build

The verifier submodules must be populated first:

```bash
cd $REPO
git submodule update --init --recursive
docker build -t veristressgt-gpu .
```

The image is large (about 10 GB): a second conda tree holds the
alpha-beta-CROWN environment, because alpha-beta-CROWN invokes itself through
`conda run -n alpha-beta-crown`.

`MAGNET_REF` pins the aiq-magnet commit the evaluator uses. It is published on
`AIQ-Kitware/aiq-magnet` before Friday 2026-09-05. Until then:

```bash
docker build --build-arg MAGNET_REF=main -t veristressgt-gpu .
```

### Known gap: pyrat, and nnenum until it is run

alpha-beta-CROWN works from this image. pyrat and nnenum are plain pip
installs of their submodule directories, and the outcome must be checked
after every build:

```bash
docker run --rm veristressgt-gpu python -c "import nnenum, swiglpk"
docker run --rm veristressgt-gpu pyrat --help
```

In the last build we checked (2026-09-02), the first command passes and the
second fails with `No module named 'pyrat.main'`: the editable install
registers `pyrat` as a namespace package instead of the real package under
`src/VeriStressGT/verifiers/pyrat/pyrat`. A run from this image alone
therefore scores pyrat 0/50 with errors, not timeouts. nnenum imports but has
not yet been run through a sweep from this image. Our reproduction worked
around both by also mounting a conda tree that carries working `pyrat` and
`nnenum` envs and listing it in `VST_CONDA_ROOTS` (see below). Making pyrat
resolve inside the image is the open item on this Dockerfile.

## Reproduce the June dry run

On the host you need the same aiq-magnet, docker with the NVIDIA container
toolkit, and tmux:

```bash
pip install "aiq-magnet[optional] @ git+https://github.com/AIQ-Kitware/aiq-magnet@5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13"
export PYTHONPATH=$REPO/src:$REPO
```

No data is fetched: the 50 UNSAT instances are committed under
`mini_sweep_bench/` with their `manifest.json`, which is what makes Gurobi
unnecessary. The verifier writes `spec.vnnlib.compiled` beside the inputs, so
the run works on a copy of the bench in the run directory, never the committed
one.

The node wrapper resolves the conda roots and interpreter from environment
variables that are forwarded into the container. With this image alone:

```bash
export VST_CONDA_ROOT=/opt/abcrown-conda
export VST_CONDA_ROOTS=/opt/abcrown-conda:/opt/conda
export VST_PYTHON=/opt/conda/bin/python
export ABCROWN_CONDA_ENV=alpha-beta-crown
export ABCROWN_VNNCOMP2024_DIR=$REPO/src/VeriStressGT/verifiers/alpha-beta-CROWN
```

To also run pyrat and nnenum today, append a conda root that has working envs
for them to `VST_CONDA_ROOTS` and add that directory to `--container_mounts`.

Then:

```bash
cd $REPO
RUN=$REPO/runs/veristressgt_mini_sweep
mkdir -p "$RUN" && cp -r mini_sweep_bench "$RUN/mini_sweep_bench"
python -m magnet.evaluation_new cards/evaluation_kwdagger.yaml \
    --output_path "$RUN" \
    --backend tmux \
    --container_image veristressgt-gpu \
    --container_mounts "$REPO" \
    --container_docker_args "--gpus device=0" \
    --container_forward_env VST_CONDA_ROOT,VST_CONDA_ROOTS,VST_PYTHON,ABCROWN_VNNCOMP2024_DIR,ABCROWN_CONDA_ENV \
    --params "matrix: {mini_sweep.spec_path: '$REPO/src/VeriStressGT/configs/mini_sweep.yaml', mini_sweep.abcrown_config: '$REPO/src/VeriStressGT/configs/abcrown_basic.yaml', mini_sweep.bench_dir: '$RUN/mini_sweep_bench', mini_sweep.run_dir: '$RUN/mini_sweep_run', mini_sweep.timeout: 60}"
```

Budget up to 50 minutes per verifier at the 60 s timeout. The June dry run,
merged as `PhaseI_DryRun/UCLA`, gave VERIFIED with correct-UNSAT fractions of
abcrown 0.98 (49/50, one timeout), pyrat 0.90 (45/50) and nnenum 0.26 (13/50).
Our reproduction matches abcrown and pyrat exactly. The nnenum errors are
structural (ONNX graphs it cannot parse) and are part of the expected result,
not a verifier failure.

The verdict is written to `$RUN/<hash>_<timestamp>/verdict.json`, with a
`latest` symlink beside it. The node's own `results.json`, with per-verifier
counts, sits under `$RUN/_kwdagger/mini_sweep/`.

## Leasing

Does not apply. This card does no live inference.

## What Kitware changes when evaluating

Our runner supplies the host-specific values: which GPU index, the conda root
that fills the pyrat/nnenum gap, tmux or Slurm as the backend, and a provenance
record written next to the verdict. The card, this image and the command shape
are exactly what is shown above.
