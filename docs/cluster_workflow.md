# ISMLL Cluster Workflow

This guide describes how to keep GitHub as the source of truth while using the
ISMLL cluster for GPU execution. The login node is for Git operations,
environment management, downloads, and Slurm submission. Training must run on
a Slurm compute node.

## Working Model

Use the local checkout for normal code and documentation changes:

1. Pull `main` locally and create a focused branch.
2. Edit, review, commit, and push the branch to GitHub.
3. Pull the accepted commit in the cluster checkout.
4. Submit a Slurm job from that exact commit.
5. Keep datasets, caches, checkpoints, and transient logs on the cluster.
6. Commit only intentional small evidence such as final `metrics.csv`,
   `summary.json`, and report figures.

Avoid editing the same branch independently on the laptop and cluster. If an
emergency cluster-side edit is necessary, commit it on a separate branch and
push it before continuing locally.

## Connect and Resume

1. Activate the University of Hildesheim WireGuard tunnel.
2. Connect to `master.ismll.de` with VS Code Remote - SSH.
3. Open the cluster checkout and prepare the shell:

```bash
cd "$HOME/SRP-Augmentation"
conda activate srp-augmentation
git status --short --branch
git pull --ff-only
squeue -u "$USER"
```

Never place passwords, VPN configurations, SSH private keys, or GitHub tokens
in the repository.

## One-Time Cluster Preparation

The compute nodes do not have Internet access. Install dependencies and cache
external inputs from the login node before submitting training:

```bash
python -m pip install -r requirements.txt
python -c "from torchvision.datasets import CIFAR100; CIFAR100(root='data/raw', train=True, download=True); CIFAR100(root='data/raw', train=False, download=True)"
python -c "from torchvision.models import ResNet50_Weights, resnet50; resnet50(weights=ResNet50_Weights.DEFAULT)"
```

The committed files under `data/splits/` are authoritative. Do not regenerate
them during routine cluster setup.

## Run the Tests

Submit CPU-only tests from the repository root:

```bash
mkdir -p logs
sbatch \
  --partition=CPU \
  --time=00:15:00 \
  --cpus-per-task=2 \
  --mem=4G \
  --job-name=srp-tests \
  --chdir="$PWD" \
  --output=logs/srp-tests-%j.log \
  --wrap="$CONDA_PREFIX/bin/python -m unittest discover -s tests -v"
```

Record the final state with `sacct -j JOB_ID` and inspect the matching log.

## Submit ResNet50 Runs

The Slurm runner requires four explicit arguments so a full experiment cannot
be started accidentally:

```bash
# One epoch, validation only, output under results/smoke/
sbatch scripts/run_resnet_comparison.slurm pretrained 20 none smoke

# Fixed comparison recipe, including final test evaluation
sbatch scripts/run_resnet_comparison.slurm pretrained 20 cutmix full
```

Accepted values:

- Initialization: `scratch`, `pretrained`.
- Budget: `20`, `50`, `100`, `450`.
- Method: `raw`, `none`, `mixup`, `cutmix`, `augmix`, `simmixup`, `simcutmix`.
- Mode: `smoke`, `full`.

The runner defaults to Conda environment `srp-augmentation`. Override it only
when necessary at submission time:

```bash
sbatch --export=ALL,CONDA_ENV=another-env \
  scripts/run_resnet_comparison.slurm pretrained 20 none smoke
```

Guided methods require their generated neighbor payload before submission.
The job exits early when that payload is absent.

The runner also sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. PyTorch requires this
for deterministic cuBLAS matrix multiplications when deterministic algorithms
are enabled on CUDA 10.2 or newer.

## Monitor and Inspect

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
tail -f logs/srp-resnet-JOB_ID.out
cat logs/srp-resnet-JOB_ID.err
```

`PD (Resources)` means the job is waiting for a suitable node. Slurm jobs keep
running after VPN or VS Code disconnects.

For a completed smoke run, verify that its run directory below
`results/smoke/` contains `metrics.csv`, `summary.json`, and
`checkpoint_best.pt`. Checkpoints are intentionally ignored by Git.

## Artifact Policy

Keep on the cluster only:

- `data/raw/` and model-download caches;
- Conda environments;
- Slurm `.out`, `.err`, and diagnostic logs;
- checkpoints (`*.pt`, `*.pth`);
- incomplete or exploratory outputs.

Candidates for GitHub after review:

- source code, scripts, and configuration;
- final `metrics.csv` and `summary.json` records;
- compact generated tables and report figures;
- documentation required to reproduce the run.

Before committing cluster-generated evidence, inspect `git status` and add
files explicitly. Never use `git add .` from a directory containing new
experiment outputs.

## Disconnect

Check `git status` and `squeue -u "$USER"`, then use **Remote-SSH: Close Remote
Connection** in VS Code. Deactivate WireGuard afterward. Queued and running
Slurm jobs are unaffected.
