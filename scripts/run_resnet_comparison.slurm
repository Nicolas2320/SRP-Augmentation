#!/usr/bin/env bash
#SBATCH --job-name=srp-resnet
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch scripts/run_resnet_comparison.slurm INITIALIZATION K METHOD MODE

Arguments:
  INITIALIZATION  scratch | pretrained
  K               20 | 50 | 100 | 450
  METHOD          raw | none | mixup | cutmix | augmix | simmixup | simcutmix
  MODE            smoke | full

smoke runs one epoch, skips test evaluation, and writes below results/smoke/.
full runs the fixed comparison recipe and writes below results/comparison_v1/.
EOF
}

if [[ $# -ne 4 ]]; then
    usage >&2
    exit 2
fi

initialization="$1"
k="$2"
method="$3"
mode="$4"

case "$initialization" in
    scratch|pretrained) ;;
    *) echo "Invalid initialization: $initialization" >&2; usage >&2; exit 2 ;;
esac

case "$k" in
    20|50|100|450) ;;
    *) echo "Invalid k: $k" >&2; usage >&2; exit 2 ;;
esac

case "$method" in
    raw|none|mixup|cutmix|augmix|simmixup|simcutmix) ;;
    *) echo "Invalid method: $method" >&2; usage >&2; exit 2 ;;
esac

case "$mode" in
    smoke|full) ;;
    *) echo "Invalid mode: $mode" >&2; usage >&2; exit 2 ;;
esac

repo_root="${SRP_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_root" || ! -f "$repo_root/src/train.py" ]]; then
    echo "Submit this job from the repository root, or set SRP_REPO_ROOT." >&2
    exit 2
fi
cd "$repo_root"

conda_sh="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda_env="${CONDA_ENV:-srp-augmentation}"
if [[ ! -f "$conda_sh" ]]; then
    echo "Conda initialization script not found: $conda_sh" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$conda_sh"
conda activate "$conda_env"

epochs=100
learning_rate=0.01
lr_gamma=0.1
milestones=(30 55 75)
if [[ "$k" == "450" ]]; then
    epochs=50
    learning_rate=0.1
    lr_gamma=0.2
    milestones=(15 30 40)
fi

output_root="results/comparison_v1/$initialization"
evaluation_args=()
if [[ "$mode" == "smoke" ]]; then
    epochs=1
    output_root="results/smoke/$initialization"
    evaluation_args+=(--skip-test)
fi

initialization_arg=--pretrained
if [[ "$initialization" == "scratch" ]]; then
    initialization_arg=--no-pretrained
fi

train_args=(
    -u src/train.py
    --dataset cifar100
    --model resnet50
    "$initialization_arg"
    --k "$k"
    --subset-seed 0
    --train-seed 0
    --augmentation "$method"
    --epochs "$epochs"
    --batch-size 32
    --optimizer sgd
    --lr "$learning_rate"
    --momentum 0.9
    --nesterov
    --weight-decay 0.0005
    --lr-milestones "${milestones[@]}"
    --lr-gamma "$lr_gamma"
    --mixup-alpha 1
    --cutmix-alpha 1
    --cutmix-prob 0.5
    --mix-prob 1
    --mix-warmup-epochs 0
    --num-workers "${SLURM_CPUS_PER_TASK:-2}"
    --output-root "$output_root"
    "${evaluation_args[@]}"
)

if [[ "$method" == "simmixup" || "$method" == "simcutmix" ]]; then
    neighbors="results/experiments/shared/neighbors/cifar100/k${k}_seed0/neighbors_class_agnostic_K40.pt"
    if [[ ! -f "$neighbors" ]]; then
        echo "Missing guided-neighbor payload: $neighbors" >&2
        exit 2
    fi
    train_args+=(
        --neighbor-path "$neighbors"
        --guided-mode class_agnostic
        --neighbor-k 20
        --neighbor-rank-start 21
        --pair-sampling uniform
    )
fi

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Conda environment: $conda_env"
echo "Mode: $mode"
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
printf 'Command:'
printf ' %q' python "${train_args[@]}"
printf '\n'

export PYTHONUNBUFFERED=1
srun python "${train_args[@]}"

echo "Finished: $(date --iso-8601=seconds)"
