#!/usr/bin/env bash
#SBATCH --job-name=srp-vit
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
  sbatch scripts/run_vit_completion.slurm K METHOD MODE SUBSET_SEED TRAIN_SEED

Arguments:
  K            300 | 450
  METHOD       none | mixup | cutmix | augmix | simmixup | simcutmix
  MODE         smoke | screen | final | evaluate
  SUBSET_SEED  0 (the frozen single-seed completion uses subset seed 0)
  TRAIN_SEED   0 (the frozen single-seed completion uses training seed 0)

The completion recipe uses the historical scratch CIFAR ViT architecture and
100 epochs for every final method. AugMix retains its historical optimizer
schedule; all other methods use the shared mixing-method schedule.
EOF
}

if [[ $# -ne 5 ]]; then
    usage >&2
    exit 2
fi

k="$1"
method="$2"
mode="$3"
subset_seed="$4"
train_seed="$5"

case "$k" in
    300|450) ;;
    *) echo "Invalid k: $k" >&2; usage >&2; exit 2 ;;
esac

case "$method" in
    none|mixup|cutmix|augmix|simmixup|simcutmix) ;;
    *) echo "Invalid method: $method" >&2; usage >&2; exit 2 ;;
esac

case "$mode" in
    smoke|screen|final|evaluate) ;;
    *) echo "Invalid mode: $mode" >&2; usage >&2; exit 2 ;;
esac

if [[ "$subset_seed" != "0" || "$train_seed" != "0" ]]; then
    echo "The frozen ViT completion is single-seed only: subset_seed=0, train_seed=0." >&2
    exit 2
fi

repo_root="${SRP_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_root" || ! -f "$repo_root/src/train.py" ]]; then
    echo "Submit from the repository root, or set SRP_REPO_ROOT." >&2
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

split_path="data/splits/cifar100/k${k}_seed${subset_seed}.json"
if [[ ! -f "$split_path" ]]; then
    echo "Missing committed split: $split_path" >&2
    exit 2
fi

epochs=100
learning_rate=0.01
lr_gamma=0.1
milestones=(30 55 75)

# Preserve the historical AugMix training recipe so its k=300 point belongs
# to the same exploratory cohort as the existing k=20/50/100/450 records.
if [[ "$method" == "augmix" ]]; then
    learning_rate=0.1
    lr_gamma=0.2
    milestones=(30 60 80)
fi

output_root="results/third_presentation/vit/final"
evaluation_args=()
if [[ "$mode" == "smoke" ]]; then
    epochs=1
    output_root="results/smoke/vit"
    evaluation_args+=(--skip-test)
elif [[ "$mode" == "screen" ]]; then
    output_root="results/third_presentation/vit/screening"
    evaluation_args+=(--skip-test)
elif [[ "$mode" == "evaluate" ]]; then
    output_root="results/third_presentation/vit/screening"
    evaluation_args+=(--evaluate-only)
fi

train_args=(
    -u src/train.py
    --dataset cifar100
    --model vit
    --no-pretrained
    --k "$k"
    --subset-seed "$subset_seed"
    --train-seed "$train_seed"
    --augmentation "$method"
    --epochs "$epochs"
    --batch-size 64
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
    neighbors="results/experiments/shared/neighbors/cifar100/k${k}_seed${subset_seed}/neighbors_class_agnostic_K40.pt"
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
echo "Frozen ViT completion: k=$k method=$method mode=$mode seed=$train_seed"
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
printf 'Command:'
printf ' %q' python "${train_args[@]}"
printf '\n'

export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
srun python "${train_args[@]}"

echo "Finished: $(date --iso-8601=seconds)"
