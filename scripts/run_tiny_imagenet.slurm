#!/usr/bin/env bash
#SBATCH --job-name=srp-tiny
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
  sbatch scripts/run_tiny_imagenet.slurm INITIALIZATION K METHOD MODE SUBSET_SEED TRAIN_SEED

Arguments:
  INITIALIZATION  scratch | pretrained
  K               20 | 100 | 450
  METHOD          none | cutmix | simcutmix
  MODE            smoke | screen | final | evaluate
  SUBSET_SEED     0 | 1 | 2 (k=450 requires 0)
  TRAIN_SEED      non-negative integer

The official Tiny ImageNet training partition supplies the fixed validation
and k-shot pools. The labeled official validation partition is used only as
the final test set; the unlabeled official test partition is never loaded.
EOF
}

if [[ $# -ne 6 ]]; then usage >&2; exit 2; fi
initialization="$1"
k="$2"
method="$3"
mode="$4"
subset_seed="$5"
train_seed="$6"

case "$initialization" in scratch|pretrained) ;; *) echo "Invalid initialization" >&2; exit 2 ;; esac
case "$k" in 20|100|450) ;; *) echo "Invalid k: $k" >&2; exit 2 ;; esac
case "$method" in none|cutmix|simcutmix) ;; *) echo "Invalid method: $method" >&2; exit 2 ;; esac
case "$mode" in smoke|screen|final|evaluate) ;; *) echo "Invalid mode: $mode" >&2; exit 2 ;; esac
case "$subset_seed" in 0|1|2) ;; *) echo "Invalid subset seed" >&2; exit 2 ;; esac
[[ "$train_seed" =~ ^[0-9]+$ ]] || { echo "Invalid training seed" >&2; exit 2; }
if [[ "$k" == "450" && "$subset_seed" != "0" ]]; then
    echo "k=450 uses the complete post-validation pool and requires subset seed 0." >&2
    exit 2
fi

repo_root="${SRP_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_root" || ! -f "$repo_root/src/train.py" ]]; then
    echo "Submit from the repository root, or set SRP_REPO_ROOT." >&2
    exit 2
fi
cd "$repo_root"
mkdir -p logs

split_path="data/splits/tinyimagenet/k${k}_seed${subset_seed}.json"
[[ -f "$split_path" ]] || { echo "Missing split: $split_path" >&2; exit 2; }
[[ -f "data/raw/tiny-imagenet-200/wnids.txt" ]] || {
    echo "Missing cached Tiny ImageNet dataset below data/raw/tiny-imagenet-200." >&2
    exit 2
}

conda_sh="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda_env="${CONDA_ENV:-srp-augmentation}"
[[ -f "$conda_sh" ]] || { echo "Conda initialization script not found: $conda_sh" >&2; exit 2; }
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

output_root="results/third_presentation/tiny_imagenet/final/$initialization"
evaluation_args=()
if [[ "$mode" == "smoke" ]]; then
    epochs=1
    output_root="results/smoke/tiny_imagenet/$initialization"
    evaluation_args+=(--skip-test)
elif [[ "$mode" == "screen" ]]; then
    output_root="results/third_presentation/tiny_imagenet/screening/$initialization"
    evaluation_args+=(--skip-test)
elif [[ "$mode" == "evaluate" ]]; then
    output_root="results/third_presentation/tiny_imagenet/screening/$initialization"
    evaluation_args+=(--evaluate-only)
fi

initialization_arg=--pretrained
[[ "$initialization" == "scratch" ]] && initialization_arg=--no-pretrained

train_args=(
    -u src/train.py --dataset tinyimagenet --model resnet50 "$initialization_arg"
    --k "$k" --subset-seed "$subset_seed" --train-seed "$train_seed"
    --augmentation "$method" --epochs "$epochs" --batch-size 32
    --optimizer sgd --lr "$learning_rate" --momentum 0.9 --nesterov
    --weight-decay 0.0005 --lr-milestones "${milestones[@]}" --lr-gamma "$lr_gamma"
    --mixup-alpha 1 --cutmix-alpha 1 --cutmix-prob 0.5
    --mix-prob 1 --mix-warmup-epochs 0
    --num-workers "${SLURM_CPUS_PER_TASK:-2}" --output-root "$output_root"
    "${evaluation_args[@]}"
)

if [[ "$method" == "simcutmix" ]]; then
    neighbors="results/experiments/shared/neighbors/tinyimagenet/k${k}_seed${subset_seed}/neighbors_class_agnostic_K40.pt"
    [[ -f "$neighbors" ]] || { echo "Missing guided-neighbor payload: $neighbors" >&2; exit 2; }
    train_args+=(
        --neighbor-path "$neighbors" --guided-mode class_agnostic
        --neighbor-k 20 --neighbor-rank-start 21 --pair-sampling uniform
    )
fi

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Dataset policy: official train -> train/validation; official labeled val -> test"
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
printf 'Command:'; printf ' %q' python "${train_args[@]}"; printf '\n'
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
srun python "${train_args[@]}"
echo "Finished: $(date --iso-8601=seconds)"
