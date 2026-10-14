#!/usr/bin/env bash
#SBATCH --job-name=srp-pair-ablation
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
  sbatch scripts/run_pair_ablation.slurm INITIALIZATION K STRATEGY MODE SUBSET_SEED TRAIN_SEED

STRATEGY:
  random | same_class | most_similar | middle_similarity | less_similar | different_label
MODE:
  smoke | screen

All conditions use CutMix alpha=1 and probability=0.5 so the comparison
isolates partner selection as closely as possible.
EOF
}

if [[ $# -ne 6 ]]; then usage >&2; exit 2; fi
initialization="$1"
k="$2"
strategy="$3"
run_mode="$4"
subset_seed="$5"
train_seed="$6"

case "$initialization" in scratch|pretrained) ;; *) echo "Invalid initialization" >&2; exit 2 ;; esac
case "$k" in 20|50|100|200|300|450) ;; *) echo "Invalid k" >&2; exit 2 ;; esac
case "$strategy" in
    random|same_class|most_similar|middle_similarity|less_similar|different_label) ;;
    *) echo "Invalid strategy: $strategy" >&2; exit 2 ;;
esac
case "$run_mode" in smoke|screen) ;; *) echo "Invalid mode: $run_mode" >&2; exit 2 ;; esac
case "$subset_seed" in 0|1|2) ;; *) echo "Invalid subset seed" >&2; exit 2 ;; esac
[[ "$train_seed" =~ ^[0-9]+$ ]] || { echo "Invalid training seed" >&2; exit 2; }
if [[ "$k" == "450" && "$subset_seed" != "0" ]]; then
    echo "k=450 only has subset seed 0." >&2
    exit 2
fi

repo_root="${SRP_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_root" || ! -f "$repo_root/src/train.py" ]]; then
    echo "Submit from the repository root, or set SRP_REPO_ROOT." >&2
    exit 2
fi
cd "$repo_root"

split_path="data/splits/cifar100/k${k}_seed${subset_seed}.json"
[[ -f "$split_path" ]] || { echo "Missing split: $split_path" >&2; exit 2; }

conda_sh="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda_env="${CONDA_ENV:-srp-augmentation}"
[[ -f "$conda_sh" ]] || { echo "Conda initialization script not found: $conda_sh" >&2; exit 2; }
# shellcheck disable=SC1090
source "$conda_sh"
conda activate "$conda_env"

epochs=100
lr=0.01
gamma=0.1
milestones=(30 55 75)
if [[ "$k" == "450" ]]; then
    epochs=50
    lr=0.1
    gamma=0.2
    milestones=(15 30 40)
fi
output_root="results/third_presentation/ablation/$initialization"
if [[ "$run_mode" == "smoke" ]]; then
    epochs=1
    output_root="results/smoke/pair_ablation/$initialization"
fi

init_arg=--pretrained
[[ "$initialization" == "scratch" ]] && init_arg=--no-pretrained
augmentation=cutmix
guided_args=()

if [[ "$strategy" != "random" ]]; then
    augmentation=simcutmix
    neighbor_mode=class_agnostic
    neighbor_k=10
    rank_start=1
    case "$strategy" in
        same_class) neighbor_mode=class_aware ;;
        most_similar) ;;
        middle_similarity) rank_start=11 ;;
        less_similar) neighbor_k=20; rank_start=21 ;;
        different_label) neighbor_mode=different_label ;;
    esac

    support_dir="results/experiments/shared/neighbors/cifar100/k${k}_seed${subset_seed}"
    shopt -s nullglob
    matches=("$support_dir"/"neighbors_${neighbor_mode}_K"*.pt)
    shopt -u nullglob
    if (( ${#matches[@]} != 1 )); then
        echo "Expected exactly one ${neighbor_mode} neighbor payload in $support_dir; found ${#matches[@]}." >&2
        exit 2
    fi
    guided_args=(
        --neighbor-path "${matches[0]}"
        --guided-mode "$neighbor_mode"
        --neighbor-k "$neighbor_k"
        --neighbor-rank-start "$rank_start"
        --pair-sampling uniform
        --mix-prob 0.5
        --mix-warmup-epochs 0
    )
fi

train_args=(
    -u src/train.py --dataset cifar100 --model resnet50 "$init_arg"
    --k "$k" --subset-seed "$subset_seed" --train-seed "$train_seed"
    --augmentation "$augmentation" --epochs "$epochs" --batch-size 32
    --optimizer sgd --lr "$lr" --momentum 0.9 --nesterov
    --weight-decay 0.0005 --lr-milestones "${milestones[@]}" --lr-gamma "$gamma"
    --cutmix-alpha 1 --cutmix-prob 0.5 --mixup-alpha 1
    --num-workers "${SLURM_CPUS_PER_TASK:-2}" --output-root "$output_root"
    --skip-test "${guided_args[@]}"
)

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Strategy: $strategy"
printf 'Command:'; printf ' %q' python "${train_args[@]}"; printf '\n'
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
srun python "${train_args[@]}"
echo "Finished: $(date --iso-8601=seconds)"
