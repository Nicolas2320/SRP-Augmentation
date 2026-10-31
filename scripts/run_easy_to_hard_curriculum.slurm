#!/usr/bin/env bash
#SBATCH --job-name=srp-curriculum
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
  sbatch scripts/run_easy_to_hard_curriculum.slurm INITIALIZATION K MODE SUBSET_SEED TRAIN_SEED

Arguments:
  INITIALIZATION  scratch | pretrained
  K               20 | 50 | 100 | 200 | 300 | 450
  MODE            smoke | screen | evaluate
  SUBSET_SEED     0 | 1 | 2 (k=450 requires 0)
  TRAIN_SEED      non-negative integer

The stepwise schedule uses ranks 21-40 in the first third, 11-30 in the
second third, and 1-10 in the final third. smoke and screen are
validation-only. evaluate tests an existing best-validation checkpoint
without retraining.
EOF
}

if [[ $# -ne 5 ]]; then usage >&2; exit 2; fi
initialization="$1"
k="$2"
run_mode="$3"
subset_seed="$4"
train_seed="$5"

case "$initialization" in scratch|pretrained) ;; *) echo "Invalid initialization" >&2; exit 2 ;; esac
case "$k" in 20|50|100|200|300|450) ;; *) echo "Invalid k" >&2; exit 2 ;; esac
case "$run_mode" in smoke|screen|evaluate) ;; *) echo "Invalid mode: $run_mode" >&2; exit 2 ;; esac
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
neighbors="results/experiments/shared/neighbors/cifar100/k${k}_seed${subset_seed}/neighbors_class_agnostic_K40.pt"
[[ -f "$split_path" ]] || { echo "Missing split: $split_path" >&2; exit 2; }
[[ -f "$neighbors" ]] || { echo "Missing neighbor payload: $neighbors" >&2; exit 2; }

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
output_root="results/third_presentation/curriculum/$initialization"
evaluation_args=(--skip-test)
if [[ "$run_mode" == "smoke" ]]; then
    epochs=3
    output_root="results/smoke/curriculum/$initialization"
elif [[ "$run_mode" == "evaluate" ]]; then
    evaluation_args=(--evaluate-only)
fi

# These are the first epochs of phases two and three. Integer division keeps
# the three phases as balanced as possible and makes the schedule deterministic.
transition_first=$((epochs / 3 + 1))
transition_second=$((2 * epochs / 3 + 1))

init_arg=--pretrained
[[ "$initialization" == "scratch" ]] && init_arg=--no-pretrained

train_args=(
    -u src/train.py --dataset cifar100 --model resnet50 "$init_arg"
    --k "$k" --subset-seed "$subset_seed" --train-seed "$train_seed"
    --augmentation simcutmix --epochs "$epochs" --batch-size 32
    --optimizer sgd --lr "$lr" --momentum 0.9 --nesterov
    --weight-decay 0.0005 --lr-milestones "${milestones[@]}" --lr-gamma "$gamma"
    --cutmix-alpha 1 --cutmix-prob 0.5 --mixup-alpha 1
    --neighbor-path "$neighbors" --guided-mode class_agnostic
    --neighbor-k 20 --neighbor-rank-start 21 --pair-sampling uniform
    --mix-prob 0.5 --mix-warmup-epochs 0
    --pair-curriculum easy_to_hard
    --curriculum-transition-epochs "$transition_first" "$transition_second"
    --num-workers "${SLURM_CPUS_PER_TASK:-2}" --output-root "$output_root"
    "${evaluation_args[@]}"
)

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Schedule: ranks 21-40 until epoch $((transition_first - 1)), ranks 11-30 until epoch $((transition_second - 1)), ranks 1-10 afterward"
printf 'Command:'; printf ' %q' python "${train_args[@]}"; printf '\n'
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
srun python "${train_args[@]}"
echo "Finished: $(date --iso-8601=seconds)"
