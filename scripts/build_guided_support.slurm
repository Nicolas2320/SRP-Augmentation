#!/usr/bin/env bash
#SBATCH --job-name=srp-support
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch scripts/build_guided_support.slurm DATASET K SUBSET_SEED MODE

Arguments:
  DATASET      cifar100 | tinyimagenet
  K            CIFAR-100: 20 | 50 | 100 | 200 | 300 | 450
               Tiny ImageNet: 20 | 100 | 450
  SUBSET_SEED  0 | 1 | 2 (k=450 requires 0)
  MODE         class_agnostic | class_aware | different_label

The job computes frozen ResNet50 ImageNet embeddings when absent, builds one
K40 neighbor payload, validates it, and records SHA-256 checksums. Existing
neighbor payloads are never overwritten.
EOF
}

if [[ $# -ne 4 ]]; then
    usage >&2
    exit 2
fi

dataset="$1"
k="$2"
subset_seed="$3"
neighbor_mode="$4"

case "$dataset" in
    cifar100)
        case "$k" in 20|50|100|200|300|450) ;; *) echo "Invalid CIFAR-100 k: $k" >&2; exit 2 ;; esac
        ;;
    tinyimagenet)
        case "$k" in 20|100|450) ;; *) echo "Invalid Tiny ImageNet k: $k" >&2; exit 2 ;; esac
        ;;
    *) echo "Invalid dataset: $dataset" >&2; exit 2 ;;
esac
case "$subset_seed" in 0|1|2) ;; *) echo "Invalid subset seed: $subset_seed" >&2; exit 2 ;; esac
case "$neighbor_mode" in
    class_agnostic|class_aware|different_label) ;;
    *) echo "Invalid neighbor mode: $neighbor_mode" >&2; exit 2 ;;
esac
if [[ "$k" == "450" && "$subset_seed" != "0" ]]; then
    echo "k=450 is the complete post-validation pool and only has subset seed 0." >&2
    exit 2
fi

repo_root="${SRP_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_root" || ! -f "$repo_root/src/proposal/compute_embeddings.py" ]]; then
    echo "Submit from the repository root, or set SRP_REPO_ROOT." >&2
    exit 2
fi
cd "$repo_root"

split_path="data/splits/${dataset}/k${k}_seed${subset_seed}.json"
[[ -f "$split_path" ]] || { echo "Missing committed split: $split_path" >&2; exit 2; }
if [[ "$dataset" == "tinyimagenet" && ! -f "data/raw/tiny-imagenet-200/wnids.txt" ]]; then
    echo "Missing cached Tiny ImageNet dataset below data/raw/tiny-imagenet-200." >&2
    exit 2
fi

conda_sh="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda_env="${CONDA_ENV:-srp-augmentation}"
[[ -f "$conda_sh" ]] || { echo "Conda initialization script not found: $conda_sh" >&2; exit 2; }
# shellcheck disable=SC1090
source "$conda_sh"
conda activate "$conda_env"

support_dir="results/experiments/shared/neighbors/${dataset}/k${k}_seed${subset_seed}"
embedding_path="$support_dir/embeddings_resnet50_imagenet.pt"
shopt -s nullglob
existing_neighbors=("$support_dir"/"neighbors_${neighbor_mode}_K"*.pt)
shopt -u nullglob
if (( ${#existing_neighbors[@]} > 0 )); then
    echo "Refusing to overwrite existing neighbor payload:" >&2
    printf '  %s\n' "${existing_neighbors[@]}" >&2
    exit 3
fi

mkdir -p logs
echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Slurm job: ${SLURM_JOB_ID:-not-slurm}"
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ ! -f "$embedding_path" ]]; then
    srun python -u src/proposal/compute_embeddings.py \
        --dataset "$dataset" \
        --k "$k" \
        --subset-seed "$subset_seed" \
        --encoder resnet50_imagenet \
        --batch-size 64 \
        --num-workers "${SLURM_CPUS_PER_TASK:-2}" \
        --device auto
else
    echo "Reusing existing embeddings: $embedding_path"
fi

srun python -u src/proposal/build_neighbors.py \
    --dataset "$dataset" \
    --k "$k" \
    --subset-seed "$subset_seed" \
    --encoder resnet50_imagenet \
    --mode "$neighbor_mode" \
    --max-neighbors 40 \
    --query-batch-size 512 \
    --device auto

srun python -u src/proposal/inspect_neighbors.py \
    --dataset "$dataset" \
    --k "$k" \
    --subset-seed "$subset_seed" \
    --encoder resnet50_imagenet \
    --mode "$neighbor_mode" \
    --max-neighbors 40

(
    cd "$support_dir"
    sha256sum embeddings_resnet50_imagenet.pt "neighbors_${neighbor_mode}_K"*.pt > "checksums_${neighbor_mode}.sha256"
)

echo "Finished: $(date --iso-8601=seconds)"
