#!/bin/bash
#SBATCH --job-name=epintlm_eval
#SBATCH --output=runs/slurm-eval-%j.out
#SBATCH --error=runs/slurm-eval-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Required: RUN_DIR. Optional: CONFIG, CHECKPOINT, CHECKPOINTS_DIR.
: "${RUN_DIR:?RUN_DIR must be set, e.g. via sbatch --export=ALL,RUN_DIR=runs/eval_all}"
CONFIG="${CONFIG:-configs/default.yaml}"

set -euo pipefail
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Bioinfomatrics

cd "$SLURM_SUBMIT_DIR"

CONFIG="$CONFIG" RUN_DIR="$RUN_DIR" \
  CHECKPOINT="${CHECKPOINT:-}" CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-}" \
  bash scripts/run_eval.sh
