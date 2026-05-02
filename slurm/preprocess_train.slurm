#!/bin/bash
#SBATCH --job-name=epintlm_train
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Environment overridable via --export=ALL,CONFIG=...,RUN_NAME=... or pre-set in shell:
CONFIG="${CONFIG:-configs/default.yaml}"
RUN_NAME="${RUN_NAME:-}"

set -euo pipefail
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Bioinfomatrics

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

echo "Config: $CONFIG"
echo "Run name: ${RUN_NAME:-<auto>}"

if [[ -n "$RUN_NAME" ]]; then
  CONFIG="$CONFIG" RUN_NAME="$RUN_NAME" bash scripts/run_preprocess_train.sh
else
  CONFIG="$CONFIG" bash scripts/run_preprocess_train.sh
fi
