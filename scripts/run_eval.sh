#!/usr/bin/env bash
# Stage B — evaluate a trained run (or pretrained checkpoints) and generate plots.
#
# Env:
#   CONFIG           Path to YAML config (default: configs/default.yaml)
#   RUN_DIR          Required. Output dir for eval artifacts.
#   CHECKPOINT       Optional. Single checkpoint path override.
#   CHECKPOINTS_DIR  Optional. Multi-checkpoint dir (one per cell line).
#
# Examples:
#   # Evaluate the best.pt produced by a training run
#   RUN_DIR=runs/2026-04-25_HeLa-S3 ./scripts/run_eval.sh
#
#   # Evaluate all 4 pretrained cell-line checkpoints together
#   RUN_DIR=runs/eval_all CHECKPOINTS_DIR=checkpoints ./scripts/run_eval.sh
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
: "${RUN_DIR:?RUN_DIR is required}"

cd "$(dirname "$0")/.."

# shellcheck source=scripts/_activate_env.sh
source scripts/_activate_env.sh

ARGS=("--config" "$CONFIG" "--run-dir" "$RUN_DIR")
[[ -n "${CHECKPOINT:-}" ]]       && ARGS+=("--checkpoint" "$CHECKPOINT")
[[ -n "${CHECKPOINTS_DIR:-}" ]]  && ARGS+=("--checkpoints-dir" "$CHECKPOINTS_DIR")

python -m epintlm.cli.eval "${ARGS[@]}"
