#!/usr/bin/env bash
# Stage A — preprocess (cached) + train.
#
# Env:
#   CONFIG    Path to YAML config (default: configs/default.yaml)
#   RUN_NAME  Run name suffix (timestamp prepended automatically)
#
# Examples:
#   ./scripts/run_preprocess_train.sh
#   CONFIG=configs/cell_lines/HeLa-S3.yaml ./scripts/run_preprocess_train.sh
#   CONFIG=configs/ablations/no_residual.yaml RUN_NAME=hela_no_resid ./scripts/run_preprocess_train.sh
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
RUN_NAME="${RUN_NAME:-}"

cd "$(dirname "$0")/.."

# shellcheck source=scripts/_activate_env.sh
source scripts/_activate_env.sh

echo "=== Preprocess ==="
python -m epintlm.cli.preprocess --config "$CONFIG"

echo "=== Train ==="
if [[ -n "$RUN_NAME" ]]; then
  python -m epintlm.cli.train --config "$CONFIG" --skip-preprocess "run.name=$RUN_NAME"
else
  python -m epintlm.cli.train --config "$CONFIG" --skip-preprocess
fi
