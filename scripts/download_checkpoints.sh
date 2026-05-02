#!/usr/bin/env bash
# Download EPINTLM pretrained checkpoints from the authors' Google Drive.
# https://drive.google.com/drive/folders/18DHZgsJqupNTnWmPrRiA3F1SrMro2q_H
#
# Requires: gdown (`pip install gdown`)
# Env:
#   DEST   Default: ./checkpoints
#
# Idempotent: skips files that already exist.

set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${DEST:-./checkpoints}"
mkdir -p "$DEST"

# shellcheck source=scripts/_activate_env.sh
source scripts/_activate_env.sh

if ! command -v gdown >/dev/null 2>&1; then
  echo "Installing gdown via pip..."
  pip install --quiet gdown
fi

FOLDER_ID="18DHZgsJqupNTnWmPrRiA3F1SrMro2q_H"

echo "==> Downloading checkpoints folder $FOLDER_ID into $DEST"
# --folder downloads everything in the folder, --remaining-ok keeps going on per-file failures
gdown --folder --id "$FOLDER_ID" -O "$DEST" --remaining-ok || true

echo
echo "==> Files in $DEST:"
ls -lh "$DEST"
