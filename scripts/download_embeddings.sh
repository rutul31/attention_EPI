#!/usr/bin/env bash
# Generate the Nucleotide-Transformer 6-mer embedding matrix.
#
# Output: $DEST/nucleotide_transformer_6_kmer_embedding.npy (4097 x 1280, ~20 MB)
#
# Requires: transformers, torch (Bioinfomatrics conda env or equivalent).
# First run downloads ~2 GB of model weights from HuggingFace.
#
# Env:
#   DEST   Default: ./data/embeddings

set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${DEST:-./data/embeddings}"
mkdir -p "$DEST"
OUT="$DEST/nucleotide_transformer_6_kmer_embedding.npy"

# shellcheck source=scripts/_activate_env.sh
source scripts/_activate_env.sh

if [[ -s "$OUT" ]]; then
  echo "==> Embeddings already present: $OUT"
  exit 0
fi

echo "==> Generating $OUT (this downloads ~2 GB from HuggingFace on first run)"
python scripts/generate_embeddings.py --output "$OUT"
echo "==> Done."
