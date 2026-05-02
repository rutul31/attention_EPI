#!/usr/bin/env bash
# Resolve and download ENCODE BigWig tracks for the 5 EPINTLM assays:
#   CTCF, DNase, H3K27ac, H3K4me1, H3K4me3.
#
# Hardcoding accessions doesn't work — ENCODE deprecates files periodically. This script
# delegates to epintlm.tools.encode_downloader, which queries ENCODE's REST API for the
# best released hg19 BigWig per (cell, assay) at runtime.
#
# Env:
#   DATA_ROOT       Default: ./data
#   CELL_LINES      Default: "GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK"
#   ENCODE_ASSAYS   Default: "CTCF DNase H3K27ac H3K4me1 H3K4me3"
#   DRY_RUN         Default: 0 (set to 1 to resolve+manifest only, skip downloads)
#
# Resumable: skips files that already exist.

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-./data}"
CELL_LINES="${CELL_LINES:-GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK}"
ENCODE_ASSAYS="${ENCODE_ASSAYS:-CTCF DNase H3K27ac H3K4me1 H3K4me3}"
DRY_RUN="${DRY_RUN:-0}"

DEST="$DATA_ROOT/raw/encode"
mkdir -p "$DEST"

# shellcheck source=scripts/_activate_env.sh
source scripts/_activate_env.sh

ARGS=(--dest "$DEST" --cells $CELL_LINES --assays $ENCODE_ASSAYS)
[[ "$DRY_RUN" == "1" ]] && ARGS+=(--dry-run)

echo "==> Resolving + downloading ENCODE BigWigs to $DEST"
python -m epintlm.tools.encode_downloader "${ARGS[@]}"

cat <<EOF

==> Done. Manifest: $DEST/manifest.json

Next: bin BigWigs into 500 bp tensors and write CTCF_DNase_6histone.500.json
+ per-cell .pt files. (epintlm/tools/bigwig_to_pt.py — pending.)

For inference / non-chromatin training, set
  data.chromatin_features.enabled: false
in your config (default). The pretrained checkpoints were trained this way.
EOF
