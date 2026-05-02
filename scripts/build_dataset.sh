#!/usr/bin/env bash
# Full TargetFinder → EPINTLM dataset pipeline:
#   1. Convert each cell-line pairs.csv → BED + label files
#   2. bedtools getfasta → enhancer/promoter FASTAs
#
# Prereqs:
#   - pairs.csv files already downloaded (run scripts/download_targetfinder.sh first)
#   - hg19.fa (uncompressed) + hg19.fa.fai index
#   - bedtools in PATH
#
# Env:
#   DATA_ROOT     Default: ./data/raw/targetfinder
#   HG19_FA       Required. Path to hg19.fa.
#   CELL_LINES    Default: "GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK"
#   AUGMENT       Default: 1 (oversample train positives 1:1 with negatives)
#   TEST_FRAC     Default: 0.10
#   SEED          Default: 2025
#
# Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-./data/raw/targetfinder}"
CELL_LINES="${CELL_LINES:-GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK}"
AUGMENT="${AUGMENT:-1}"
TEST_FRAC="${TEST_FRAC:-0.10}"
SEED="${SEED:-2025}"
: "${HG19_FA:?HG19_FA must point to an uncompressed, indexed hg19.fa}"

# shellcheck source=scripts/_activate_env.sh
source scripts/_activate_env.sh

PAIRS_DIR="$DATA_ROOT/pairs"

echo "==> Step 1: pairs.csv → BED + label files"
for cell in $CELL_LINES; do
  pairs="$PAIRS_DIR/${cell}_pairs.csv"
  if [[ ! -s "$pairs" ]]; then
    echo "  [skip] $cell (missing $pairs — run download_targetfinder.sh first)"
    continue
  fi
  echo "  [run]  $cell"
  ARGS=(--pairs "$pairs" --cell "$cell" --out-dir "$DATA_ROOT"
        --test-frac "$TEST_FRAC" --seed "$SEED")
  [[ "$AUGMENT" == "1" ]] && ARGS+=(--augment-train)
  python -m epintlm.tools.targetfinder_to_bed "${ARGS[@]}"
done

echo
echo "==> Step 2: BED → FASTA via bedtools getfasta"
DATA_ROOT="$DATA_ROOT" CELL_LINES="$CELL_LINES" HG19_FA="$HG19_FA" \
  bash scripts/extract_fastas.sh

echo
echo "✅ Dataset build complete. Run preprocess+train next:"
echo "    sbatch slurm/preprocess_train.slurm"
