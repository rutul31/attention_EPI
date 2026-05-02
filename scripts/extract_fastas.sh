#!/usr/bin/env bash
# Extract per-element FASTA sequences from hg19 using bedtools getfasta,
# given the EPINTLM-format BED files produced by epintlm.tools.targetfinder_to_bed.
#
# The BED format is custom (4 cols: label, chrom, start, end). bedtools expects
# chrom in column 1, so we strip the label column on the fly via process substitution.
#
# Env:
#   DATA_ROOT      Default: ./data/raw/targetfinder
#   HG19_FA        Required. Path to hg19.fa (uncompressed, indexed via `samtools faidx hg19.fa`).
#   CELL_LINES     Default: "GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK"
#
# Idempotent: skips already-extracted FASTAs.

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-./data/raw/targetfinder}"
CELL_LINES="${CELL_LINES:-GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK}"
: "${HG19_FA:?HG19_FA must point to an uncompressed, indexed hg19.fa}"

if [[ ! -f "$HG19_FA" ]]; then
  echo "ERROR: HG19_FA not found: $HG19_FA" >&2
  exit 1
fi
if [[ ! -f "${HG19_FA}.fai" ]]; then
  echo "ERROR: ${HG19_FA}.fai missing. Run: samtools faidx ${HG19_FA}" >&2
  exit 1
fi
command -v bedtools >/dev/null 2>&1 || { echo "ERROR: bedtools not in PATH"; exit 1; }

BED_DIR="$DATA_ROOT/bed_files"
SEQ_DIR="$DATA_ROOT/sequence_data"
mkdir -p "$SEQ_DIR"

extract_one() {
  local cell="$1" element="$2" split="$3"
  local in_bed="$BED_DIR/${cell}_${element}_${split}.bed"
  local out_fa="$SEQ_DIR/${cell}_${element}_${split}.fasta"

  if [[ ! -f "$in_bed" ]]; then
    echo "  [skip] $in_bed (does not exist)"
    return 0
  fi
  if [[ -s "$out_fa" ]]; then
    echo "  [skip] $out_fa (already present)"
    return 0
  fi

  echo "  [run]  $in_bed → $out_fa"
  # Strip label column (col 1) so bedtools sees a standard 3-col BED on stdin.
  awk -F'\t' 'BEGIN{OFS="\t"} {print $2, $3, $4}' "$in_bed" \
    | bedtools getfasta -fi "$HG19_FA" -bed - -fo "$out_fa"
}

for cell in $CELL_LINES; do
  for element in enhancer promoter; do
    for split in train test; do
      extract_one "$cell" "$element" "$split"
    done
  done
done

echo "✅ FASTA extraction complete."
