#!/usr/bin/env bash
# Download TargetFinder enhancer-promoter pair tables (Whalen et al. 2016, Nat Genet).
# Repo: https://github.com/shwhalen/targetfinder  (paper/targetfinder/{cell}/output-ep/pairs.csv)
#
# These ~6–10 MB CSVs are the source of truth for enhancer/promoter coordinates and labels.
# Converting them to FASTA requires hg19 + bedtools — see scripts/build_dataset.sh.
#
# Env:
#   DATA_ROOT    Default: ./data
#   CELL_LINES   Default: "GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK"
#
# Diagnostics:
#   - Some HPC compute nodes have restricted egress. If this fails on a compute node,
#     run on the login node instead, then re-submit the SLURM job for FASTA extraction.
#   - HTTPS to raw.githubusercontent.com must be reachable. Check with:
#         curl -sI https://raw.githubusercontent.com/shwhalen/targetfinder/master/README.md

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-./data}"
CELL_LINES="${CELL_LINES:-GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK}"

TF_BASE="https://raw.githubusercontent.com/shwhalen/targetfinder/master/paper/targetfinder"
DEST="$DATA_ROOT/raw/targetfinder/pairs"
mkdir -p "$DEST"

# Pick a downloader. wget on Grace exists; curl is universal.
if command -v wget >/dev/null 2>&1; then
  fetch() { wget -c -q --show-progress -O "$2" "$1"; }
elif command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL -o "$2" "$1"; }
else
  echo "ERROR: neither wget nor curl available" >&2; exit 1
fi

# Connectivity check — fail fast with an informative message
echo "==> Connectivity check..."
if ! curl -fsI -m 10 "https://raw.githubusercontent.com/shwhalen/targetfinder/master/README.md" >/dev/null 2>&1; then
  echo "ERROR: cannot reach raw.githubusercontent.com (HTTPS). " \
       "If on an HPC compute node with no egress, run this on the login node instead." >&2
  exit 2
fi
echo "    OK"

echo "==> Downloading TargetFinder pair tables to $DEST"
status=0
for cell in $CELL_LINES; do
  url="$TF_BASE/$cell/output-ep/pairs.csv"
  out="$DEST/${cell}_pairs.csv"
  if [[ -s "$out" ]]; then
    echo "    [skip] $cell (present, $(wc -l <"$out") rows)"
    continue
  fi
  echo "    [run]  $cell ← $url"
  if fetch "$url" "$out"; then
    rows=$(wc -l <"$out" || echo 0)
    if [[ "$rows" -lt 100 ]]; then
      echo "    !! suspicious row count ($rows); leaving file in place but flagging." >&2
      status=1
    fi
  else
    echo "    ✗ failed for $cell" >&2
    rm -f "$out"
    status=1
  fi
done

cat <<'EOF'

==> Pair tables downloaded.

Next: build the full BED + FASTA dataset (HPC, requires hg19.fa + bedtools):
  HG19_FA=/path/to/hg19.fa bash scripts/build_dataset.sh

Or end-to-end via SLURM:
  sbatch --export=ALL,HG19_FA=/scratch/$USER/hg19.fa slurm/build_dataset.slurm

For inference only on the bundled HeLa-S3 test split, no further action needed.
EOF

exit "$status"
