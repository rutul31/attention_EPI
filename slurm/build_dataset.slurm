#!/bin/bash
#SBATCH --job-name=epintlm_build
#SBATCH --output=runs/slurm-build-%j.out
#SBATCH --error=runs/slurm-build-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

# Convert pre-staged pairs.csv files → BED → FASTA via bedtools.
# This SLURM job does NO downloading (Grace compute nodes have no internet).
#
# Prerequisites — run BEFORE submitting this job:
#   On the login node:
#     bash scripts/download_targetfinder.sh   # produces data/raw/targetfinder/pairs/*.csv
#   And ensure hg19 reference is available:
#     wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz \
#          -O /scratch/user/$USER/hg19.fa.gz
#     gunzip /scratch/user/$USER/hg19.fa.gz
#     module load SAMtools && samtools faidx /scratch/user/$USER/hg19.fa
#
# Submit:
#   sbatch --export=ALL,HG19_FA=/scratch/user/$USER/hg19.fa slurm/build_dataset.slurm

: "${HG19_FA:?HG19_FA is required (path to indexed hg19.fa)}"

set -euo pipefail
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Bioinfomatrics

module load BEDTools 2>/dev/null || module load bedtools 2>/dev/null || \
  echo "WARN: 'BEDTools' module not loaded — assuming bedtools is on PATH"

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

PAIRS_DIR="${DATA_ROOT:-./data/raw/targetfinder}/pairs"
if ! ls "$PAIRS_DIR"/*_pairs.csv >/dev/null 2>&1; then
  echo "ERROR: no pair tables in $PAIRS_DIR." >&2
  echo "       Run scripts/download_targetfinder.sh on the login node first." >&2
  exit 1
fi

echo "=== Convert pairs.csv → BED + FASTA ==="
HG19_FA="$HG19_FA" \
  DATA_ROOT="${DATA_ROOT:-./data/raw/targetfinder}" \
  CELL_LINES="${CELL_LINES:-GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK}" \
  AUGMENT="${AUGMENT:-1}" \
  bash scripts/build_dataset.sh

echo "=== Done. Submit slurm/preprocess_train.slurm next. ==="
