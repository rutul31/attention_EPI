#!/bin/bash
# DEPRECATED on Grace HPRC: compute nodes have no outbound internet.
# Downloads must run on the LOGIN NODE (or transfer node). See README → "Acquiring data".
#
# This script is left as a fallback for HPC sites where compute nodes can reach the internet.
# On Grace it will fail at the connectivity check.
#
#SBATCH --job-name=epintlm_download
#SBATCH --output=runs/slurm-download-%j.out
#SBATCH --error=runs/slurm-download-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

cat <<'EOF' >&2
WARNING: this SLURM job submits a download job to a compute node.
On TAMU Grace HPRC, compute nodes have no outbound internet — this will fail.
Run downloads interactively on the login node instead:

  ssh rutul31@grace.hprc.tamu.edu
  cd /scratch/user/rutul31/EPINTLM
  module load Anaconda3 && source activate Bioinfomatrics
  bash scripts/download_targetfinder.sh        # ~50 MB, seconds
  bash scripts/download_encode.sh              # ~30 GB, hours — use tmux

Then submit slurm/build_dataset.slurm (no download, FASTA extraction only).
EOF

bash scripts/download_targetfinder.sh
bash scripts/download_encode.sh
bash scripts/download_checkpoints.sh
