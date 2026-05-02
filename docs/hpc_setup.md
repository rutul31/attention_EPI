# HPC setup — TAMU Grace

Grace separates work into two environments with different network access:

| Where | Internet egress? | Use it for |
|---|---|---|
| **Login node** (`grace.hprc.tamu.edu`) | Yes | Small downloads, dataset staging, light prep work |
| **Compute node** (allocated by SLURM) | **No** | Training, preprocessing, FASTA extraction (CPU-bound or GPU-bound work on already-staged data) |

This means **all `download_*.sh` scripts must run on the login node**, not via `sbatch`.
A SLURM job that calls them will fail at the connectivity check. The `build_dataset.slurm`
job is structured to assume the pair tables already exist and only does the conversion +
FASTA extraction (no internet needed once `hg19.fa` is on disk).

## End-to-end recipe

```bash
ssh rutul31@grace.hprc.tamu.edu
cd /scratch/user/rutul31/EPINTLM
module load Anaconda3
source activate Bioinfomatrics
```

### 1. Stage data on the login node

```bash
# TargetFinder pair tables (~50 MB total, takes seconds)
bash scripts/download_targetfinder.sh

# EPINTLM pretrained checkpoints (~880 MB, ~1 min)
bash scripts/download_checkpoints.sh

# 6-mer NT embeddings (~20 MB; first run downloads ~2 GB of model weights)
bash scripts/download_embeddings.sh

# hg19 reference (~3 GB) — only needed if you'll do FASTA extraction later
mkdir -p /scratch/user/rutul31/refs
cd /scratch/user/rutul31/refs
wget -c https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz
module load SAMtools
samtools faidx hg19.fa
cd /scratch/user/rutul31/EPINTLM

# ENCODE BigWigs (~30 GB) — long. Run inside tmux so you can detach.
tmux new -s encode
bash scripts/download_encode.sh
# Ctrl-B then D to detach. tmux a -t encode to reattach.
```

If `download_encode.sh` is too slow on the login node (HPRC has soft caps), do a dry-run
first to verify accession resolution works, then run the actual download in chunks via
the `CELL_LINES` env var:

```bash
DRY_RUN=1 bash scripts/download_encode.sh                  # resolve only, ~10s
CELL_LINES=HeLa-S3 bash scripts/download_encode.sh         # one cell at a time
```

### 2. Submit compute jobs

```bash
# Convert pair tables → BED → FASTA (no internet needed; uses hg19.fa)
sbatch --export=ALL,HG19_FA=/scratch/user/rutul31/refs/hg19.fa \
       slurm/build_dataset.slurm

# Preprocess + train one cell line
sbatch --export=ALL,CONFIG=configs/cell_lines/HeLa-S3.yaml \
       slurm/preprocess_train.slurm

# Evaluate the trained run (or pretrained checkpoints)
sbatch --export=ALL,RUN_DIR=runs/eval_pretrained,CHECKPOINTS_DIR=checkpoints \
       slurm/eval.slurm
```

## Troubleshooting

**`cannot reach raw.githubusercontent.com`** — you submitted a download script via SLURM.
Compute nodes are firewalled; run the script on the login node directly.

**`Disk quota exceeded`** — `/scratch/user/$USER/` has a quota; the ENCODE BigWigs are
~30 GB. Check with `showquota` (HPRC-specific). Move large files into a project allocation
if available.

**`module: command not found` inside SLURM** — you're using a partition that doesn't
auto-source `/etc/profile.d/modules.sh`. Add `source /etc/profile.d/modules.sh` near the
top of the SLURM script, before `module load`.

**Slow login-node downloads** — HPRC throttles long-running login-node sessions. Use
`tmux` for ENCODE (~hours) or split by cell line via `CELL_LINES=` env var.
