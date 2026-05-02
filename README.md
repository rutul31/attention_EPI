# EPINTLM

A reproduction-grade implementation of **EPINTLM** — Enhancer–Promoter Prediction with
Pretrained k-mer Embeddings and Residual Cross-Attention (Nguyen et al., *Briefings in
Bioinformatics*, 2026).

This repository wraps the published model in a config-driven, two-stage pipeline:
**(A)** preprocess + train and **(B)** evaluate + plot. Everything is reproducible from a
single YAML config; runs are isolated under `runs/{timestamp}_{name}/` with manifests,
logs, checkpoints, metrics, and plots.

---

## Layout

```
epintlm/             # Main Python package (importable)
  config.py            YAML loader + dataclass schema
  logging_utils.py     stdlib logging w/ run-id formatter
  manifest.py          Per-run manifest (git, env, config snapshot)
  data/                Tokenizer, FASTA loader, dataset, preprocess
  models/              EPIModel + swappable encoder/attention/head blocks
  training/            Trainer, EarlyStopping, MetricsTracker, AttentionMonitor
  eval/                Evaluator + plotting
  cli/                 Entry points: preprocess, train, eval

configs/             # YAML configs (default + per-cell-line + ablations)
scripts/             # Shell wrappers + data acquisition scripts
slurm/               # SLURM job templates for HPC
data/                # Raw + processed data (gitignored)
checkpoints/         # Pretrained / trained checkpoints
runs/                # Per-run outputs (gitignored)
tests/               # pytest unit tests
docs/                # Architecture notes + paper PDF
DATA.md              # Full data inventory + acquisition guide
```

---

## Quick start

### Install

```bash
conda create -n Bioinfomatrics python=3.11 -y
conda activate Bioinfomatrics
pip install -r requirements.txt
```

### Inference smoke test (uses bundled HeLa-S3 test data + pretrained checkpoints)

```bash
# 1. Preprocess (one-time, cached). Re-runs are no-ops.
python -m epintlm.cli.preprocess --config configs/cell_lines/HeLa-S3.yaml

# 2. Evaluate the 4 pretrained checkpoints against HeLa-S3 test data.
RUN_DIR=runs/eval_pretrained CHECKPOINTS_DIR=checkpoints \
  bash scripts/run_eval.sh
```

Outputs land in `runs/eval_pretrained/eval/`:
- `metrics.json` — per-cell-line AUC/AUPR/F1/accuracy
- `predictions/{cell}_y_{true,score}.npy` — raw scores for re-plotting
- `plots/{A_roc,B_pr,CD_compare,E_ablation,F_repro}.png`

### Full training (HeLa-S3, 40 epochs)

```bash
CONFIG=configs/cell_lines/HeLa-S3.yaml \
  bash scripts/run_preprocess_train.sh
```

A new directory `runs/{YYYY-MM-DD_HH-MM-SS}_HeLa-S3/` is created with:
- `config.yaml` — effective config snapshot
- `manifest.json` — git commit, hostname, GPU info, timestamp
- `train.log` — full log
- `checkpoints/{best,last,epoch_N}.pt`
- `metrics/training_metrics.json`
- `plots/attention_evolution.png`, `*_heatmaps.png`

### Override hyperparameters (CLI)

Any config field can be overridden via dot-notation positional args:

```bash
python -m epintlm.cli.train --config configs/cell_lines/HeLa-S3.yaml \
  training.batch_size=128 \
  training.num_epochs=20 \
  run.seed=42 \
  run.name=hela_quick
```

### Run an ablation

```bash
CONFIG=configs/ablations/no_residual.yaml RUN_NAME=ablation_no_resid \
  bash scripts/run_preprocess_train.sh
```

---

## Running on TAMU Grace HPC

```bash
# From the Grace login node, in /scratch/user/$USER/EPINTLM:
sbatch slurm/preprocess_train.slurm                                       # default config
sbatch --export=ALL,CONFIG=configs/cell_lines/GM12878.yaml \
       slurm/preprocess_train.slurm

# After training completes:
sbatch --export=ALL,RUN_DIR=runs/2026-04-25_HeLa-S3 slurm/eval.slurm

# Or evaluate all 4 pretrained checkpoints in one shot:
sbatch --export=ALL,RUN_DIR=runs/eval_pretrained,CHECKPOINTS_DIR=checkpoints \
       slurm/eval.slurm
```

See `docs/hpc_setup.md` for environment setup details.

---

## Data acquisition

See [DATA.md](DATA.md) for the full inventory. Short version:

| Data | Where | How |
|------|-------|-----|
| Pretrained checkpoints (~880 MB) | Google Drive | `bash scripts/download_checkpoints.sh` |
| 6-mer NT embeddings (~20 MB) | Generated locally | `bash scripts/download_embeddings.sh` |
| TargetFinder pair tables | GitHub | `bash scripts/download_targetfinder.sh` |
| hg19 reference (HPC, ~3 GB) | UCSC | manual / module-load |
| ENCODE chromatin BigWigs (HPC, ~30 GB) | ENCODE portal | `bash scripts/download_encode.sh` |

---

## Configuration

`configs/default.yaml` is the master schema. Cell-line and ablation configs are sparse
overrides on top of it (see `configs/cell_lines/HeLa-S3.yaml` for an example).

Top-level config sections:

```yaml
run:        # run name, seed, device, output_dir
data:       # cell_line, paths, splits, preprocessing (k-mer size, sequence lengths)
model:      # embedding, encoders, gru, attention (with ablation toggles), head
training:   # batch size, epochs, optimizer, scheduler, early stopping
eval:       # batch size, cell_lines list, plot toggle
logging:    # level, format, console + file flags
```

Ablation toggles in `model.attention`:
- `use_self_attn: bool`
- `use_cross_attn: bool`  (Table 3 ablation: drop this)
- `use_residual: bool`    (Table 3 ablation: drop this)

---

## Reproducing paper results

| Goal | Command |
|------|---------|
| Reproduce Table 2 (HeLa-S3) | `RUN_DIR=runs/repro_hela CHECKPOINTS_DIR=checkpoints bash scripts/run_eval.sh` |
| Reproduce Table 3/4 (ablation) | Train with `configs/ablations/no_residual.yaml` and `configs/ablations/no_cross_attn.yaml`, compare best AUC/AUPR |
| Reproduce Supp Fig. S1 (ROC/PR) | The eval step generates `A_roc_curves.png` and `B_pr_curves.png` |

Expected HeLa-S3 numbers (from paper): AUC ≈ 0.970, AUPR ≈ 0.865.

---

## Citation

Nguyen, T. L., Kha, H. Q., Nguyen, P. K., Le, M. H. N., Le, D. T., & Quoc Khanh Le, N. (2026).
EPINTLM: enhancer–promoter prediction with pretrained k-mer embeddings and residual cross-attention.
*Briefings in Bioinformatics*, 27(1), bbag064.
