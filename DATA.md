# EPINTLM Data Inventory

This document lists every file required to fully reproduce EPINTLM training, evaluation,
and inference, along with where it comes from and how to acquire it.

The pretrained checkpoints + HeLa-S3 test data + k-mer embeddings are small enough that they
ship with the repo (or can be downloaded directly to a Mac). Everything larger is HPC-only.

## Quick status

| # | Item | Size | Status | Acquisition |
|---|------|------|--------|-------------|
| 1 | k-mer embeddings (`.npy`) | 20 MB | Local (or generate) | `scripts/download_embeddings.sh` |
| 2 | Pretrained checkpoints (4 cell lines) | ~880 MB | Local | `scripts/download_checkpoints.sh` |
| 3 | HeLa-S3 test FASTA + BED + labels | ~18 MB | Local | Bundled (subset of #4) |
| 4 | TargetFinder pair tables (6 cell lines) | ~10 MB | Download | `scripts/download_targetfinder.sh` |
| 5 | hg19 reference FASTA (for FASTA extraction) | ~3 GB | HPC only | Manual (e.g. UCSC) |
| 6 | ENCODE chromatin BigWigs (6 cells × 5 assays) | ~30 GB | HPC only | `scripts/download_encode.sh` |

---

## 1. Nucleotide-Transformer 6-mer Embedding Matrix

- **File:** `data/embeddings/nucleotide_transformer_6_kmer_embedding.npy`
- **Shape / dtype:** `(4097, 1280)` float32. 4096 k-mers + 1 zero-row null token.
- **Source:** Generated locally with `scripts/generate_embeddings.py`, which mean-pools
  the hidden states of `InstaDeepAI/nucleotide-transformer-500m-human-ref` (HuggingFace).
- **License:** Model is released under CC-BY-NC-SA 4.0 by InstaDeep.
- **Acquisition:** `bash scripts/download_embeddings.sh`
  - First run downloads ~2 GB of model weights to your HuggingFace cache.
  - Subsequent runs are no-ops if the `.npy` exists.

## 2. Pretrained EPINTLM Checkpoints

- **Files:** `checkpoints/L1_HELA.pt`, `L1_NU_GM12878.pt`, `L1_NU_HUVEC.pt`, `L1_NU_IMR90.pt`
- **Size:** 220 MB each (4 × 220 MB ≈ 880 MB)
- **Source:** Authors' Google Drive folder linked from the paper.
  https://drive.google.com/drive/folders/18DHZgsJqupNTnWmPrRiA3F1SrMro2q_H
- **Notes:**
  - These checkpoints use legacy key names (`l1GRU`, `l2GRU`) and 4096-row embeddings.
    The loader at `epintlm/models/epintlm.py:load_checkpoint_with_remapping` handles both.
  - Trained with `DummyGenomicFeatures` (zeros) — set `data.chromatin_features.enabled=false`
    when loading them, otherwise the gene-data inputs distribution will not match.
- **Acquisition:** `bash scripts/download_checkpoints.sh` (uses `gdown`).

## 3. HeLa-S3 Test Data (bundled subset of TargetFinder)

- **Files:**
  - `data/raw/targetfinder/sequence_data/HeLa-S3_enhancer_test.fasta` (11 MB)
  - `data/raw/targetfinder/sequence_data/HeLa-S3_promoter_test.fasta` (7 MB)
  - `data/raw/targetfinder/sequence_data/HeLa-S3_label_test.txt`
  - `data/raw/targetfinder/bed_files/HeLa-S3_enhancer_test.bed`
  - `data/raw/targetfinder/bed_files/HeLa-S3_promoter_test.bed`
- **Counts:** 3654 samples (174 positive, 3480 negative — matches paper Table 1).
- **Status:** Bundled — useful for inference smoke tests without any downloads.

## 4. TargetFinder Dataset (full, 6 cell lines)

- **Source:** Whalen, Schreiber, Wold, Pollard. *Enhancer–promoter interactions are encoded
  by complex genomic signatures on looping chromatin.* Nature Genetics, 2016.
  GitHub: https://github.com/shwhalen/targetfinder
- **Cell lines:** GM12878, HeLa-S3, HUVEC, IMR90, K562, NHEK
- **Train/test counts (from paper Table 1):**
  | Cell line | Train pos | Train neg | Test pos | Test neg |
  |-----------|-----------|-----------|----------|----------|
  | GM12878   | 38040     | 37980     | 211      | 4220     |
  | HeLa-S3   | 31320     | 31320     | 174      | 3480     |
  | HUVEC     | 27440     | 27360     | 152      | 3040     |
  | IMR90     | 22580     | 22500     | 125      | 2500     |
  | K562      | 35600     | 35550     | 197      | 3950     |
  | NHEK      | 23240     | 23040     | 129      | 2560     |
- **Acquisition (full pipeline, HPC):**
  ```bash
  # 1. Download pair tables (small, can run anywhere with internet):
  bash scripts/download_targetfinder.sh

  # 2. On HPC (needs hg19.fa + bedtools): convert to BED + extract FASTAs:
  HG19_FA=/path/to/hg19.fa bash scripts/build_dataset.sh
  ```
  Or as a single SLURM job:
  ```bash
  sbatch --export=ALL,HG19_FA=/scratch/$USER/hg19.fa slurm/build_dataset.slurm
  ```
- **Pair → BED converter:** `epintlm/tools/targetfinder_to_bed.py`
  (stratified 90/10 split with `--augment-train` for 1:1 class balance).
- **License:** Released alongside the paper for non-commercial research use.

## 5. hg19 Reference Genome

- **Source:** UCSC Genome Browser (or any standard hg19 mirror).
- **Required when:** Generating the `_train` FASTA files from TargetFinder pair tables.
- **Size:** ~3 GB (compressed; ~3.2 GB uncompressed).
- **Acquisition:** Not scripted — module-load on Grace (`module load BEDTools`)
  or download from `https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz`.

## 6. ENCODE Chromatin Tracks

Required only when `data.chromatin_features.enabled=true`. The pretrained checkpoints
were trained without these (using `DummyGenomicFeatures`), so most reproductions can
skip section 6 entirely.

- **Assays:** CTCF, DNase, H3K27ac, H3K4me1, H3K4me3 (5 per cell line × 6 cells = 30 BigWigs)
- **Format:** ENCODE fold-change-over-control BigWig, hg19.
- **Size:** ~1 GB per BigWig × 30 ≈ 30 GB.
- **Acquisition:**
  1. `bash scripts/download_encode.sh` — wraps `epintlm/tools/encode_downloader.py`,
     which queries the ENCODE REST API at runtime to find the current released BigWig per
     (cell, assay) (hardcoded accessions don't work — ENCODE deprecates files).
     Writes a `manifest.json` recording exact accessions used.
  2. (Pending) `epintlm/tools/bigwig_to_pt.py` would bin them to 500 bp tensors and write
     `data/chromatin_features/CTCF_DNase_6histone.500.json` + per-cell `.pt` files.
- **License:** ENCODE data release policy — free for research with attribution.

---

## Directory Layout (after acquisition)

```
data/
├── embeddings/
│   └── nucleotide_transformer_6_kmer_embedding.npy
├── raw/
│   ├── targetfinder/
│   │   ├── pairs/{cell}_pairs.csv
│   │   ├── bed_files/{cell}_{enhancer,promoter}_{train,test}.bed
│   │   └── sequence_data/{cell}_{enhancer,promoter}_{train,test}.fasta
│   │                     {cell}_label_{train,test}.txt
│   └── encode/{cell}/{CTCF,DNase,H3K27ac,H3K4me1,H3K4me3}.bigWig
├── chromatin_features/
│   ├── CTCF_DNase_6histone.500.json
│   └── *.pt
└── processed/                              # Generated by `python -m epintlm.cli.preprocess`
    ├── {cell}_combined_train.pt
    ├── {cell}_combined_val.pt
    ├── {cell}_combined_test.pt
    └── {cell}_cache.json

checkpoints/
├── L1_HELA.pt
├── L1_NU_GM12878.pt
├── L1_NU_HUVEC.pt
└── L1_NU_IMR90.pt
```

## Minimal-data path (inference only)

If you only want to run inference with the pretrained checkpoints:

1. Items 1, 2, 3 must be present (all bundled or scripted).
2. Set `data.chromatin_features.enabled=false` in your config (default).
3. `bash scripts/run_eval.sh RUN_DIR=runs/inference CHECKPOINTS_DIR=checkpoints`
