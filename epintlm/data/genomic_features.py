"""Real chromatin-feature loader — reads BED enhancer/promoter coords + per-cell .pt feature tracks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

HG19_CHROM_SIZES = {
    "chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276,
    "chr5": 180915260, "chr6": 171115067, "chr7": 159138663, "chrX": 155270560,
    "chr8": 146364022, "chr9": 141213431, "chr10": 135534747, "chr11": 135006516,
    "chr12": 133851895, "chr13": 115169878, "chr14": 107349540, "chr15": 102531392,
    "chr16": 90354753, "chr17": 81195210, "chr18": 78077248, "chr20": 63025520,
    "chrY": 59373566, "chr19": 59128983, "chr22": 51304566, "chr21": 48129895,
}


class GenomicFeatures(Dataset):
    """Loads per-cell-line chromatin track .pt files and slices feature bins per sample."""

    def __init__(
        self,
        enh_bed: str | Path,
        pro_bed: str | Path,
        feats_config_path: str | Path,
        feats_order: Sequence[str],
        cell: str,
        enh_seq_len: int = 3000,
        pro_seq_len: int = 2500,
        bin_size: int = 500,
    ):
        self.enh_bed = Path(enh_bed)
        self.pro_bed = Path(pro_bed)
        self.cell = cell
        self.enh_seq_len = int(enh_seq_len)
        self.pro_seq_len = int(pro_seq_len)
        self.bin_size = int(bin_size)
        self.feats_order = list(feats_order)
        self.num_feats = len(self.feats_order)

        self.feats_config = self._resolve_paths(Path(feats_config_path))
        self.chrom_bins = {chrom: length // self.bin_size for chrom, length in HG19_CHROM_SIZES.items()}

        self.samples: List[tuple] = []
        self.feats: Dict[str, Dict[str, torch.Tensor]] = {}
        self._load_datasets()

    @staticmethod
    def _resolve_paths(config_path: Path) -> Dict[str, Dict[str, str]]:
        with config_path.open("r") as f:
            cfg = json.load(f)

        location = cfg.pop("_location", str(config_path.parent))
        for cell, assays in cfg.items():
            for assay, fn in assays.items():
                cfg[cell][assay] = os.path.join(location, fn)
        return cfg

    def _load_datasets(self) -> None:
        with self.enh_bed.open("r") as f1, self.pro_bed.open("r") as f2:
            for row1, row2 in zip(f1, f2):
                r1 = row1.strip().split("\t")
                r2 = row2.strip().split("\t")

                label = int(r1[0])
                enh_chrom, enh_start, enh_end = r1[1], int(float(r1[2])), int(float(r1[3]))
                pro_chrom, pro_start, pro_end = r2[1], int(float(r2[2])), int(float(r2[3]))
                cell = r1[-1] if "all" in str(self.enh_bed) else self.cell

                enh_mid = (enh_start + enh_end) // 2
                pro_mid = (pro_start + pro_end) // 2

                eb_start = max(0, (enh_mid - self.enh_seq_len // 2) // self.bin_size)
                eb_end = min(self.chrom_bins[enh_chrom], (enh_mid + self.enh_seq_len // 2) // self.bin_size)
                pb_start = max(0, (pro_mid - self.pro_seq_len // 2) // self.bin_size)
                pb_end = min(self.chrom_bins[pro_chrom], (pro_mid + self.pro_seq_len // 2) // self.bin_size)

                self.samples.append((eb_start, eb_end, pb_start, pb_end, enh_chrom, pro_chrom, cell, label))

                if cell not in self.feats:
                    self.feats[cell] = {feat: torch.load(self.feats_config[cell][feat]) for feat in self.feats_order}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> torch.Tensor:
        eb_s, eb_e, pb_s, pb_e, enh_c, pro_c, cell, _ = self.samples[idx]
        enh_ar = torch.zeros((0, eb_e - eb_s))
        pro_ar = torch.zeros((0, pb_e - pb_s))

        for feat in self.feats_order:
            enh_ar = torch.cat((enh_ar, self.feats[cell][feat][enh_c][eb_s:eb_e].view(1, -1)), dim=0)
            pro_ar = torch.cat((pro_ar, self.feats[cell][feat][pro_c][pb_s:pb_e].view(1, -1)), dim=0)

        return torch.cat((torch.flatten(enh_ar), torch.flatten(pro_ar)), dim=0)
