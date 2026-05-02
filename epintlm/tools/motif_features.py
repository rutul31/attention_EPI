"""Compute per-pair motif feature vectors from enhancer + promoter FASTAs.

Motif scoring approach:
  - Use a small curated TF panel (default: 27 motifs known to be enriched at EPIs:
    CTCF, YY1, RAD21, p53/p63, MYC, JUN/FOS family, GATA1/2/3, ELK1, SP1, etc.).
  - Score each FASTA sequence with each motif's PSSM via Bio.motifs (BioPython).
  - For each (enhancer, promoter) pair we emit a 2 * len(motifs)-dim vector
    [max_score_per_motif on enhancer, max_score_per_motif on promoter].
  - Default panel size 27 → 54-dim vector, padded to 55 dims to match the existing
    `model.gene_data.input_dim=55` slot. No model architecture change required.

If pyjaspar is available, motifs are pulled from JASPAR 2024 CORE.
Otherwise we fall back to a hard-coded set of consensus PWMs (degraded but functional).

Usage:
  python -m epintlm.tools.motif_features \\
    --enh-fasta data/raw/targetfinder/sequence_data/HeLa-S3_enhancer_train.fasta \\
    --pro-fasta data/raw/targetfinder/sequence_data/HeLa-S3_promoter_train.fasta \\
    --out data/processed/HeLa-S3_motifs_train.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# Curated panel of TFs known to be active at enhancer/promoter loci.
DEFAULT_MOTIFS = [
    "CTCF", "YY1", "RAD21", "MYC", "MAX",
    "JUN", "FOS", "JUNB", "FOSL1",
    "GATA1", "GATA2", "GATA3",
    "TP53", "TP63",
    "ELK1", "ETS1", "GABPA",
    "SP1", "KLF4",
    "FOXA1", "STAT1", "STAT3",
    "RUNX1", "REST",
    "USF1", "USF2",
    "TCF7L2",
]


def _read_fasta(path: Path) -> List[str]:
    """Return the non-header lines of a FASTA, uppercased, in file order."""
    out = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            out.append(line.upper())
    return out


def _load_pssms(motif_names: List[str]) -> List[Tuple[str, "object"]]:
    """Try pyjaspar first; fall back to consensus-derived PWMs from Bio.motifs if unavailable."""
    pssms: List[Tuple[str, "object"]] = []
    try:
        from pyjaspar import jaspardb
        jdb = jaspardb(release="JASPAR2024")
        for name in motif_names:
            hits = jdb.fetch_motifs(collection="CORE", tax_group=["vertebrates"], tf_name=name)
            if not hits:
                print(f"  [warn] no JASPAR hit for {name}; skipping.", file=sys.stderr)
                continue
            m = hits[0]
            # Use log-odds PSSM with uniform background
            pssm = m.counts.normalize(pseudocounts=0.5).log_odds()
            pssms.append((name, pssm))
        if pssms:
            print(f"  Loaded {len(pssms)}/{len(motif_names)} motifs from JASPAR.")
            return pssms
    except ImportError:
        print("  pyjaspar not installed; using hard-coded consensus fallbacks.", file=sys.stderr)
    except Exception as e:
        print(f"  pyjaspar failed ({e}); using consensus fallbacks.", file=sys.stderr)

    # Fallback: hand-curated consensus k-mers per TF, scored as exact-match counters.
    # This is much weaker but degrades gracefully when JASPAR isn't reachable.
    consensus = {
        "CTCF": "CCGCGNGGNGGCAG", "YY1": "CCATNTT", "RAD21": "CCGCGNGGNGGCAG",
        "MYC": "CACGTG", "MAX": "CACGTG",
        "JUN": "TGACTCA", "FOS": "TGACTCA", "JUNB": "TGACTCA", "FOSL1": "TGACTCA",
        "GATA1": "AGATAA", "GATA2": "AGATAA", "GATA3": "AGATAA",
        "TP53": "RRRCWWGYYY", "TP63": "RRRCWWGYYY",
        "ELK1": "CCGGAA", "ETS1": "CCGGAA", "GABPA": "CCGGAA",
        "SP1": "GGGGCGGGG", "KLF4": "GGGGTGGGG",
        "FOXA1": "TGTTTAC", "STAT1": "TTCNNNGAA", "STAT3": "TTCNNGGAA",
        "RUNX1": "TGTGGT", "REST": "NTCAGCACC",
        "USF1": "CACGTG", "USF2": "CACGTG",
        "TCF7L2": "CTTTGT",
    }

    class _ConsensusScorer:
        """Counts approximate consensus matches; ambiguous bases match anything."""
        def __init__(self, pattern: str):
            self.pattern = pattern.upper()

        def calculate(self, seq: str) -> List[float]:
            iupac = {"R": "AG", "Y": "CT", "W": "AT", "S": "CG", "K": "GT", "M": "AC", "N": "ACGT"}
            scores = []
            seq = seq.upper()
            L = len(self.pattern)
            for i in range(len(seq) - L + 1):
                window = seq[i : i + L]
                ok = True
                for w, p in zip(window, self.pattern):
                    if p in "ACGT":
                        if w != p:
                            ok = False
                            break
                    elif p in iupac:
                        if w not in iupac[p]:
                            ok = False
                            break
                scores.append(1.0 if ok else 0.0)
            return scores

    for name in motif_names:
        pat = consensus.get(name)
        if pat is None:
            continue
        pssms.append((name, _ConsensusScorer(pat)))
    print(f"  Loaded {len(pssms)} consensus fallback patterns.")
    return pssms


def _max_score_per_motif(seq: str, pssms) -> np.ndarray:
    """Return max PSSM score per motif for one sequence."""
    out = np.zeros(len(pssms), dtype=np.float32)
    for i, (_name, pssm) in enumerate(pssms):
        try:
            scores = pssm.calculate(seq)
        except Exception:
            continue
        if scores is None:
            continue
        # Bio.motifs.matrix.PositionSpecificScoringMatrix.calculate returns a numpy array
        # for sequences ≥ motif length, or a single float for exactly-matching length.
        if isinstance(scores, (int, float)):
            out[i] = float(scores)
        else:
            arr = np.asarray(scores, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[i] = float(arr.max())
    return out


def compute_motif_features(
    enh_fasta: Path,
    pro_fasta: Path,
    out_path: Path,
    motif_names: List[str] | None = None,
    target_dim: int = 55,
) -> torch.Tensor:
    """Score every (enhancer, promoter) pair with the motif panel and save a (N, target_dim) tensor."""
    motif_names = motif_names or DEFAULT_MOTIFS
    pssms = _load_pssms(motif_names)
    if not pssms:
        raise RuntimeError("No motifs could be loaded. Install pyjaspar or check fallback patterns.")

    enh_seqs = _read_fasta(enh_fasta)
    pro_seqs = _read_fasta(pro_fasta)
    if len(enh_seqs) != len(pro_seqs):
        raise ValueError(f"FASTA length mismatch: {len(enh_seqs)} vs {len(pro_seqs)}")

    n = len(enh_seqs)
    feat_per_seq = len(pssms)
    pair_dim = 2 * feat_per_seq
    if pair_dim > target_dim:
        raise ValueError(f"pair_dim {pair_dim} > target_dim {target_dim}; reduce panel.")

    out = np.zeros((n, target_dim), dtype=np.float32)
    print(f"Scoring {n} pairs against {feat_per_seq} motifs (pair_dim={pair_dim}, padded to {target_dim})...")
    for i in range(n):
        enh_v = _max_score_per_motif(enh_seqs[i], pssms)
        pro_v = _max_score_per_motif(pro_seqs[i], pssms)
        out[i, :feat_per_seq] = enh_v
        out[i, feat_per_seq:pair_dim] = pro_v
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(out)
    torch.save(tensor, out_path)
    print(f"Saved motif features → {out_path} (shape={tensor.shape})")
    return tensor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enh-fasta", required=True, type=Path)
    parser.add_argument("--pro-fasta", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-dim", type=int, default=55,
                        help="Output feature dim (must equal model.gene_data.input_dim).")
    args = parser.parse_args(argv)

    compute_motif_features(args.enh_fasta, args.pro_fasta, args.out, target_dim=args.target_dim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
