"""Generate nucleotide_transformer_6_kmer_embedding.npy.

Uses HuggingFace `InstaDeepAI/nucleotide-transformer-500m-human-ref` to produce a
4097 x 1280 embedding lookup table for all 6-mers (4^6 = 4096) plus a zero null token.
"""

from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
K = 6


def generate_all_kmers(k: int = K) -> list[str]:
    bases = ["A", "C", "G", "T"]
    return ["".join(p) for p in itertools.product(bases, repeat=k)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    print(f"Loading {MODEL_NAME} (downloads ~2GB on first run)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()

    hidden_size = model.config.hidden_size
    kmers = generate_all_kmers(K)
    print(f"Embedding {len(kmers)} 6-mers (hidden_size={hidden_size})")

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(kmers), args.batch_size):
            batch = kmers[i : i + args.batch_size]
            tokens = tokenizer(batch, return_tensors="pt", padding=True)
            outputs = model(**tokens, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(pooled.cpu().numpy())
            if (i // args.batch_size) % 10 == 0:
                print(f"  {i + len(batch)}/{len(kmers)}")

    embeddings = np.vstack(all_embeddings)
    null_row = np.zeros((1, hidden_size), dtype=np.float32)
    matrix = np.vstack([embeddings, null_row]).astype(np.float32)
    print(f"Final shape: {matrix.shape}; any NaN: {np.isnan(matrix).any()}")

    np.save(args.output, matrix)
    print(f"Saved → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
