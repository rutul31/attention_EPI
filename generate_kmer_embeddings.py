"""
Generate nucleotide_transformer_6_kmer_embedding.npy

Uses the HuggingFace Nucleotide Transformer v2 model to produce a
4097 x 1280 embedding lookup table for all possible 6-mers (4^6 = 4096)
plus one null/padding token.

Reference:
  EPINTLM: enhancer–promoter prediction with pretrained k-mer embeddings
  and residual cross-attention (Briefings in Bioinformatics, 2026)
"""

import itertools
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
K = 6
OUTPUT_FILE = "nucleotide_transformer_6_kmer_embedding.npy"

def generate_all_kmers(k=6):
    """Generate all 4^k DNA k-mers."""
    bases = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in itertools.product(bases, repeat=k)]

def main():
    print(f"Loading Nucleotide Transformer model: {MODEL_NAME}")
    print("(This may download ~2GB on first run)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    
    # Get the hidden size from the model config
    hidden_size = model.config.hidden_size
    print(f"Model hidden size: {hidden_size}")
    
    kmers = generate_all_kmers(K)
    print(f"Generated {len(kmers)} unique {K}-mers")
    
    # We'll process k-mers in batches to be efficient
    batch_size = 64
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(kmers), batch_size):
            batch_kmers = kmers[i:i+batch_size]
            
            # Tokenize the k-mers
            tokens = tokenizer(batch_kmers, return_tensors="pt", padding=True)
            
            # Get hidden states from the base model (not the MLM head)
            outputs = model(**tokens, output_hidden_states=True)
            
            # Use the last hidden state
            last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
            
            # Mean-pool across sequence positions (excluding special tokens)
            # Attention mask tells us which positions are real tokens
            mask = tokens['attention_mask'].unsqueeze(-1)  # (batch, seq_len, 1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, hidden_size)
            
            all_embeddings.append(pooled.cpu().numpy())
            
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {i + len(batch_kmers)}/{len(kmers)} k-mers...")
    
    # Stack all embeddings: shape (4096, hidden_size)
    embeddings = np.vstack(all_embeddings)
    
    # Add null/padding token embedding (zeros) at index 4096
    null_embedding = np.zeros((1, hidden_size), dtype=np.float32)
    embedding_matrix = np.vstack([embeddings, null_embedding]).astype(np.float32)
    
    print(f"\nFinal embedding matrix shape: {embedding_matrix.shape}")
    print(f"Any NaN values: {np.isnan(embedding_matrix).any()}")
    
    np.save(OUTPUT_FILE, embedding_matrix)
    print(f"✅ Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
