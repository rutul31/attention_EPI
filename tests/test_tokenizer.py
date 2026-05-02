import torch

from epintlm.data.tokenizer import build_vocab, create_tokenizer


def test_vocab_size_for_k6():
    vocab = build_vocab(6)
    assert len(vocab) == 4 ** 6 + 1
    assert vocab["null"] == 4096


def test_tokenizer_basic_kmer():
    tok = create_tokenizer(k=3)
    out = tok("ACGT")
    # k=3 gives 4-3+1 = 2 tokens: ACG, CGT
    assert out.shape == (2,)
    assert out.dtype == torch.long


def test_tokenizer_unknown_chars_map_to_null():
    tok = create_tokenizer(k=3)
    # 'NNN' isn't in the vocab; should map to null index
    out = tok("NNN")
    assert int(out[0]) == 4 ** 3
