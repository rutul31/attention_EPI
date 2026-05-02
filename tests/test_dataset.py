import numpy as np
import torch

from epintlm.data.dataset import SeqGenDataset
from epintlm.data.dummy_features import DummyGenomicFeatures


def test_dummy_genomic_features_shape():
    ds = DummyGenomicFeatures(size=10, feat_dim=55)
    assert len(ds) == 10
    assert ds[0].shape == (55,)
    assert (ds[0] == 0).all()


def test_seqgen_dataset_iteration():
    n = 5
    enh = torch.randint(0, 4097, (n, 100), dtype=torch.long)
    pro = torch.randint(0, 4097, (n, 80), dtype=torch.long)
    gene = DummyGenomicFeatures(size=n, feat_dim=55)
    labels = np.array([0, 1, 0, 1, 1], dtype=np.int64)

    ds = SeqGenDataset(enh, pro, gene, labels)
    assert len(ds) == n

    e, p, g, lbl = ds[2]
    assert e.shape == (100,)
    assert p.shape == (80,)
    assert g.shape == (55,)
    assert int(lbl) == 0
