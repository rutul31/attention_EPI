import json

from epintlm.training.early_stopping import EarlyStopping
from epintlm.training.metrics import MetricsTracker


def test_early_stopping_triggers():
    es = EarlyStopping(patience=2, min_delta=0.01)
    es(0.5)            # best so far
    es(0.4)            # worse — counter=1
    es(0.4)            # still worse — counter=2 → triggers
    assert es.early_stop is True


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=3, min_delta=0.01)
    es(0.5)
    es(0.4)            # counter=1
    es(0.6)            # better → counter resets
    assert es.counter == 0
    assert es.best_score == 0.6
    assert es.early_stop is False


def test_metrics_tracker_save_roundtrip(tmp_path):
    mt = MetricsTracker()
    mt.update(epoch=0, train_loss=0.7, val_loss=0.6, val_aupr=0.8, val_auc=0.9, learning_rate=1e-3)
    mt.update(epoch=1, train_loss=0.5, val_loss=0.5, val_aupr=0.85, val_auc=0.92, learning_rate=1e-3)
    out = tmp_path / "metrics.json"
    mt.save(out)
    loaded = json.loads(out.read_text())
    assert loaded["epoch"] == [0, 1]
    assert loaded["val_aupr"] == [0.8, 0.85]
