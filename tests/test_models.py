"""Shape, token-count, and training-loop smoke tests on random inputs."""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.engine import calculate_metrics, train
from src.model_configs import MODEL_BUILDERS
from src.models import create_modular_cnn

EXPECTED_TOKENS = {"A_5blk_2tok": 2, "B_5blk_36tok": 36, "C_5blk_392tok": 392,
                   "D_4blk_36tok": 36, "E_3blk_392tok": 392}


@pytest.mark.parametrize("name", list(MODEL_BUILDERS))
def test_forward_pass_returns_one_age_per_scan(name):
    model = MODEL_BUILDERS[name]().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 57, 57, 71))
    assert out.shape == (2, 1)
    assert model.get_model_info()["n_params"] > 0


@pytest.mark.parametrize("name,tokens", EXPECTED_TOKENS.items())
def test_hybrid_token_counts(name, tokens):
    assert MODEL_BUILDERS[name]().get_model_info()["num_tokens"] == tokens


def test_calculate_metrics_on_perfect_predictions():
    y = np.array([9.0, 10.5, 12.0, 14.0])
    m = calculate_metrics(y, y)
    assert m["MSE"] == 0 and m["MAE"] == 0
    assert m["R_squared"] == pytest.approx(1.0) and m["Pearson"] == pytest.approx(1.0)


def test_train_runs_and_returns_history():
    x, y = torch.randn(16, 1, 16, 16, 16), torch.rand(16) * 5 + 9
    ds = TensorDataset(x, y)
    model = create_modular_cnn([{"filters": 4}, {"filters": 8}], [8])
    _, holdout_loss, train_losses, val_losses = train(
        model, ds, ds, ds, num_epochs=2, device="cpu", lr=1e-3,
        optimizer_type="AdamW", batch_size=8, num_workers=0, verbose=False)
    assert len(train_losses) == len(val_losses) == 2
    assert np.isfinite(holdout_loss)
