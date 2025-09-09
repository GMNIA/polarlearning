import inspect
import torch
import numpy as np
import pytest

from model import MLP
from main import train_model, set_seed


def test_train_model_default_epochs_is_200():
    sig = inspect.signature(train_model)
    assert sig.parameters["epochs"].default == 200


@pytest.mark.longtest
def test_training_runs_with_200_epochs_but_early_stops_quickly():
    # Small, easy synthetic dataset; early stopping with small patience keeps it fast
    set_seed(123)
    n, d = 64, 8
    X = torch.randn(n, d, dtype=torch.float32)
    # Simple target to make learning feasible
    y = (X[:, :2].sum(dim=1, keepdim=True) * 0.5).contiguous()

    device = torch.device("cpu")
    model = MLP().to(device)

    # Snapshot weights
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # epochs=200 but patience small so it won't actually do 200 iterations
    train_model(
        model,
        X,
        y,
        device=device,
        epochs=200,
        lr=1e-2,
        val_split=0.2,
        patience=2,
        min_delta=0.0,
    )

    after = model.state_dict()
    # At least one parameter should have changed
    changed = any(not torch.allclose(before[k], after[k]) for k in before)
    assert changed
