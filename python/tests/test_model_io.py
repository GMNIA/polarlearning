import os
import torch
import numpy as np
from src.model import MLP


def test_saved_model_loads_and_runs(tmp_path):
    # Create a tiny model and save it
    model = MLP()
    save_path = tmp_path / "mlp.pt"
    torch.save(model.state_dict(), save_path)

    # Load back
    loaded = MLP()
    loaded.load_state_dict(torch.load(save_path, map_location="cpu"))
    loaded.eval()

    # Run a forward pass
    x = torch.randn(4, 8)
    with torch.no_grad():
        y = loaded(x)

    assert y.shape == (4, 1)
