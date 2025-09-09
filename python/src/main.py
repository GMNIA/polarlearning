from __future__ import annotations
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from datasets import load_raw_dataset, process_and_save
from model import MLP


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data():
    print("\n📊 Loading raw dataset (parquet or CSV fallback)...")
    raw_df = load_raw_dataset()
    print(f"Loaded: {raw_df.height} rows, {raw_df.width} columns")

    print("\n🔄 Processing and saving processed CSV...")
    x_train, y_train, x_test, y_test = process_and_save(raw_df)

    X_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
    y_train_t = torch.tensor(y_train.to_numpy().reshape(-1, 1), dtype=torch.float32)
    X_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
    y_test_t = torch.tensor(y_test.to_numpy().reshape(-1, 1), dtype=torch.float32)

    return X_train, y_train_t, X_test, y_test_t


def train_model(model: MLP, X_train: torch.Tensor, y_train: torch.Tensor, *, device: torch.device, epochs: int = 1, lr: float = 1e-3):
    ds = TensorDataset(X_train, y_train)
    batch_size = int(max(1, np.ceil(len(ds) / 10)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    criterion = nn.L1Loss()  # MAE
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print(f"\n🧠 Training for {epochs} epochs, batch_size={batch_size}")
    for epoch in range(epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        running /= len(loader)
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} - loss: {running:.6f}  ({dt:.2f}s)")


def evaluate_model(model: MLP, X_test: torch.Tensor, y_test: torch.Tensor, *, device: torch.device) -> float:
    model.eval()
    criterion = nn.L1Loss()
    with torch.no_grad():
        preds = model(X_test.to(device))
        test_mae = criterion(preds, y_test.to(device)).item()
    print(f"\nTest MAE: {test_mae:.6f}")
    return test_mae


def save_model(model: MLP, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Saved model to {path}")


def load_model(path: str, *, device: torch.device) -> MLP:
    model = MLP().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"📦 Loaded model from {path}")
    return model


def main():
    print("🚀 PolarLearning (Python) - Train & Test")
    print("======================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(42)
    X_train, y_train, X_test, y_test = prepare_data()
    model = MLP().to(device)

    epochs = int(os.getenv("EPOCHS", "1"))
    lr = float(os.getenv("LR", "1e-3"))
    train_model(model, X_train, y_train, device=device, epochs=epochs, lr=lr)

    # Evaluate trained model
    _ = evaluate_model(model, X_test, y_test, device=device)

    # Save and re-load, then evaluate again to validate serialization
    save_path = os.path.join("models", "mlp.pt")
    save_model(model, save_path)
    reloaded = load_model(save_path, device=device)
    _ = evaluate_model(reloaded, X_test, y_test, device=device)


if __name__ == "__main__":
    main()
