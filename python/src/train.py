from __future__ import annotations
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from datasets import load_raw_parquet, process_and_save
from model import MLP


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("🚀 PolarLearning (Python) - PyTorch Training")
    print("===========================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n📊 Loading raw parquet dataset...")
    raw_df = load_raw_parquet()
    print(f"Loaded: {raw_df.height} rows, {raw_df.width} columns")

    print("\n🔄 Processing and saving processed CSV...")
    x_train, y_train, x_test, y_test = process_and_save(raw_df)

    X_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
    y_train_t = torch.tensor(y_train.to_numpy().reshape(-1, 1), dtype=torch.float32)
    X_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
    y_test_t = torch.tensor(y_test.to_numpy().reshape(-1, 1), dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train_t)
    batch_size = int(max(1, np.ceil(len(train_ds) / 10)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = MLP().to(device)
    criterion = nn.L1Loss()  # Match Rust MAE usage
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = int(os.getenv("EPOCHS", "1"))
    print(f"\n🧠 Training for {epochs} epochs, batch_size={batch_size}")

    set_seed(42)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        running /= len(train_loader)
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} - loss: {running:.6f}  ({dt:.2f}s)")

    # Eval
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device))
        test_loss = criterion(preds, y_test_t.to(device)).item()
    print(f"\nTest MAE: {test_loss:.6f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", "mlp.pt"))
    print("✅ Saved model to models/mlp.pt")


if __name__ == "__main__":
    main()
