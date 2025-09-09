from __future__ import annotations
import os
import time
from typing import Dict
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


def train_model(
    model: MLP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 20,
    min_delta: float = 0.0,
    optimizer_name: str = "sgd",
):
    """
    Train with MAE loss using a validation split and early stopping.

    Defaults reflect common literature for small tabular MLPs:
    - up to ~200 epochs with early stopping (patience ~20)
    - ReduceLROnPlateau scheduler on validation metric
    """
    assert 0.0 < val_split < 0.5, "val_split should be in (0, 0.5)"

    n = X_train.shape[0]
    n_val = max(1, int(n * val_split))
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    ds_tr = TensorDataset(X_train[tr_idx], y_train[tr_idx])
    ds_val = TensorDataset(X_train[val_idx], y_train[val_idx])

    batch_size = int(max(16, np.ceil(len(ds_tr) / 10)))
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    criterion = nn.L1Loss()  # MAE
    
    # Configurable optimizer
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Use 'sgd' or 'adam'")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    print(
        f"\n🧠 Training up to {epochs} epochs, bs={batch_size}, val={val_split:.0%}, patience={patience}"
    )
    print(f"    Optimizer: {optimizer_name.upper()}, LR: {lr}")

    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    wait = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        running /= max(1, len(loader_tr))

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in loader_val:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                vloss = criterion(preds, yb).item()
                val_loss += vloss
            val_loss /= max(1, len(loader_val))

        scheduler.step(val_loss)

        dt = time.time() - t0
        print(
            f"Epoch {epoch+1}/{epochs} - train_mae: {running:.6f}  val_mae: {val_loss:.6f}  ({dt:.2f}s)"
        )

        # Early stopping check
        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(
                    f"⏹️ Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)"
                )
                break

    # Restore best weights
    model.load_state_dict(best_state)


def evaluate_model(model: MLP, X_test: torch.Tensor, y_test: torch.Tensor, *, device: torch.device) -> Dict[str, float]:
    """
    Comprehensive evaluation on test dataset with multiple metrics.
    """
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device))
        y_test_dev = y_test.to(device)
        
        # Calculate multiple regression metrics
        mae = nn.L1Loss()(preds, y_test_dev).item()
        mse = nn.MSELoss()(preds, y_test_dev).item()
        rmse = np.sqrt(mse)
        
        # R-squared (coefficient of determination)
        y_mean = torch.mean(y_test_dev)
        ss_tot = torch.sum((y_test_dev - y_mean) ** 2).item()
        ss_res = torch.sum((y_test_dev - preds) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Mean Absolute Percentage Error (MAPE)
        mape = torch.mean(torch.abs((y_test_dev - preds) / torch.clamp(torch.abs(y_test_dev), min=1e-8))).item() * 100
        
        # Additional statistics
        pred_std = torch.std(preds).item()
        target_std = torch.std(y_test_dev).item()
        
        metrics = {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'pred_std': pred_std,
            'target_std': target_std
        }
        
    print(f"\n📊 Test Dataset Evaluation Results:")
    print(f"   MAE (Mean Absolute Error): {mae:.6f}")
    print(f"   MSE (Mean Squared Error):  {mse:.6f}")
    print(f"   RMSE (Root Mean Sq Error): {rmse:.6f}")
    print(f"   R² (Coefficient of Determ): {r2:.6f}")
    print(f"   MAPE (Mean Abs % Error):   {mape:.2f}%")
    print(f"   Prediction Std Dev:        {pred_std:.6f}")
    print(f"   Target Std Dev:            {target_std:.6f}")
    
    # Performance analysis
    mae_pct = (mae / target_std) * 100 if target_std > 0 else 0
    print(f"\n📈 Performance Analysis:")
    print(f"   MAE as % of target std: {mae_pct:.1f}%")
    
    if mae_pct < 30:
        print("   ✅ Excellent performance (MAE < 30% of target std)")
    elif mae_pct < 50:
        print("   ✅ Good performance (MAE < 50% of target std)")
    elif mae_pct < 70:
        print("   ⚠️  Acceptable performance (MAE < 70% of target std)")
    else:
        print("   ❌ Poor performance (MAE > 70% of target std)")
        print("   💡 Tuning suggestions:")
        print("      - Try Adam optimizer: docker run -e OPTIMIZER=adam ...")
        print("      - Lower learning rate: docker run -e LR=0.0005 ...")
        print("      - More epochs: docker run -e EPOCHS=300 -e PATIENCE=30 ...")
    
    return metrics


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

    # Literature-aligned defaults: up to 200 epochs with early stopping on val MAE
    epochs = int(os.getenv("EPOCHS", "200"))
    lr = float(os.getenv("LR", "1e-3"))
    patience = int(os.getenv("PATIENCE", "20"))
    val_split = float(os.getenv("VAL_SPLIT", "0.2"))
    optimizer_name = os.getenv("OPTIMIZER", "sgd")
    
    train_model(
        model,
        X_train,
        y_train,
        device=device,
        epochs=epochs,
        lr=lr,
        val_split=val_split,
        patience=patience,
        optimizer_name=optimizer_name,
    )

    # Evaluate trained model on test dataset
    print(f"\n🎯 Evaluating model on test dataset ({X_test.shape[0]} samples)...")
    train_metrics = evaluate_model(model, X_test, y_test, device=device)

    # Save and re-load, then evaluate again to validate serialization
    save_path = os.path.join("models", "mlp.pt")
    save_model(model, save_path)
    reloaded = load_model(save_path, device=device)
    
    print(f"\n🔄 Re-evaluating reloaded model to verify serialization...")
    reload_metrics = evaluate_model(reloaded, X_test, y_test, device=device)
    
    # Verify metrics are consistent after save/load
    mae_diff = abs(train_metrics['mae'] - reload_metrics['mae'])
    if mae_diff < 1e-6:
        print(f"✅ Serialization verified: MAE difference = {mae_diff:.2e}")
    else:
        print(f"⚠️  Warning: MAE differs after reload by {mae_diff:.6f}")
        
    # Training summary
    print(f"\n📋 Training Summary:")
    print(f"   • Default epochs: {epochs} (early stopping: patience={patience})")
    print(f"   • Learning rate: {lr}")
    print(f"   • Validation split: {val_split:.0%}")
    print(f"   • Final test MAE: {train_metrics['mae']:.6f}")
    print(f"   • Final test R²: {train_metrics['r2']:.6f}")
    print(f"   • Model saved to: {save_path}")


if __name__ == "__main__":
    main()
