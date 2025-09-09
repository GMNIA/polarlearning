from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple
import polars as pl
from sklearn.preprocessing import StandardScaler
import numpy as np

CALIFORNIA_COLUMNS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "median_house_value",
]

@dataclass
class DatasetPaths:
    raw_csv: str
    processed_csv: str
    raw_parquet: str


def resolve_datasets_root() -> str:
    env = os.getenv("DATASETS_DIR")
    if env and os.path.exists(env):
        return env
    if os.path.exists("datasets/CaliforniaHousing"):  # inside python/
        return "datasets"
    if os.path.exists("../datasets/CaliforniaHousing"):  # project root
        return "../datasets"
    return "datasets"


def get_california_paths() -> DatasetPaths:
    root = resolve_datasets_root()
    raw_dir = os.path.join("model-input-data", "raw")
    proc_dir = os.path.join("model-input-data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    return DatasetPaths(
        raw_csv=os.path.join(raw_dir, "CaliforniaHousing.csv"),
        processed_csv=os.path.join(proc_dir, "CaliforniaHousing.csv"),
        raw_parquet=os.path.join(root, "CaliforniaHousing", "cal_housing.parquet"),
    )


def load_raw_dataset() -> pl.DataFrame:
    paths = get_california_paths()
    # Prefer parquet if available
    if os.path.exists(paths.raw_parquet):
        df = pl.read_parquet(paths.raw_parquet)
        cols = [c for c in CALIFORNIA_COLUMNS if c in df.columns]
        if len(cols) == len(CALIFORNIA_COLUMNS):
            df = df.select(CALIFORNIA_COLUMNS)
        return df

    # Fallback: read the .data file (comma-separated, no header)
    root = resolve_datasets_root()
    data_path = os.path.join(root, "CaliforniaHousing", "cal_housing.data")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {paths.raw_parquet} or {data_path}")

    df = pl.read_csv(data_path, has_header=False)
    # Assign canonical column names
    if df.width == len(CALIFORNIA_COLUMNS):
        df = df.rename({old: new for old, new in zip(df.columns, CALIFORNIA_COLUMNS)})
    else:
        # If widths mismatch, try to select first 9 cols and rename
        df = df.select([pl.all().slice(0, 9)])
        df = df.rename({old: new for old, new in zip(df.columns, CALIFORNIA_COLUMNS)})
    return df


def process_and_save(df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    paths = get_california_paths()
    
    # Save raw data first (before any processing)
    df.write_csv(paths.raw_csv)
    print(f"💾 Saved raw data to: {paths.raw_csv}")
    
    # Split features/target
    features = df.select(CALIFORNIA_COLUMNS[:-1])
    target = df.select([CALIFORNIA_COLUMNS[-1]])

    # Scale features using StandardScaler (zero mean, unit variance)
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(features.to_numpy())
    features_scaled = pl.DataFrame(X_scaled, schema=features.columns)
    
    # Scale target values as well (important for regression!)
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(target.to_numpy())
    target_scaled = pl.DataFrame(y_scaled, schema=target.columns)
    
    print(f"📊 Feature scaling: mean={features.to_numpy().mean():.1f} → {X_scaled.mean():.6f}")
    print(f"📊 Target scaling: mean={target.to_numpy().mean():.1f} → {y_scaled.mean():.6f}")
    print(f"📊 Target range: [{target.to_numpy().min():.0f}, {target.to_numpy().max():.0f}] → [{y_scaled.min():.2f}, {y_scaled.max():.2f}]")

    # Save processed CSV (scaled features + scaled target)
    out = pl.concat([features_scaled, target_scaled], how="horizontal")
    out.write_csv(paths.processed_csv)
    print(f"� Saved processed data (both features & target scaled) to: {paths.processed_csv}")

    # Train/test split 80/20
    n = out.height
    n_train = int(0.8 * n)
    x_train = out.select(features.columns).head(n_train)
    y_train = out.select([target.columns[0]]).head(n_train)
    x_test = out.select(features.columns).tail(n - n_train)
    y_test = out.select([target.columns[0]]).tail(n - n_train)
    return x_train, y_train, x_test, y_test
