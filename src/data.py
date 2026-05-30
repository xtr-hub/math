"""Data loading and preprocessing."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_raw(filename: str) -> pd.DataFrame:
    """Load a dataset from data/raw/."""
    path = DATA_DIR / "raw" / filename
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {path.suffix}")


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save a processed DataFrame to data/processed/."""
    path = DATA_DIR / "processed" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
