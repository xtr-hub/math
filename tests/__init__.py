"""Tests for data module."""

import pandas as pd
from src.data import save_processed
from pathlib import Path
import tempfile


def test_save_and_load_processed_csv():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.csv"
        df.to_csv(path, index=False)
        loaded = pd.read_csv(path)
        assert len(loaded) == 2
        assert list(loaded.columns) == ["a", "b"]
