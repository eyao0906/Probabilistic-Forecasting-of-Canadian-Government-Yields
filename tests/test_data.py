"""Quick sanity tests for data loading and validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MATURITY_COLS, TAU
from src.data import load_weekly, train_test_split, select_origins, yields_to_numpy


def test_load_canada():
    df = load_weekly("canada")
    assert set(MATURITY_COLS).issubset(df.columns)
    assert df["Date"].is_monotonic_increasing
    assert df[MATURITY_COLS].isna().sum().sum() == 0
    assert len(df) > 1000


def test_load_us():
    df = load_weekly("us")
    assert set(MATURITY_COLS).issubset(df.columns)
    assert df["Date"].is_monotonic_increasing
    assert len(df) > 1000


def test_train_test_split():
    df = load_weekly("canada")
    train, test = train_test_split(df)
    assert train["Date"].max() <= pd.Timestamp("2015-12-31")
    assert test["Date"].min() >= pd.Timestamp("2016-01-01")
    # Some rows beyond OUT_OF_SAMPLE_END are dropped by design
    assert len(train) + len(test) <= len(df)
    assert len(train) > 0 and len(test) > 0


def test_select_origins():
    df = load_weekly("canada")
    train, test = train_test_split(df)
    origins = select_origins(test)
    assert len(origins) > 100
    assert origins["Date"].min() >= pd.Timestamp("2016-01-01")


def test_yields_to_numpy():
    df = load_weekly("canada").head(10)
    arr = yields_to_numpy(df)
    assert arr.shape == (10, 8)
    assert np.all(np.isfinite(arr))


if __name__ == "__main__":
    test_load_canada()
    test_load_us()
    test_train_test_split()
    test_select_origins()
    test_yields_to_numpy()
    print("All data tests passed.")
