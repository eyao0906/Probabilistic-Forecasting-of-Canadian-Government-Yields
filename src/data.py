"""Data loading, validation, and weekly sampling utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CANADA_WEEKLY,
    MATURITY_COLS,
    OUT_OF_SAMPLE_END,
    OUT_OF_SAMPLE_START,
    PROCESSED_DIR,
    ROOT,
    TAU,
    US_WEEKLY,
)


def _ensure_processed() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_weekly(country: str = "canada") -> pd.DataFrame:
    """Load pre-built weekly yield file for Canada or US.

    Parameters
    ----------
    country : str
        ``"canada"`` or ``"us"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Date`` and ``y0.25``..``y30`` columns in percent.
    """
    path = CANADA_WEEKLY if country.lower().startswith("ca") else US_WEEKLY
    if not path.exists():
        raise FileNotFoundError(f"Weekly file not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Basic validation
    missing = df[MATURITY_COLS].isna().sum().sum()
    if missing:
        raise ValueError(f"Weekly file contains {missing} missing yield values.")
    return df


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into initial training and out-of-sample periods.

    Returns
    -------
    train : pd.DataFrame
        Rows up to 2015-12-31 inclusive.
    test : pd.DataFrame
        Rows from 2016-01-01 onward within the requested end date.
    """
    train = df[df["Date"] <= "2015-12-31"].copy()
    test = df[(df["Date"] >= "2016-01-01") & (df["Date"] <= OUT_OF_SAMPLE_END)].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def select_origins(df: pd.DataFrame) -> pd.DataFrame:
    """Select every ``ORIGIN_STEP``-th weekly observation from the out-of-sample period.

    Keeps only origins whose realized target date falls within the requested period.
    """
    from .config import ORIGIN_STEP

    oos = df[df["Date"] >= "2016-01-01"].copy()
    origins = oos.iloc[::ORIGIN_STEP].copy()
    return origins.reset_index(drop=True)


def yields_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """Return yields as an (T, 8) array in the order defined by ``TAU``."""
    return df[MATURITY_COLS].to_numpy(dtype=float)


if __name__ == "__main__":
    _ensure_processed()
    ca = load_weekly("canada")
    us = load_weekly("us")
    print(f"Canada weekly: {len(ca)} rows, {ca['Date'].min().date()} -> {ca['Date'].max().date()}")
    print(f"US weekly: {len(us)} rows, {us['Date'].min().date()} -> {us['Date'].max().date()}")
    train, test = train_test_split(ca)
    origins = select_origins(test)
    print(f"Train: {len(train)}, Test: {len(test)}, Origins: {len(origins)}")
