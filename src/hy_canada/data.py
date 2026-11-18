"""Load the existing complete weekly yield panel without changing it."""

from pathlib import Path

import numpy as np
import pandas as pd

MATURITIES = np.array([0.25, 0.5, 1, 2, 5, 7, 10, 30])
COLUMNS = ["Date", "y0.25", "y0.5", "y1", "y2", "y5", "y7", "y10", "y30"]


def load_panel(path: Path, start: str, end: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frame = pd.read_csv(path)
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"Expected ordered columns {COLUMNS}, got {list(frame.columns)}")
    dates = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be present, unique, and increasing")
    values = frame[COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Yield panel contains missing or nonfinite values")
    weeks = dates.dt.to_period("W-SUN")
    if weeks.duplicated().any():
        raise ValueError("More than one observation appears in a calendar week")
    if (weeks.astype("int64").diff().dropna() != 1).any():
        raise ValueError("Weekly panel has a missing calendar week")
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date > end_date or start_date < dates.iloc[0] or end_date > dates.iloc[-1]:
        raise ValueError("Requested range must be ordered and inside the available panel")
    selected = (dates >= start_date) & (dates <= end_date)
    if not selected.any():
        raise ValueError("No observations in requested range")
    return pd.DatetimeIndex(dates[selected]), values[selected.to_numpy()]
