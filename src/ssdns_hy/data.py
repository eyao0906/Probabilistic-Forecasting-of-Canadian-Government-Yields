"""Strict, read-only loading and audit of the preserved Canadian yield panel."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

MATURITIES_YEARS = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0, 30.0])
YIELD_COLUMNS = ("y0.25", "y0.5", "y1", "y2", "y5", "y7", "y10", "y30")


@dataclass(frozen=True)
class ForecastOrigin:
    origin_index: int
    target_index: int
    horizon_observations: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_weekly_panel(path: Path) -> tuple[list[date], np.ndarray]:
    """Load sorted weekly yields in percentage points without filling gaps."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(("Date", *YIELD_COLUMNS)).issubset(
            reader.fieldnames
        ):
            raise ValueError(f"{path} lacks the required Date and maturity columns")
        rows = list(reader)
    dates = [date.fromisoformat(row["Date"]) for row in rows]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("weekly dates must be nonempty, unique, and ascending")
    values = np.array(
        [[float(row[column]) for column in YIELD_COLUMNS] for row in rows],
        dtype=float,
    )
    if values.shape != (len(dates), len(YIELD_COLUMNS)):
        raise ValueError("yield panel has an unexpected shape")
    if not np.isfinite(values).all():
        raise ValueError("yield panel contains missing or non-finite values")
    return dates, values


def select_forecast_origins(
    dates: list[date],
    oos_start: date,
    oos_end: date,
    horizons: tuple[int, ...],
    stride_observations: int,
) -> list[ForecastOrigin]:
    """Select the frozen every-k-observation origin/target design."""
    if stride_observations < 1 or not horizons or any(h < 1 for h in horizons):
        raise ValueError("stride and all forecast horizons must be positive")
    if dates != sorted(set(dates)):
        raise ValueError("dates must be unique and ascending")
    oos_indices = [
        i for i, current in enumerate(dates) if oos_start <= current <= oos_end
    ]
    origins: list[ForecastOrigin] = []
    for origin in oos_indices[::stride_observations]:
        for horizon in horizons:
            target = origin + horizon
            if target < len(dates) and oos_start <= dates[target] <= oos_end:
                origins.append(ForecastOrigin(origin, target, horizon))
    return origins


def audit_input_hashes(root: Path) -> dict:
    """Compare preserved raw/processed data with the committed source manifest."""
    manifest_path = root / "data" / "processed" / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = root / manifest["raw_file"].replace("\\", "/")
    processed_path = root / manifest["processed_file"].replace("\\", "/")
    raw_actual = sha256_file(raw_path)
    processed_actual = sha256_file(processed_path)
    return {
        "raw": {
            "path": str(raw_path.relative_to(root)),
            "expected_sha256": manifest["raw_sha256"],
            "actual_sha256": raw_actual,
            "matches_manifest": raw_actual == manifest["raw_sha256"],
        },
        "processed": {
            "path": str(processed_path.relative_to(root)),
            "expected_sha256": manifest["processed_sha256"],
            "actual_sha256": processed_actual,
            "matches_manifest": processed_actual == manifest["processed_sha256"],
        },
    }
