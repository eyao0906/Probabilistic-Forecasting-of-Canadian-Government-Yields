"""Reproducible Bank of Canada daily-to-weekly data pipeline."""
from __future__ import annotations

if __name__ == "__main__":
    import sys as _legacy_sys
    from pathlib import Path as _LegacyPath
    _legacy_sys.path.insert(0, str(_LegacyPath(__file__).resolve().parent))
    from legacy_guard import require_legacy_opt_in as _require_legacy_opt_in
    _require_legacy_opt_in()

import argparse
import hashlib
import json
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "boc_daily_2000_2026.csv"
OUT = ROOT / "data" / "processed" / "canada_weekly.csv"
MANIFEST = ROOT / "data" / "processed" / "source_manifest.json"
SOURCE_PAGE = "https://www.bankofcanada.ca/rates/interest-rates/bond-yield-curves/"
MATURITIES = [0.25, 0.5, 1, 2, 5, 7, 10, 30]
COLUMNS = {0.25: "ZC025YR", 0.5: "ZC050YR", 1: "ZC100YR", 2: "ZC200YR",
           5: "ZC500YR", 7: "ZC700YR", 10: "ZC1000YR", 30: "ZC3000YR"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(start: str, end: str, force: bool = False) -> Path:
    """Fetch the official BoC CSV; never replace an existing raw file implicitly."""
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if RAW.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {RAW}; pass --force after preserving it.")
    url = ("https://www.bankofcanada.ca/stats/results/csv?lookupPage=lookup_yield_curve.php"
           f"&dFrom={start}&dTo={end}")
    request = urllib.request.Request(url, headers={"User-Agent": "MQF-MRP-research/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, RAW.open("wb") as target:
        while chunk := response.read(1 << 20):
            target.write(chunk)
    return RAW


def prepare(raw_path: Path = RAW, out_path: Path = OUT) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]
    if "Date" not in raw:
        raise ValueError("BoC source is missing its Date column")
    required = [COLUMNS[t] for t in MATURITIES]
    missing_columns = [c for c in required if c not in raw]
    if missing_columns:
        raise ValueError(f"BoC source is missing maturity fields: {missing_columns}")
    daily = raw[["Date", *required]].copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    for c in required:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    if daily["Date"].isna().any():
        raise ValueError("Unparseable dates in raw source")
    if daily["Date"].duplicated().any():
        raise ValueError("Duplicate source dates")
    daily = daily.sort_values("Date")
    valid = daily.dropna(subset=required).copy()
    if valid.empty:
        raise ValueError("No complete eight-maturity yield curves")
    # The official series is decimal yields; analysis is reported in percentage points.
    weekly = valid.copy()
    iso = weekly["Date"].dt.isocalendar()
    weekly["_iso_year"] = iso.year.to_numpy()
    weekly["_iso_week"] = iso.week.to_numpy()
    weekly = weekly.groupby(["_iso_year", "_iso_week"], sort=True, as_index=False).tail(1)
    weekly = weekly.sort_values("Date").reset_index(drop=True)
    result = pd.DataFrame({"Date": weekly["Date"].dt.strftime("%Y-%m-%d")})
    for tau, col in zip(MATURITIES, required):
        result[f"y{tau:g}"] = weekly[col].to_numpy(dtype=float) * 100.0
    if result.isna().any().any() or result["Date"].duplicated().any():
        raise AssertionError("Processed weekly data failed completeness/uniqueness checks")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, float_format="%.10g")
    stat = raw_path.stat()
    manifest = {
        "source": "Bank of Canada daily zero-coupon Government of Canada yield curves",
        "source_page": SOURCE_PAGE,
        "download_endpoint": "https://www.bankofcanada.ca/stats/results/csv?lookupPage=lookup_yield_curve.php&dFrom=2000-01-01&dTo=2026-10-09",
        "retrieval_evidence": "Existing raw file supplied in the project; filesystem modified time used as a retrieval proxy, not an independently authenticated server timestamp.",
        "raw_file": str(raw_path.relative_to(ROOT)),
        "raw_sha256": sha256(raw_path),
        "raw_filesystem_mtime_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "processing_run_utc": datetime.now(timezone.utc).isoformat(),
        "daily_source_rows": int(len(raw)),
        "complete_daily_rows_eight_maturities": int(len(valid)),
        "weekly_rows": int(len(result)),
        "date_first": str(result["Date"].iloc[0]),
        "date_last": str(result["Date"].iloc[-1]),
        "maturities_years": MATURITIES,
        "raw_units": "decimal yields",
        "processed_units": "percentage points (decimal yield multiplied by 100)",
        "sampling": "last available complete observation in each ISO calendar week; not interpolated or averaged",
        "missingness": "rows missing any of the eight selected maturities are excluded before weekly sampling",
        "processed_file": str(out_path.relative_to(ROOT)),
        "processed_sha256": sha256(out_path),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download the BoC raw file before processing")
    parser.add_argument("--force", action="store_true", help="allow raw file replacement when --download is used")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    raw_path = download_raw(args.start, args.end, args.force) if args.download else args.raw
    frame, manifest = prepare(raw_path, args.out)
    print(json.dumps({"rows": len(frame), "first": manifest["date_first"],
                      "last": manifest["date_last"], "manifest": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
