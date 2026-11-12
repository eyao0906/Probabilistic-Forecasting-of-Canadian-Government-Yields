"""Reproducible Bank of Canada daily-to-weekly data pipeline (stdlib only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

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


def path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def download_raw(start: str, end: str, force: bool = False, destination: Path | None = None) -> Path:
    """Validated, atomic download to an explicit new vintage; never overwrite raw."""
    if __package__:
        from .acquisition import acquire_one, atomic_new
    else:
        from acquisition import acquire_one, atomic_new
    target = Path(destination) if destination is not None else RAW
    if target.exists():
        raise FileExistsError(f"Raw source exists: {target}; choose a new --raw vintage path. --force cannot replace it.")
    date.fromisoformat(start); date.fromisoformat(end)
    if start > end: raise ValueError('Start date is after end date')
    url = ("https://www.bankofcanada.ca/stats/results/csv?lookupPage=lookup_yield_curve.php"
           f"&dFrom={start}&dTo={end}")
    key = 'boc_' + hashlib.sha256((url+str(target)).encode()).hexdigest()[:16]
    report = acquire_one({'id':key,'url':url,'path':target.name,'kind':'boc_csv'}, target.parent)
    atomic_new(target.with_suffix(target.suffix+'.source.json'), json.dumps(report,indent=2).encode('utf-8'))
    return target


def prepare(raw_path: Path = RAW, out_path: Path = OUT) -> tuple[list[dict], dict]:
    raw_path, out_path = Path(raw_path), Path(out_path)
    if raw_path.resolve() == out_path.resolve():
        raise ValueError("Raw input and output must not be the same path")
    with raw_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Empty BoC source file")
        headers = {str(name).strip(): name for name in reader.fieldnames}
        if "Date" not in headers:
            raise ValueError("BoC source is missing its Date column")
        required = [COLUMNS[t] for t in MATURITIES]
        missing_columns = [c for c in required if c not in headers]
        if missing_columns:
            raise ValueError(f"BoC source is missing maturity fields: {missing_columns}")
        daily = []
        seen_dates = set()
        total_rows = 0
        missing_rows = 0
        for raw in reader:
            total_rows += 1
            date_text = (raw.get(headers["Date"]) or "").strip()
            try:
                day = date.fromisoformat(date_text[:10])
            except ValueError as exc:
                raise ValueError(f"Unparseable source date {date_text!r}") from exc
            if day in seen_dates:
                raise ValueError(f"Duplicate source date {day.isoformat()}")
            seen_dates.add(day)
            values = []
            complete = True
            for name in required:
                text = (raw.get(headers[name]) or "").strip()
                try:
                    value = float(text)
                except ValueError:
                    complete = False
                    break
                if not math.isfinite(value):
                    complete = False
                    break
                values.append(value)
            if complete:
                daily.append((day, values))
            else:
                missing_rows += 1
    if not daily:
        raise ValueError("No complete eight-maturity yield curves")
    daily.sort(key=lambda row: row[0])
    # Use the last available complete trading observation in each ISO week.
    weekly_by_key = {}
    for day, values in daily:
        iso = day.isocalendar()
        weekly_by_key[(iso.year, iso.week)] = (day, values)
    weekly = sorted(weekly_by_key.values(), key=lambda row: row[0])
    result = []
    for day, values in weekly:
        row = {"Date": day.isoformat()}
        row.update({f"y{tau:g}": value * 100.0 for tau, value in zip(MATURITIES, values)})
        result.append(row)
    if len({r["Date"] for r in result}) != len(result):
        raise AssertionError("Weekly dates are not unique")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["Date", *[f"y{tau:g}" for tau in MATURITIES]]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result)
    stat = raw_path.stat()
    manifest = {
        "source": "Bank of Canada daily zero-coupon Government of Canada yield curves",
        "source_page": SOURCE_PAGE,
        "download_endpoint_template": "https://www.bankofcanada.ca/stats/results/csv?lookupPage=lookup_yield_curve.php&dFrom={start}&dTo={end}",
        "retrieval_evidence": "Existing raw file supplied in the project; filesystem modified time used as a retrieval proxy, not an independently authenticated server timestamp.",
        "raw_file": path_label(raw_path),
        "raw_sha256": sha256(raw_path),
        "raw_filesystem_mtime_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "processing_run_utc": datetime.now(timezone.utc).isoformat(),
        "daily_source_rows": total_rows,
        "complete_daily_rows_eight_maturities": len(daily),
        "rows_missing_selected_maturities": missing_rows,
        "weekly_rows": len(result),
        "date_first": result[0]["Date"],
        "date_last": result[-1]["Date"],
        "maturities_years": MATURITIES,
        "raw_units": "decimal yields",
        "processed_units": "percentage points (decimal yield multiplied by 100)",
        "sampling": "last available complete observation in each ISO calendar week; not interpolated or averaged",
        "missingness": "daily rows missing any selected maturity are excluded before weekly sampling",
        "processed_file": path_label(out_path),
        "processed_sha256": sha256(out_path),
    }
    transaction_path = raw_path.with_suffix(raw_path.suffix+'.source.json')
    if transaction_path.exists():
        transaction = json.loads(transaction_path.read_text(encoding='utf-8'))
        if transaction.get('sha256') != manifest['raw_sha256']:
            raise ValueError('Raw acquisition sidecar checksum mismatch')
        manifest['retrieval_evidence'] = 'Authenticated download metadata with matching SHA-256'
        manifest['download_transaction'] = transaction
    manifest_path = out_path.parent / "source_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
    raw_path = download_raw(args.start, args.end, args.force, args.raw) if args.download else args.raw
    frame, manifest = prepare(raw_path, args.out)
    print(json.dumps({"rows": len(frame), "first": manifest["date_first"],
                      "last": manifest["date_last"], "missing_rows": manifest["rows_missing_selected_maturities"],
                      "manifest": str(args.out.parent / "source_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
