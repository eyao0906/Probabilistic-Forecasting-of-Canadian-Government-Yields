"""Render saved screening traces and calculate per-chain lag-one correlations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "hautsch_yang_state_space_20261117"
OUTPUT = RESULTS / "trace_diagnostics_20261117"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = {}
    for folder in ("m3_conditional_screen_20261117", "sv_conditional_interwoven_20261117"):
        for path in sorted((RESULTS / folder).glob("*_traces.npz")):
            stage = path.name.removesuffix("_traces.npz")
            summary_path = path.with_name(stage + ".json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["trace_sha256"] != sha256(path):
                raise ValueError(f"trace checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                traces = {
                    name: archive[name].copy()
                    for name in archive.files
                    if name in summary["diagnostics"]["rank_rhat"] or name == "h_selected"
                }
            traces = {
                name: value.reshape(value.shape[0], value.shape[1], -1)
                for name, value in traces.items()
            }
            if not traces or any(
                value.shape[0] != 4 or value.shape[1] < 4 or not np.isfinite(value).all()
                for value in traces.values()
            ):
                raise ValueError(f"invalid or empty traces: {path}")
            panels = [
                (name, index, value[:, :, index])
                for name, value in traces.items() for index in range(value.shape[2])
            ]
            rows = math.ceil(len(panels) / 3)
            figure, axes = plt.subplots(rows, 3, figsize=(12, 2.1 * rows), squeeze=False)
            stage_records = {}
            for axis, (name, index, values) in zip(axes.flat, panels):
                for chain, series in enumerate(values):
                    axis.plot(series, linewidth=.45, alpha=.7, label=str(chain + 1))
                axis.set_title(f"{name}[{index}]", fontsize=8)
                axis.tick_params(labelsize=6)
                lag_one = []
                for series in values:
                    if np.std(series[:-1]) == 0 or np.std(series[1:]) == 0:
                        lag_one.append(None)
                    else:
                        lag_one.append(float(np.corrcoef(series[:-1], series[1:])[0, 1]))
                entry = {"lag1_autocorrelation_by_chain": lag_one}
                if name in {"factor_persistence", "phi"}:
                    entry["fraction_within_1_percent_of_bounds"] = float(np.mean(np.abs(values) > .98))
                elif name == "decay":
                    entry["fraction_within_1_percent_of_bounds"] = float(
                        np.mean((values < .0695) | (values > 1.9805))
                    )
                elif name == "variance" and folder.startswith("sv_"):
                    entry["fraction_within_1_percent_of_bounds"] = float(
                        np.mean((values < .04) | (values > 3.96))
                    )
                stage_records[f"{name}[{index}]"] = entry
            for axis in list(axes.flat)[len(panels):]:
                axis.set_visible(False)
            figure.suptitle(f"{folder}: {stage} (chains 1–4)", fontsize=10)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            figure.legend(handles, labels, title="Chain", loc="upper right", ncol=4, fontsize=6)
            figure.tight_layout()
            figure.savefig(OUTPUT / f"{folder}_{stage}.png", dpi=110)
            plt.close(figure)
            records[f"{folder}/{stage}"] = {
                "trace_sha256": sha256(path), "parameters": stage_records
            }
    if len(records) != 8:
        raise ValueError(f"expected eight staged trace files, found {len(records)}")
    result = {
        "script_sha256": sha256(Path(__file__)),
        "boundary_definition": "within 1% of declared finite parameter support",
        "stages": records,
    }
    (OUTPUT / "trace_diagnostics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Rendered {len(records)} trace plots with per-chain lag-one diagnostics.")


if __name__ == "__main__":
    main()
