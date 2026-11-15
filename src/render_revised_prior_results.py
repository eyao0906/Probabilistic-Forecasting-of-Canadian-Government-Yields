"""Render paper tables and figures from the accepted revised-prior experiment."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "bayesian_prior_revised_20261114"
BASELINES = ROOT / "results" / "validated_baselines"
GENERATED = ROOT / "paper" / "generated" / "revised_prior"
FIGURES = ROOT / "figures" / "finite" / "revised_prior"
FACTORS = ("level", "slope", "curvature")
QUANTILES = (0.05, 0.5, 0.95)
MODEL_LABELS = {"M3": "constant volatility", "M4": "stochastic volatility"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_accepted_inputs() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    run_summary_path = RUN / "summary.json"
    plan_path = RUN / "plan.json"
    analysis_path = RUN / "analysis" / "summary.json"
    if not all(path.is_file() for path in (run_summary_path, plan_path, analysis_path)):
        raise FileNotFoundError("The complete revised-prior rolling analysis is required")
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if (not run_summary.get("accepted") or run_summary.get("completed_origins") != 130
        or digest(plan_path) != run_summary.get("plan_sha256")
        or not analysis.get("accepted_main_analysis") or analysis.get("n_origins") != 130):
        raise ValueError("Revised-prior results have not passed the complete archive audit")
    baseline_metadata = json.loads(
        (BASELINES / "metadata.json").read_text(encoding="utf-8")
    )
    if baseline_metadata.get("input_sha256") != plan.get("input_sha256"):
        raise ValueError("Classical baselines do not match the revised-prior experiment input")
    baseline_verification = json.loads(
        (BASELINES / "verification.json").read_text(encoding="utf-8")
    )
    if baseline_verification.get("n_origins") != 130 or baseline_verification.get("score_rows") != 780:
        raise ValueError("Classical benchmark verification is incomplete")
    baseline_summary = pd.read_csv(BASELINES / "summary.csv")
    if set(baseline_summary["model"]) != {"M0", "M1", "M2"} or set(baseline_summary["horizon"]) != {1, 4}:
        raise ValueError("Classical benchmark summary does not contain the accepted model set")
    paired = pd.read_csv(RUN / "analysis" / "paired_summary.csv")
    maturity = pd.read_csv(RUN / "analysis" / "per_maturity.csv")
    if set(paired["horizon"]) != {1, 4} or set(maturity["model"]) != {"M3", "M4"}:
        raise ValueError("Unexpected revised-prior analysis table shape")
    return run_summary, plan, paired, maturity


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_joint_tables(maturity: pd.DataFrame) -> None:
    baseline = pd.read_csv(BASELINES / "summary.csv")
    for horizon in (1, 4):
        rows = []
        for _, row in baseline[baseline["horizon"] == horizon].iterrows():
            rows.append({
                "model": row["model"],
                "dss": row["dss"],
                "energy": row["energy_score"],
                "rmse": 100 * row["pooled_rmse"],
                "cover": 100 * row["coverage90"],
                "width": 100 * row["width90"],
            })
        analysis = json.loads((RUN / "analysis" / "summary.json").read_text(encoding="utf-8"))
        metrics = analysis["horizons"][str(horizon)]["main"]["metrics"]
        rows_by_model = {row["model"]: row for row in rows}
        for model, internal in (("M3", "M4"), ("M4", "M5")):
            width = maturity[
                (maturity["horizon"] == horizon) & (maturity["model"] == model)
            ]["mean_interval_width_90"].mean()
            rows_by_model[model] = {
                "model": model,
                "dss": metrics["dss"][internal],
                "energy": metrics["energy_score"][internal],
                "rmse": 100 * metrics["rmse"][internal],
                "cover": 100 * metrics["coverage_90"][internal],
                "width": 100 * width,
            }
        rows = [rows_by_model[f"M{i}"] for i in range(5)]
        lines = [
            "| Model | DSS | Energy | RMSE (bp) | Cover (%) | Width (bp) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        lines.extend(
            f"| {row['model']} | {fmt(row['dss'])} | {fmt(row['energy'])} | "
            f"{fmt(row['rmse'])} | {fmt(row['cover'])} | {fmt(row['width'])} |"
            for row in rows
        )
        label = "One-week" if horizon == 1 else "Four-week"
        lines.extend([
            "",
            f": {label} forecasts at 130 common origins under the revised training-calibrated priors. "
            "M3 is Bayesian DNS with constant factor volatility; M4 adds stochastic factor volatility. "
            "Lower DSS, Energy Score and RMSE are better; marginal coverage targets 90%. "
            "Width is mean central-90% marginal interval width. One basis point equals 0.01 percentage points. "
            f"{{#tbl-joint-h{horizon}}}",
            "",
        ])
        (GENERATED / f"presentation_joint_h{horizon}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )


def write_paired_table(paired: pd.DataFrame) -> None:
    metric_names = {
        "dss": ("DSS", 1.0, "score"),
        "energy_score": ("Energy", 1.0, "score"),
        "rmse": ("RMSE (bp)", 100.0, "bp"),
        "coverage_90": ("Coverage (pp)", 100.0, "pp"),
    }
    lines = [
        "| Horizon | Metric | M4 minus M3 | 95% interval |",
        "|---|---|---:|---|",
    ]
    for horizon in (1, 4):
        for metric, (label, multiplier, _) in metric_names.items():
            row = paired[(paired["horizon"] == horizon) & (paired["metric"] == metric)]
            if len(row) != 1:
                raise ValueError(f"Expected one paired result for {horizon}-week {metric}")
            item = row.iloc[0]
            difference = multiplier * item["M4_minus_M3"]
            lower = multiplier * item["ci95_lower"]
            upper = multiplier * item["ci95_upper"]
            lines.append(
                f"| {horizon} | {label} | {fmt(difference, 4)} | "
                f"[{fmt(lower, 4)}, {fmt(upper, 4)}] |"
            )
    lines.extend([
        "",
        ": Paired M4-minus-M3 stationary-bootstrap 95% percentile intervals under the revised priors: "
        "expected block length eight origins, 9,999 replicates, seed 271026. "
        "Coverage differences are percentage points (pp). {#tbl-finite-paired}",
        "",
    ])
    (GENERATED / "presentation_finite_paired.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_block_table() -> None:
    analysis = json.loads((RUN / "analysis" / "summary.json").read_text(encoding="utf-8"))
    lines = [
        "| Horizon | Loss | L=4 | L=8 | L=12 |",
        "|---|---|---|---|---|",
    ]
    for horizon in (1, 4):
        for metric, label in (("dss", "DSS"), ("energy_score", "Energy")):
            cells = []
            for length in (4, 8, 12):
                horizon_result = analysis["horizons"][str(horizon)]
                result = (horizon_result["main"] if length == 8 else
                          horizon_result["dependence_sensitivity"][str(length)])
                lower, upper = result["metrics"][metric]["ci95"]
                cells.append(f"[{fmt(lower, 4)}, {fmt(upper, 4)}]")
            lines.append(f"| {horizon} | {label} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        ": Dependence sensitivity: paired M4-minus-M3 score intervals under expected stationary-bootstrap "
        "lengths 4, 8 and 12 origins. {#tbl-finite-blocks}",
        "",
    ])
    (GENERATED / "presentation_finite_blocks.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_maturity_table() -> None:
    classical = pd.read_csv(BASELINES / "maturity_by_origin.csv")
    classical = classical[classical["horizon"] == 4]
    grouped = classical.groupby(["model", "maturity"], as_index=False).agg(
        rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
        coverage=("coverage90", "mean"),
    )
    rows = [
        {
            "model": row["model"],
            "maturity": row["maturity"],
            "rmse": row["rmse"],
            "coverage": row["coverage"],
        }
        for _, row in grouped.iterrows()
    ]
    revised = pd.read_csv(RUN / "analysis" / "per_maturity.csv")
    revised = revised[revised["horizon"] == 4]
    for _, row in revised.iterrows():
        rows.append({
            "model": row["model"],
            "maturity": row["maturity_years"],
            "rmse": row["rmse"],
            "coverage": row["coverage_90"],
        })
    rows.sort(key=lambda row: (int(row["model"][1:]), row["maturity"]))
    lines = [
        "| Model | Maturity | RMSE | Coverage |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['model']} | {row['maturity']:.3f} | {row['rmse']:.3f} | {row['coverage']:.3f} |"
        for row in rows
    )
    lines.extend([
        "",
        ": Four-week point accuracy and marginal 90% interval coverage under the revised priors. "
        "{#tbl-maturity}",
        "",
    ])
    (GENERATED / "maturity_h4.md").write_text("\n".join(lines), encoding="utf-8")


def load_factor_draws(chain_dir: Path, model: str, factor: str) -> tuple[np.ndarray, list[str]]:
    pooled = []
    names = None
    for chain in range(4):
        path = chain_dir / f"{model}_{factor}_chain{chain}.npz"
        with np.load(path, allow_pickle=False) as archive:
            report = json.loads(str(archive["report"]))
            draws = archive["draws"].copy()
        if names is not None and report["names"] != names:
            raise ValueError(f"Inconsistent parameter order in {path}")
        names = report["names"]
        if draws.ndim != 2 or not np.isfinite(draws).all():
            raise ValueError(f"Invalid posterior draws in {path}")
        pooled.append(draws)
    if names is None:
        raise ValueError("No posterior chains were found")
    return np.concatenate(pooled, axis=0), names


def write_posterior_parameter_table(plan: dict) -> None:
    final_date = plan["origins"][-1]["date"]
    chain_dir = RUN / "origins" / final_date / "chains"
    rows = []
    for factor in FACTORS:
        draws, names = load_factor_draws(chain_dir, "M5", factor)
        for parameter in ("b", "long_run_mean", "m", "phi", "v"):
            if parameter not in names:
                raise ValueError(f"Missing {parameter} in revised posterior draws")
            values = draws[:, names.index(parameter)]
            q05, median, q95 = np.quantile(values, QUANTILES)
            rows.append({
                "origin": final_date,
                "factor": factor,
                "parameter": parameter,
                "posterior_draws": len(values),
                "q05": float(q05),
                "median": float(median),
                "q95": float(q95),
            })
    path = GENERATED / "posterior_parameter_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Factor | $b_j$ | Long-run mean $\\mu_j$ | $m_j$ | $\\phi_j$ | $v_j$ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for factor in FACTORS:
        by_parameter = {row["parameter"]: row for row in rows if row["factor"] == factor}
        cells = [factor.capitalize()]
        for parameter in ("b", "long_run_mean", "m", "phi", "v"):
            row = by_parameter[parameter]
            cells.append(f"{row['median']:.3f} [{row['q05']:.3f}, {row['q95']:.3f}]")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        f": Posterior medians and equal-tailed 90% intervals for the accepted revised-prior "
        f"stochastic-volatility fit at {final_date}; four chains pooled. "
        "{#tbl-posterior-parameters}",
        "",
    ])
    (GENERATED / "posterior_parameter_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_volatility_figure(plan: dict, run_summary: dict) -> None:
    rows = []
    for origin in plan["origins"]:
        origin_date = origin["date"]
        forecast_path = RUN / "origins" / origin_date / "forecast.json"
        expected_hash = run_summary["forecasts_sha256"].get(origin_date)
        if not forecast_path.is_file() or digest(forecast_path) != expected_hash:
            raise ValueError(f"Changed forecast record at {origin_date}")
        forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
        if not forecast.get("accepted") or forecast["specification"]["origin"] != origin_date:
            raise ValueError(f"Unaccepted forecast record at {origin_date}")
        chain_dir = RUN / "origins" / origin_date / "chains"
        if digest(chain_dir / "summary.json") != forecast["parameter_summary_sha256"]:
            raise ValueError(f"Changed posterior summary at {origin_date}")
        checkpoints = forecast["chain_checkpoint_sha256"]
        for factor in FACTORS:
            constant, constant_names = load_factor_draws(chain_dir, "M4", factor)
            stochastic, stochastic_names = load_factor_draws(chain_dir, "M5", factor)
            for model in ("M4", "M5"):
                for chain in range(4):
                    path = chain_dir / f"{model}_{factor}_chain{chain}.npz"
                    if checkpoints.get(str(path.resolve())) != digest(path):
                        raise ValueError(f"Changed posterior checkpoint: {path}")
            if "log_variance" not in constant_names or "h_last" not in stochastic_names:
                raise ValueError(f"Missing volatility state for {factor} at {origin_date}")
            constant_sd = np.exp(0.5 * constant[:, constant_names.index("log_variance")]) * 100
            stochastic_sd = np.exp(0.5 * stochastic[:, stochastic_names.index("h_last")]) * 100
            lower, median, upper = np.quantile(stochastic_sd, QUANTILES)
            rows.append({
                "origin": origin_date,
                "factor": factor,
                "m4_lower_90_bp": lower,
                "m4_median_bp": median,
                "m4_upper_90_bp": upper,
                "m3_constant_median_bp": float(np.median(constant_sd)),
            })
    data = pd.DataFrame(rows)
    data.to_csv(FIGURES / "estimated_factor_volatility.csv", index=False)
    colors = {"M3": "#c47732", "M4": "#315f78"}
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 3.8), sharex=True)
    for ax, factor in zip(axes, FACTORS):
        series = data[data["factor"] == factor].sort_values("origin")
        dates = pd.to_datetime(series["origin"])
        legend_labels = factor == FACTORS[0]
        ax.fill_between(
            dates, series["m4_lower_90_bp"], series["m4_upper_90_bp"],
            color=colors["M4"], alpha=0.18,
            label="M4 90% posterior interval" if legend_labels else "_nolegend_",
        )
        ax.plot(
            dates, series["m4_median_bp"], color=colors["M4"], linewidth=1.35,
            label="M4 posterior median" if legend_labels else "_nolegend_",
        )
        ax.plot(
            dates, series["m3_constant_median_bp"], color=colors["M3"],
            linestyle="--", linewidth=1.15,
            label="M3 posterior median (constant)" if legend_labels else "_nolegend_",
        )
        ax.set_ylabel("SD (bp)")
        ax.set_title(f"{factor.capitalize()} factor", loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    axes[-1].set_xlabel("Forecast origin")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Posterior weekly factor innovation volatility", y=1.02, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "estimated_factor_volatility.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_score_figures(paired: pd.DataFrame, maturity: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    colors = {1: "#1f77b4", 4: "#d95f02"}
    for ax, metric, title in zip(
        axes, ("dss", "energy_score"), ("Dawid–Sebastiani score", "Energy Score")
    ):
        subset = paired[paired["metric"] == metric].sort_values("horizon")
        for index, (_, row) in enumerate(subset.iterrows()):
            estimate = row["M4_minus_M3"]
            lower, upper = row["ci95_lower"], row["ci95_upper"]
            ax.errorbar(
                estimate, index,
                xerr=[[estimate - lower], [upper - estimate]],
                fmt="o", color=colors[int(row["horizon"])], capsize=3, linewidth=1.4,
            )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(range(len(subset)), [str(value) for value in subset["horizon"]])
        ax.set_ylabel("Horizon (weeks)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.22)
        ax.set_xlabel("M4 − M3 (lower favors M4)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "paired_scores.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    subset = maturity[maturity["horizon"] == 4]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    colors = {"M3": "#1f77b4", "M4": "#ff7f0e"}
    for model, group in subset.groupby("model", sort=True):
        group = group.sort_values("maturity_years")
        axes[0].plot(
            group["maturity_years"], group["rmse"] * 100, "-o",
            color=colors[model], label=f"{model} ({MODEL_LABELS[model]})",
        )
        axes[1].plot(
            group["maturity_years"], group["coverage_90"] * 100, "-o",
            color=colors[model], label=f"{model} ({MODEL_LABELS[model]})",
        )
    axes[1].axhline(90, color="black", linestyle=":", linewidth=0.8)
    axes[0].set(xlabel="Maturity (years)", ylabel="RMSE (basis points)")
    axes[1].set(xlabel="Maturity (years)", ylabel="Central-90% coverage (%)")
    ticks = [0.25, 0.5, 1, 2, 5, 7, 10, 30]
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ticks, [str(value) for value in ticks])
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "maturity_h4.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    run_summary, plan, paired, maturity = require_accepted_inputs()
    write_joint_tables(maturity)
    write_paired_table(paired)
    write_block_table()
    write_maturity_table()
    write_posterior_parameter_table(plan)
    write_volatility_figure(plan, run_summary)
    write_score_figures(paired, maturity)
    provenance = {
        "rolling_summary_sha256": digest(RUN / "summary.json"),
        "plan_sha256": digest(RUN / "plan.json"),
        "analysis_summary_sha256": digest(RUN / "analysis" / "summary.json"),
        "classical_baseline_summary_sha256": digest(BASELINES / "summary.csv"),
        "classical_baseline_verification_sha256": digest(BASELINES / "verification.json"),
        "renderer_sha256": digest(Path(__file__)),
        "n_origins": 130,
        "last_origin": plan["origins"][-1]["date"],
        "status": "generated from accepted revised-prior Bayesian results and verified M0-M2 baselines",
    }
    (GENERATED / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Wrote revised-prior paper outputs to {GENERATED} and {FIGURES}")


if __name__ == "__main__":
    main()
