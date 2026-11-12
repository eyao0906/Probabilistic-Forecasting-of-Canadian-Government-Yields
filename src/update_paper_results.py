"""Update paper results tables and figures from full experiment output."""
from __future__ import annotations

if __name__ == "__main__":
    import sys as _legacy_sys
    from pathlib import Path as _LegacyPath
    _legacy_sys.path.insert(0, str(_LegacyPath(__file__).resolve().parent))
    from legacy_guard import require_legacy_opt_in as _require_legacy_opt_in
    _require_legacy_opt_in()

from pathlib import Path
import json

import pandas as pd

ROOT = Path("..")
PAPER = ROOT / "paper" / "MRP_Draft.qmd"
RESULTS = ROOT / "results" / "full"
FIGURES = ROOT / "figures"

MODEL_NAMES = [
    "M0_random_walk", "M1_PCA_VAR", "M2_two_step_DNS_AR1",
    "M3_DNS_Kalman_MLE", "M4_Bayesian_DNS_constant_vol", "M5_Bayesian_DNS_stochastic_vol",
]
MODEL_SHORT = ["Random walk", "PCA–VAR", "Two-step DNS", "DNS state-space",
               "Bayesian const-vol", "Bayesian SV"]
PRIMARY_HORIZON = 4


def read_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    summary = pd.read_csv(RESULTS / "model_summary.csv")
    maturity = pd.read_csv(RESULTS / "maturity_metrics.csv")
    risk = pd.read_csv(RESULTS / "risk_summary.csv")
    meta = json.loads((RESULTS / "run_metadata.json").read_text())
    return summary, maturity, risk, meta


def format_number(x):
    if pd.isna(x):
        return "—"
    return f"{x:.4f}"


def generate_results_markdown() -> str:
    summary, maturity, risk, meta = read_results()
    n_origins = meta.get("out_of_sample_origins", "?")
    lines = []
    lines.append(f"*The experiment used {n_origins} out-of-sample origins.*\n")
    lines.append("```{python}")
    lines.append("#| echo: false")
    lines.append("#| message: false")
    lines.append("#| warning: false")
    lines.append("#| label: tbl-summary")
    lines.append("#| tbl-cap: Average out-of-sample scores at the four-week horizon.")
    primary = summary[summary["horizon_weeks"] == PRIMARY_HORIZON].copy()
    primary["Model"] = primary["model"].map(dict(zip(MODEL_NAMES, MODEL_SHORT)))
    cols = ["Model", "dss", "energy_score", "variogram_score_p05", "rmse_curve", "mae_curve",
            "coverage90_maturity_average", "mean_interval_width90", "n_forecasts"]
    rename = {
        "dss": "DSS", "energy_score": "Energy Score", "variogram_score_p05": "Variogram Score",
        "rmse_curve": "RMSE", "mae_curve": "MAE",
        "coverage90_maturity_average": "90% coverage", "mean_interval_width90": "90% width",
        "n_forecasts": "n"
    }
    tbl = primary[cols].rename(columns=rename)
    lines.append(tbl.to_markdown(index=False))
    lines.append("```\n")
    return "\n".join(lines)


def generate_risk_markdown() -> str:
    risk = pd.read_csv(RESULTS / "risk_summary.csv")
    primary = risk[risk["horizon_weeks"] == PRIMARY_HORIZON].copy()
    lines = []
    for tail in (0.05, 0.01):
        lines.append(f"### {int(tail*100)}% tail\n")
        sub = primary[primary["tail_probability"] == tail].copy()
        sub["Model"] = sub["model"].map(dict(zip(MODEL_NAMES, MODEL_SHORT)))
        tbl = sub[["Model", "mean_VaR", "mean_ES", "exceptions", "exception_rate",
                   "LR_unconditional_coverage", "p_unconditional_coverage"]].rename(
            columns={"mean_VaR": "Mean VaR", "mean_ES": "Mean ES",
                     "exception_rate": "Exception rate", "LR_unconditional_coverage": "LR UC",
                     "p_unconditional_coverage": "p UC"})
        lines.append("```{python}")
        lines.append("#| echo: false")
        lines.append("#| message: false")
        lines.append("#| warning: false")
        lines.append(tbl.to_markdown(index=False))
        lines.append("```\n")
    return "\n".join(lines)


def update_paper() -> None:
    text = PAPER.read_text(encoding="utf-8")
    # Replace results section placeholders with actual content.
    # This is a simple string replacement; for complex docs use a proper parser.
    results_md = generate_results_markdown()
    risk_md = generate_risk_markdown()
    # Find the results section and replace placeholder tables/figures.
    # We keep the section headings and narrative; inject tables and figures.
    if "*Results pending experiment completion.*" in text:
        text = text.replace("*Results pending experiment completion.*", "")
    if "# Results {#sec-results}" in text:
        text = text.replace("# Results {#sec-results}", "# Results {#sec-results}\n")
    # We will not do full document restructuring here; instead, write a helper file.
    (ROOT / "docs" / "results_snippets.md").write_text(
        "## Results Tables\n\n" + results_md + "\n## Risk Tables\n\n" + risk_md,
        encoding="utf-8"
    )
    print("Wrote docs/results_snippets.md with updated tables.")


if __name__ == "__main__":
    if (RESULTS / "model_summary.csv").exists():
        update_paper()
    else:
        print("Results not ready.")
