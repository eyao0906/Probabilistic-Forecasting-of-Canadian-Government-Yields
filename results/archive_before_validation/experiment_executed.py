"""Rolling out-of-sample experiments for models 0--5 in the approved proposal."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import (  # noqa: E402
    ar1_fit, bayes_constant_scenarios, dns_residual_variance, fit_lambda,
    fit_sv_chain, fit_state_space, forecast_sv_from_chains, gaussian_dns_forecast,
    ns_loadings, pca_var_forecast, random_walk_forecast, rhat,
    effective_sample_size, state_space_forecast,
)
from src.scores import (  # noqa: E402
    dss, energy_score, hac_dm, model_confidence_set, point_metrics,
    var_backtests, variogram_score,
)

MODEL_NAMES = ["M0_random_walk", "M1_PCA_VAR", "M2_two_step_DNS_AR1",
               "M3_DNS_Kalman_MLE", "M4_Bayesian_DNS_constant_vol",
               "M5_Bayesian_DNS_stochastic_vol"]
MATURITIES = np.array([0.25, 0.5, 1, 2, 5, 7, 10, 30], dtype=float)
YIELD_COLUMNS = [f"y{tau:g}" for tau in MATURITIES]


def read_panel(path: Path) -> tuple[list[date], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not set(["Date", *YIELD_COLUMNS]).issubset(reader.fieldnames):
            raise ValueError(f"Expected Date and yield columns {YIELD_COLUMNS} in {path}")
        rows = list(reader)
    dates = [date.fromisoformat(row["Date"]) for row in rows]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError("Weekly dates must be unique and ascending")
    y = np.array([[float(row[c]) for c in YIELD_COLUMNS] for row in rows], dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Yield panel contains non-finite values")
    return dates, y


def select_origins(dates: list[date], oos_start: date, oos_end: date,
                   horizons: list[int], stride: int) -> list[tuple[int, int, int]]:
    """Return (origin, horizon, target) indices without targets outside the OOS window."""
    oos = [i for i, day in enumerate(dates) if oos_start <= day <= oos_end]
    selected = []
    for t in oos[::stride]:
        targets = [t + h for h in horizons]
        eligible = all(target < len(dates) and oos_start <= dates[target] <= oos_end
                       for target in targets)
        if eligible:
            selected.extend((t, h, t + h) for h in horizons)
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iso_time() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_forecast_archive(path: Path, **arrays) -> None:
    safe = dict(arrays)
    if "target_dates" in safe:
        safe["target_dates"] = np.asarray(safe["target_dates"], dtype="U10")
    np.savez_compressed(path, **safe)


def estimate_portfolio_loss(yield_scenarios: np.ndarray, current_yield: np.ndarray) -> np.ndarray:
    """First-order loss on an equal-weight synthetic long zero-coupon DV01 basket."""
    weights = np.full(len(MATURITIES), 1.0 / len(MATURITIES))
    duration = MATURITIES
    delta_decimal = (yield_scenarios - current_yield[None, :]) / 100.0
    pnl = -np.sum(delta_decimal * (weights * duration)[None, :], axis=1)
    return -pnl


def diagnostic_sv(chains, factor_names: list[str], origin_date: date) -> list[dict]:
    output = []
    names = ["a", "b", "h_intercept", "phi", "sigma2_h"]
    for factor, pair in zip(factor_names, chains):
        values = np.stack([
            np.column_stack((c.beta[:, 0], c.beta[:, 1], c.intercept_h, c.phi, c.sigma2_h))
            for c in pair
        ])
        rh = rhat(values)
        ess = effective_sample_size(values)
        for j, parameter in enumerate(names):
            output.append({"origin_date": origin_date.isoformat(), "factor": factor,
                           "parameter": parameter, "rhat": float(rh[j]), "ess": float(ess[j]),
                           "chain1_acceptance_phi": pair[0].acceptance_phi,
                           "chain2_acceptance_phi": pair[1].acceptance_phi,
                           "diagnostic_draws_per_chain": len(pair[0].phi)})
    return output


def run(pilot: bool = False, max_origins: int | None = None) -> dict:
    config = json.loads((ROOT / "config" / "experiment.json").read_text(encoding="utf-8"))
    data_path = ROOT / "data" / "processed" / "canada_weekly.csv"
    dates, yields_all = read_panel(data_path)
    training_start = date.fromisoformat(config["training_start"])
    training_end = date.fromisoformat(config["training_end"])
    oos_start = date.fromisoformat(config["oos_start"])
    oos_end = date.fromisoformat(config["oos_end"])
    horizons = [int(v) for v in config["horizons_weeks"]]
    origins = select_origins(dates, oos_start, oos_end, horizons,
                             int(config["origin_stride_observations"]))
    origin_indices = sorted(set(t for t, _, _ in origins))
    if max_origins is not None:
        origin_indices = origin_indices[:max_origins]
        origins = [v for v in origins if v[0] in origin_indices]
    if pilot:
        warmup, mcmc_draws, energy_draws, risk_draws = 25, 48, 64, 256
        mcs_reps, diagnostic_draws, diagnostic_warmup = 39, 80, 40
        state_space_maxiter = 3
    else:
        warmup = int(config["mcmc_warmup"])
        mcmc_draws = int(config["mcmc_draws"])
        energy_draws = int(config["energy_score_draws"])
        risk_draws = 2048
        mcs_reps = int(config["mcs_bootstrap_replicates"])
        diagnostic_draws = int(config["mcmc_diagnostic_draws"])
        diagnostic_warmup = int(config["mcmc_diagnostic_warmup"])
        state_space_maxiter = int(config.get("state_space_maxiter", 50))
    draws = min(int(config["posterior_predictive_draws"]), energy_draws)
    rng = np.random.default_rng(int(config["seed"]))
    initial_indices = [i for i, day in enumerate(dates) if training_start <= day <= training_end]
    if not initial_indices:
        raise ValueError("No observations in initial training window")
    y_initial = yields_all[initial_indices]
    grid_spec = config["lambda_grid_per_year"]
    lambda_grid = np.linspace(float(grid_spec[0]), float(grid_spec[1]), int(grid_spec[2]))
    lam, initial_factors, lambda_sse = fit_lambda(y_initial, MATURITIES, lambda_grid)
    load = ns_loadings(MATURITIES, lam)
    first_training_index = min(i for i, day in enumerate(dates) if day >= training_start)
    output_dir = ROOT / "results" / ("pilot" if pilot else "full")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / "experiment.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log_path.open("a", encoding="utf-8").write(
        f"[{iso_time()}] START {'pilot' if pilot else 'full'} origins={len(origin_indices)} lambda={lam:.6f}\n")

    horizon_index = {h: j for j, h in enumerate(horizons)}
    origin_index = {t: i for i, t in enumerate(origin_indices)}
    n_origins, n_horizons, n_models, n_maturities = len(origin_indices), len(horizons), len(MODEL_NAMES), 8
    scen_store = np.full((n_origins, n_horizons, n_models, draws, n_maturities), np.nan)
    mean_store = np.full((n_origins, n_horizons, n_models, n_maturities), np.nan)
    cov_store = np.full((n_origins, n_horizons, n_models, n_maturities, n_maturities), np.nan)
    target_store = np.full((n_origins, n_horizons, n_maturities), np.nan)
    target_dates = np.empty((n_origins, n_horizons), dtype=object)
    score_rows, risk_rows, fit_rows, diag_rows = [], [], [], []
    previous_state_params = None
    diagnostic_origin = max(origin_indices) if origin_indices else None

    for count, t in enumerate(origin_indices, start=1):
        training = yields_all[first_training_index:t + 1]
        if dates[t] > oos_end or dates[t] < oos_start:
            raise AssertionError("Forecast origin outside out-of-sample window")
        factors = np.linalg.lstsq(load, training.T, rcond=None)[0].T
        r_diag = dns_residual_variance(training, load, float(config["variance_floor_percent_squared"]))
        factor_names = ["level", "slope", "curvature"]

        # Gaussian DNS state-space benchmark is re-estimated on the expanding sample.
        params_start = previous_state_params
        if params_start is None:
            from src.models import state_space_start
            params_start = state_space_start(training, load)
        state_params, fit_info = fit_state_space(training, load, params_start, state_space_maxiter)
        previous_state_params = state_params.copy()
        fit_info.update({"origin_date": dates[t].isoformat(), "n_training": len(training)})
        fit_rows.append(fit_info)

        # Bayesian SV posterior is refit at each selected origin; one rolling chain per factor.
        sv_chains = [fit_sv_chain(factors[:, j], np.random.default_rng(int(config["seed"]) + t * 17 + j),
                                  warmup=warmup, draws=mcmc_draws) for j in range(3)]
        for j, chain in enumerate(sv_chains):
            fit_rows.append({"origin_date": dates[t].isoformat(), "model": "M5_SV_factor",
                             "factor": factor_names[j], "acceptance_phi": chain.acceptance_phi,
                             "n_training": len(training), "posterior_draws": len(chain.phi)})

        for h in horizons:
            target = t + h
            if target >= len(dates) or dates[target] > oos_end:
                continue
            hi = horizon_index[h]
            oi = origin_index[t]
            observed = yields_all[target]
            target_store[oi, hi] = observed
            target_dates[oi, hi] = dates[target].isoformat()
            model_scenarios: dict[str, np.ndarray] = {}
            model_covariances: dict[str, np.ndarray] = {}
            # M0: random-walk mean and historical h-week change covariance.
            s0, c0 = random_walk_forecast(training, h, rng, draws,
                                          float(config["variance_floor_percent_squared"]))
            model_scenarios[MODEL_NAMES[0]], model_covariances[MODEL_NAMES[0]] = s0, c0
            # M1: three-PC VAR(1) with yield reconstruction uncertainty.
            s1, c1 = pca_var_forecast(training, h, rng, draws,
                                      float(config["variance_floor_percent_squared"]))
            model_scenarios[MODEL_NAMES[1]], model_covariances[MODEL_NAMES[1]] = s1, c1
            # M2: two-step DNS with independent Gaussian AR(1) factors.
            s2, c2 = gaussian_dns_forecast(factors, load, r_diag, h, rng, draws)
            model_scenarios[MODEL_NAMES[2]], model_covariances[MODEL_NAMES[2]] = s2, c2
            # M3: latent-state DNS, Kalman filter and rolling quasi-MLE.
            s3, c3 = state_space_forecast(training, load, state_params, h, rng, draws)
            model_scenarios[MODEL_NAMES[3]], model_covariances[MODEL_NAMES[3]] = s3, c3
            # M4: Bayesian constant-variance AR(1) factors.
            s4 = bayes_constant_scenarios(factors, load, r_diag, h, rng, draws)
            model_scenarios[MODEL_NAMES[4]] = s4
            model_covariances[MODEL_NAMES[4]] = np.cov(s4, rowvar=False, ddof=1)
            # M5: Bayesian SV factors, posterior parameter/state draws and reconstruction noise.
            s5 = forecast_sv_from_chains(factors, load, r_diag, h, rng, draws, sv_chains)
            model_scenarios[MODEL_NAMES[5]] = s5
            model_covariances[MODEL_NAMES[5]] = np.cov(s5, rowvar=False, ddof=1)

            for mi, name in enumerate(MODEL_NAMES):
                scenarios = model_scenarios[name]
                mean = scenarios.mean(axis=0)
                covariance = model_covariances[name]
                if name.startswith("M4") or name.startswith("M5"):
                    covariance = (covariance + covariance.T) / 2
                    # Floors are fixed in advance and are only numerical stabilization.
                    vals, vecs = np.linalg.eigh(covariance)
                    covariance = (vecs * np.maximum(vals, float(config["variance_floor_percent_squared"]))) @ vecs.T
                score = dss(observed, mean, covariance)
                em = energy_score(scenarios, observed)
                vm = variogram_score(scenarios, observed, power=0.5)
                pm = point_metrics(observed, mean, scenarios)
                scen_store[oi, hi, mi] = scenarios
                mean_store[oi, hi, mi] = mean
                cov_store[oi, hi, mi] = covariance
                score_rows.append({"origin_date": dates[t].isoformat(), "target_date": dates[target].isoformat(),
                                   "horizon_weeks": h, "model": name, "dss": score,
                                   "energy_score": em, "variogram_score_p05": vm,
                                   "rmse_curve": pm["rmse"], "mae_curve": pm["mae"],
                                   "coverage90_maturity_average": pm["coverage90"],
                                   "mean_interval_width90": pm["mean_width90"]})
                # Risk distribution uses larger Monte Carlo samples than the proper-score ensemble.
                if name in (MODEL_NAMES[0], MODEL_NAMES[1], MODEL_NAMES[2], MODEL_NAMES[3]):
                    risk_scenarios = rng.multivariate_normal(mean, covariance, size=risk_draws)
                elif name == MODEL_NAMES[4]:
                    risk_scenarios = bayes_constant_scenarios(factors, load, r_diag, h, rng, risk_draws)
                else:
                    risk_scenarios = forecast_sv_from_chains(factors, load, r_diag, h, rng, risk_draws, sv_chains)
                predictive_losses = estimate_portfolio_loss(risk_scenarios, training[-1])
                realized_loss = float(estimate_portfolio_loss(observed[None, :], training[-1])[0])
                var95 = float(np.quantile(predictive_losses, 0.95))
                var99 = float(np.quantile(predictive_losses, 0.99))
                es95 = float(predictive_losses[predictive_losses >= var95].mean())
                es99 = float(predictive_losses[predictive_losses >= var99].mean())
                risk_rows.append({"origin_date": dates[t].isoformat(), "target_date": dates[target].isoformat(),
                                  "horizon_weeks": h, "model": name, "realized_loss": realized_loss,
                                  "VaR95": var95, "ES95": es95, "VaR99": var99, "ES99": es99,
                                  "risk_draws": risk_draws})

        if t == diagnostic_origin and not pilot:
            diagnostic_chains = []
            for j in range(3):
                pair = [fit_sv_chain(factors[:, j], np.random.default_rng(int(config["seed"]) + 700000 + j * 31 + chain_id),
                                     warmup=diagnostic_warmup, draws=diagnostic_draws)
                        for chain_id in range(2)]
                diagnostic_chains.append(pair)
            diag_rows.extend(diagnostic_sv(diagnostic_chains, factor_names, dates[t]))
        if count % 5 == 0 or count == len(origin_indices):
            elapsed = time.perf_counter() - started
            msg = f"[{iso_time()}] progress {count}/{len(origin_indices)} elapsed_s={elapsed:.1f}"
            print(msg, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

    if not score_rows:
        raise RuntimeError("No forecasts were generated")
    write_csv(output_dir / "score_by_origin.csv", score_rows)
    write_csv(output_dir / "risk_by_origin.csv", risk_rows)
    write_csv(output_dir / "fit_diagnostics.csv", fit_rows)
    write_csv(output_dir / "mcmc_diagnostics.csv", diag_rows)
    save_forecast_archive(output_dir / "forecast_draws.npz", scenarios=scen_store, means=mean_store,
                          covariances=cov_store, targets=target_store,
                          origin_dates=np.array([dates[t].isoformat() for t in origin_indices]),
                          target_dates=target_dates, horizons=np.array(horizons),
                          model_names=np.array(MODEL_NAMES), maturities=MATURITIES)

    # Pooled performance tables and maturity-level point/calibration diagnostics.
    summary_rows, maturity_rows = [], []
    for h in horizons:
        for name in MODEL_NAMES:
            subset = [r for r in score_rows if r["horizon_weeks"] == h and r["model"] == name]
            if not subset:
                continue
            summary_rows.append({"horizon_weeks": h, "model": name, **{
                key: float(np.mean([r[key] for r in subset])) for key in
                ["dss", "energy_score", "variogram_score_p05", "rmse_curve", "mae_curve",
                 "coverage90_maturity_average", "mean_interval_width90"]}, "n_forecasts": len(subset)})
            mi = MODEL_NAMES.index(name)
            for j, tau in enumerate(MATURITIES):
                err = target_store[:, horizon_index[h], j] - mean_store[:, horizon_index[h], mi, j]
                draws_j = scen_store[:, horizon_index[h], mi, :, j]
                lo, hi = np.nanquantile(draws_j, [0.05, 0.95], axis=1)
                valid = np.isfinite(err)
                maturity_rows.append({"horizon_weeks": h, "model": name, "maturity_years": tau,
                                      "rmse": float(np.sqrt(np.mean(err[valid] ** 2))),
                                      "mae": float(np.mean(np.abs(err[valid]))),
                                      "coverage90": float(np.mean((target_store[valid, horizon_index[h], j] >= lo[valid]) &
                                                                   (target_store[valid, horizon_index[h], j] <= hi[valid]))),
                                      "mean_width90": float(np.mean(hi[valid] - lo[valid])),
                                      "n_forecasts": int(valid.sum())})
    write_csv(output_dir / "model_summary.csv", summary_rows)
    write_csv(output_dir / "maturity_metrics.csv", maturity_rows)

    dm_results = {}
    mcs_results = {}
    rng_compare = np.random.default_rng(int(config["seed"]) + 9001)
    for h in horizons:
        for score_name in ("dss", "energy_score"):
            pivot = {}
            for row in score_rows:
                if row["horizon_weeks"] == h:
                    pivot.setdefault(row["origin_date"], {})[row["model"]] = row[score_name]
            paired = [(d[MODEL_NAMES[4]], d[MODEL_NAMES[5]]) for d in pivot.values()
                      if MODEL_NAMES[4] in d and MODEL_NAMES[5] in d]
            if paired:
                diff = np.array([a - b for a, b in paired])
                dm_results[f"h{h}_{score_name}_M4_minus_M5"] = hac_dm(diff, max_lag=3)
            matrix = []
            names_here = []
            for name in MODEL_NAMES:
                pairs = [(d[name], d) for d in pivot.values() if name in d]
                if pairs:
                    matrix.append([v for v, _ in pairs])
                    names_here.append(name)
            if matrix and len({len(v) for v in matrix}) == 1 and not pilot:
                mcs_results[f"h{h}_{score_name}"] = model_confidence_set(
                    np.column_stack(matrix), names_here, rng_compare, reps=mcs_reps,
                    expected_block_length=float(config["mcs_expected_block_length_origins"]))
    (output_dir / "dm_tests.json").write_text(json.dumps(dm_results, indent=2), encoding="utf-8")
    (output_dir / "model_confidence_sets.json").write_text(json.dumps(mcs_results, indent=2), encoding="utf-8")

    risk_summary = []
    for h in horizons:
        for name in MODEL_NAMES:
            subset = [r for r in risk_rows if r["horizon_weeks"] == h and r["model"] == name]
            if not subset:
                continue
            for tail in (0.05, 0.01):
                var_key = "VaR95" if tail == 0.05 else "VaR99"
                bt = var_backtests(np.array([r["realized_loss"] for r in subset]),
                                   np.array([r[var_key] for r in subset]), tail)
                risk_summary.append({"horizon_weeks": h, "model": name, "tail_probability": tail,
                                     "mean_VaR": float(np.mean([r[var_key] for r in subset])),
                                     "mean_ES": float(np.mean([r["ES95" if tail == 0.05 else "ES99"] for r in subset])),
                                     **bt})
    write_csv(output_dir / "risk_summary.csv", risk_summary)

    # Deterministic figures regenerated entirely from the saved forecast arrays.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    initial_mean = yields_all[initial_indices].mean(axis=0)
    oos_indices = [i for i, d in enumerate(dates) if oos_start <= d <= oos_end]
    oos_mean = yields_all[oos_indices].mean(axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(MATURITIES, initial_mean, marker="o", label="Initial window (2000–2015)")
    ax.plot(MATURITIES, oos_mean, marker="s", label="Out of sample (2016–2025)")
    ax.set(xlabel="Maturity (years)", ylabel="Yield (%)", title="Average Canadian zero-coupon yield curve")
    ax.legend(frameon=False); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(figures / ("yield_curve_sample_pilot.png" if pilot else "yield_curve_sample.png"), dpi=180)
    plt.close(fig)
    primary = [r for r in summary_rows if r["horizon_weeks"] == 4]
    if primary:
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.bar([r["model"].replace("_", "\n", 1) for r in primary], [r["dss"] for r in primary], color="#496a81")
        ax.set(ylabel="Mean DSS (lower is better)", title="Four-week probabilistic forecast comparison")
        ax.tick_params(axis="x", labelsize=8); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
        fig.savefig(figures / ("dss_comparison_pilot.png" if pilot else "dss_comparison.png"), dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for name in (MODEL_NAMES[4], MODEL_NAMES[5]):
            rows = [r for r in maturity_rows if r["horizon_weeks"] == 4 and r["model"] == name]
            ax.plot([r["maturity_years"] for r in rows], [r["coverage90"] for r in rows], marker="o", label=name[3:])
        ax.axhline(0.90, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Maturity (years)", ylabel="Empirical 90% interval coverage", ylim=(0, 1),
               title="Marginal interval calibration at four weeks")
        ax.legend(frameon=False); ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(figures / ("coverage90_pilot.png" if pilot else "coverage90.png"), dpi=180)
        plt.close(fig)

    elapsed = time.perf_counter() - started
    metadata = {"scope": "pilot" if pilot else "full", "started_utc": iso_time(),
                "elapsed_seconds": elapsed, "rows_total": len(dates), "date_first": dates[0].isoformat(),
                "date_last": dates[-1].isoformat(), "initial_training_rows": len(initial_indices),
                "lambda_per_year": lam, "lambda_grid_sse_percent_squared": lambda_sse,
                "out_of_sample_origins": len(origin_indices), "forecast_records": len(score_rows),
                "horizons_weeks": horizons, "models": MODEL_NAMES, "mcmc_warmup": warmup,
                "mcmc_draws_per_factor": mcmc_draws, "predictive_draws": draws,
                "risk_draws_per_forecast": risk_draws, "mcs_repetitions": mcs_reps,
                "state_space_maxiter": state_space_maxiter, "state_space_fit_success_count": sum(bool(r.get("success")) for r in fit_rows if "success" in r),
                "state_space_fit_count": sum(1 for r in fit_rows if "success" in r),
                "diagnostic_origin": dates[diagnostic_origin].isoformat() if diagnostic_origin is not None and not pilot else None}
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{iso_time()}] FINISH {metadata['scope']} elapsed_s={elapsed:.1f} forecasts={len(score_rows)}\n")
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="small, explicitly preliminary MCMC/forecast pilot")
    parser.add_argument("--max-origins", type=int, help="limit origin count for a smoke/pilot run")
    args = parser.parse_args()
    run(pilot=args.pilot, max_origins=args.max_origins)


if __name__ == "__main__":
    main()
