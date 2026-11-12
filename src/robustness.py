"""Supplementary availability-lag sensitivity; not historical vintage reconstruction."""
from bisect import bisect_right
from datetime import timedelta


def available_index(dates, origin_index, lag_days=14):
    if lag_days < 0:
        raise ValueError('Availability lag must be nonnegative')
    cutoff = dates[origin_index] - timedelta(days=lag_days)
    index = bisect_right(dates, cutoff, hi=origin_index + 1) - 1
    if index < 0:
        raise ValueError('No eligible curve before availability cutoff')
    return index


def lagged_forecast(dates, yields, origin_index, target_index, model, load, first=0, lag_days=14):
    from src.validated_baselines import forecast_baseline
    k = available_index(dates, origin_index, lag_days)
    if k < first or target_index <= origin_index:
        raise ValueError('Invalid training or target index')
    mean, covariance = forecast_baseline(yields[first:k+1], model, target_index-k, load)
    return mean, covariance, k


def run(out=None, max_origins=None, ensemble=1024, lag_days=14, seed=20261028):
    import json, time, hashlib
    from pathlib import Path
    from datetime import date, datetime, timezone
    import numpy as np
    from scipy.stats import norm
    from src.experiment import read_panel, select_origins, write_csv
    from src.models import fit_lambda, ns_loadings
    from src.scores import dss, energy_score, variogram_score
    root = Path(__file__).resolve().parents[1]
    out = Path(out) if out is not None else root / 'results/availability_robustness'
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    panel = root / 'data/processed/canada_weekly.csv'
    dates, y = read_panel(panel)
    first = next(i for i, day in enumerate(dates) if day >= date(2000, 1, 1))
    selected = select_origins(dates, date(2016, 1, 1), date(2025, 12, 31), [1, 4], 4)
    if max_origins is not None:
        keep = {t for t, _, _ in selected[:2 * max_origins]}
        selected = [row for row in selected if row[0] in keep]
    first_available = available_index(dates, selected[0][0], lag_days)
    initial_end = min(first_available, max(i for i, day in enumerate(dates) if day <= date(2015, 12, 31)))
    tau = np.array([.25, .5, 1, 2, 5, 7, 10, 30])
    lam, _, _ = fit_lambda(y[first:initial_end+1], tau, np.linspace(.05, 2., 160))
    load = ns_loadings(tau, lam)
    rng = np.random.default_rng(seed)
    rows, means, covs, targets, keys = [], [], [], [], []
    for t, h, target in selected:
        for model in ['M0', 'M1', 'M2']:
            mean, cov, k = lagged_forecast(dates, y, t, target, model, load, first, lag_days)
            actual = y[target]; err = actual - mean
            draws = rng.multivariate_normal(mean, cov, size=ensemble)
            sd = np.sqrt(np.diag(cov)); z = norm.ppf(.95)
            rows.append({'origin_date': dates[t].isoformat(), 'target_date': dates[target].isoformat(),
                         'latest_fit_date': dates[k].isoformat(), 'horizon': h, 'model': model,
                         'bridge_steps': target-k, 'training_n': k-first+1, 'dss': dss(actual, mean, cov),
                         'energy_score': energy_score(draws, actual), 'variogram_score': variogram_score(draws, actual),
                         'squared_error_average': float(np.mean(err**2)), 'mae': float(np.mean(abs(err))),
                         'coverage90': float(np.mean(abs(err) <= z*sd)), 'width90': float(np.mean(2*z*sd))})
            means.append(mean); covs.append(cov); targets.append(actual); keys.append([dates[t].isoformat(), str(h), model])
    write_csv(out / 'scores.csv', rows)
    summary = []
    for h in [1, 4]:
        for model in ['M0', 'M1', 'M2']:
            sub = [row for row in rows if row['horizon'] == h and row['model'] == model]
            summary.append({'horizon': h, 'model': model, 'n': len(sub),
                            **{key: float(np.mean([r[key] for r in sub])) for key in ['dss', 'energy_score', 'variogram_score', 'mae', 'coverage90', 'width90']},
                            'pooled_rmse': float(np.sqrt(np.mean([r['squared_error_average'] for r in sub])))})
    write_csv(out / 'summary.csv', summary)
    np.savez_compressed(out / 'forecasts.npz', means=np.array(means), covariances=np.array(covs), targets=np.array(targets), keys=np.array(keys))
    report = {'status': 'supplementary assumed availability-lag sensitivity, not vintage-correct real-time test',
              'lag_days': lag_days, 'lambda_per_year': lam, 'lambda_training_latest': dates[initial_end].isoformat(),
              'seed': seed, 'ensemble': ensemble, 'forecast_records': len(rows), 'origins': len({r['origin_date'] for r in rows}),
              'input_sha256': hashlib.sha256(panel.read_bytes()).hexdigest(), 'elapsed_s': time.perf_counter()-started,
              'utc': datetime.now(timezone.utc).isoformat(), 'limitation': 'Fixed 14 calendar days is an assumption, not authenticated historical release dates; revised vintage remains.'}
    (out / 'metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=2))
