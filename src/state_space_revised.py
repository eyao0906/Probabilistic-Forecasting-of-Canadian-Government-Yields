"""Independently implemented restricted Gaussian DNS state-space benchmark M3.

Stationary initialization, diagonal transition/Q/R, exact predictive moments.
This module never calls the legacy Kalman/filter/forecast or writes its results.
"""
from __future__ import annotations
import numpy as np
import time
from scipy.optimize import minimize
from numba import njit


def evaluate_forecast(mean, covariance, observed, current, seed):
    """Comparable accepted-baseline scores; MC is used only for ensemble scores/risk."""
    from scipy.stats import norm
    from src.scores import dss, energy_score, variogram_score
    tau = np.array([.25, .5, 1., 2., 5., 7., 10., 30.])
    scenarios = np.random.default_rng(seed).multivariate_normal(mean, covariance, size=1024,
                                                                 check_valid='raise')
    error = observed-mean; sd = np.sqrt(np.diag(covariance)); z = norm.ppf(.95)
    hits = abs(error) <= z*sd; widths = 2*z*sd
    score = dss(observed, mean, covariance)
    metrics = {'dss': score, 'gaussian_negative_log_score': float(.5*(8*np.log(2*np.pi)+score)),
               'energy_score': energy_score(scenarios, observed),
               'variogram_score': variogram_score(scenarios, observed),
               'squared_error_average': float(np.mean(error**2)),
               'mae': float(np.mean(abs(error))), 'coverage90': float(hits.mean()),
               'width90': float(widths.mean())}
    maturity = [{'maturity': float(tau[j]), 'squared_error': float(error[j]**2),
                 'absolute_error': float(abs(error[j])), 'coverage90': int(hits[j]),
                 'pit': float(norm.cdf(error[j]/sd[j])), 'width90': float(widths[j])} for j in range(8)]
    delta = ((scenarios-current)/100) @ (tau/8)
    risk = {'duration_shock_loss': float(((observed-current)/100)@(tau/8))}
    for p in [.95, .99]:
        threshold = float(np.quantile(delta, p)); label = str(round(100*p))
        risk['VaR'+label] = threshold; risk['ES'+label] = float(delta[delta >= threshold].mean())
    return {'means': mean, 'covariances': covariance, 'scenarios': scenarios,
            'targets': observed, 'metrics': metrics, 'maturity': maturity, 'risk': risk}


def prepare_design(dates, y, config):
    from datetime import date
    from src.models import ns_loadings, fit_lambda
    from src.experiment import select_origins
    initial = [i for i, d in enumerate(dates)
               if date.fromisoformat(config['training_start']) <= d <= date.fromisoformat(config['training_end'])]
    first = initial[0]
    lam, _, sse = fit_lambda(y[initial], np.array(config['maturities_years']),
                            np.linspace(*config['lambda_grid_per_year']))
    selected = select_origins(dates, date.fromisoformat(config['oos_start']),
                              date.fromisoformat(config['oos_end']), config['horizons_weeks'],
                              config['origin_stride_observations'])
    return {'lambda_per_year': lam, 'initial_sse': sse, 'initial_indices': initial,
            'initial_training_n': len(initial), 'first_training_index': first,
            'selected': selected, 'origin_indices': sorted({t for t, _, _ in selected}),
            'load': ns_loadings(np.array(config['maturities_years']), lam)}


def parameter_bounds(dimension):
    """Preserve legacy restrictions, explicitly: mu/u/log(Q)/log(R)."""
    return np.array([(-20., 30.)]*3 + [(-3., 3.)]*3 + [(-15., 5.)]*(3+dimension))


def starting_values(y, load, n_starts=3, warm_start=None):
    factors = np.linalg.lstsq(load, y.T, rcond=None)[0].T
    mu = factors.mean(axis=0); rho = np.empty(3); q = np.empty(3)
    for j in range(3):
        lag = factors[:-1, j]; target = factors[1:, j]
        coef = np.linalg.lstsq(np.column_stack((np.ones(len(lag)), lag)), target, rcond=None)[0]
        rho[j] = np.clip(coef[1], -.98, .98)
        intercept = np.mean(target-rho[j]*lag)
        q[j] = max(np.mean((target-intercept-rho[j]*lag)**2), 1e-4)
    r = np.maximum(np.mean((y-factors@load.T)**2, axis=0), 1e-6)
    base = np.r_[mu, np.arctanh(rho), np.log(q), np.log(r)]
    bounds = parameter_bounds(y.shape[1])
    starts = [base]
    for k in range(1, n_starts):
        candidate = base.copy()
        if k == 1 and warm_start is not None:
            candidate = np.asarray(warm_start).copy()
        elif k % 2:
            candidate[3:6] *= .65
            candidate[6:9] += np.log(2.)
            candidate[9:] += np.log(.5)
        else:
            candidate[:3] += np.array([.4, -.3, .5])
            candidate[3:6] *= 1.2
            candidate[6:9] += np.log(.5)
            candidate[9:] += np.log(2.)
        starts.append(candidate)
    return [np.clip(p, bounds[:, 0], bounds[:, 1]) for p in starts]


def projected_gradient(params, gradient, bounds):
    """KKT residual: outward slopes at active bounds are permitted, not hidden."""
    projected = gradient.copy()
    projected[(params <= bounds[:, 0]+1e-7) & (gradient > 0)] = 0.
    projected[(params >= bounds[:, 1]-1e-7) & (gradient < 0)] = 0.
    return projected


def fit_mle(y, load, n_starts=3, maxiter=1000, warm_start=None):
    """Multistart constrained MLE; convergence is assessed separately from exit flags."""
    y = np.ascontiguousarray(y, dtype=float); load = np.ascontiguousarray(load, dtype=float)
    bounds = parameter_bounds(y.shape[1]); diagnostics = []; candidates = []
    starts = starting_values(y, load, n_starts, warm_start)
    def objective(p):
        value, gradient = likelihood_gradient(y, load, p)
        return value/len(y), gradient/len(y)
    # Exclude first compilation from individual optimizer timings, report at run level.
    objective(starts[0])
    for k, start in enumerate(starts):
        before = time.perf_counter(); initial_nll = filter_state(y, load, start)[0]
        result = minimize(objective, start, method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 2e-7,
                                   'maxls': 40, 'maxcor': 20})
        value, gradient = likelihood_gradient(y, load, result.x)
        projected = projected_gradient(result.x, gradient, bounds)
        pg = float(np.max(abs(projected))/len(y))
        active = np.flatnonzero((result.x <= bounds[:, 0]+1e-6) | (result.x >= bounds[:, 1]-1e-6))
        diagnostic = {'start_id': k, 'initial_nll': float(initial_nll), 'nll': float(value),
                      'success': bool(result.success), 'message': str(result.message),
                      'iterations': int(result.nit), 'evaluations': int(result.nfev),
                      'elapsed_s': time.perf_counter()-before, 'projected_gradient_per_observation': pg,
                      'raw_gradient_inf': float(np.max(abs(gradient))),
                      'active_bound_indices': active.tolist(), 'gradient': gradient.tolist(),
                      'params': result.x.tolist(), 'initial_params': start.tolist()}
        diagnostics.append(diagnostic); candidates.append(result.x)
    best = int(np.argmin([d['nll'] for d in diagnostics])); chosen = diagnostics[best]
    spread = float(np.ptp([d['nll'] for d in diagnostics]))
    accepted = all(d['projected_gradient_per_observation'] < 1e-5 and np.isfinite(d['nll'])
                   for d in diagnostics) and spread < .02
    return {'params': candidates[best], 'nll': chosen['nll'], 'diagnostics': diagnostics,
            'selected_start': best, 'start_nll_spread': spread, 'accepted': bool(accepted),
            'projected_gradient_per_observation': chosen['projected_gradient_per_observation']}


@njit(cache=False)
def _filter(y, load, params):
    """Scalar innovation form is exact because R is diagonal."""
    mu = params[:3]
    rho = np.tanh(params[3:6])
    q = np.exp(params[6:9]); r = np.exp(params[9:])
    state = mu.copy(); p = np.diag(q / (1 - rho * rho))
    total = 0.0
    for t in range(len(y)):
        for i in range(y.shape[1]):
            h = load[i]
            b = p @ h
            variance = h @ b + r[i]
            innovation = y[t, i] - h @ state
            total += .5 * (np.log(2*np.pi) + np.log(variance) + innovation**2 / variance)
            state += b * (innovation / variance)
            p -= np.outer(b, b) / variance
        p = .5 * (p + p.T)
        if t < len(y)-1:
            state = mu + rho * (state - mu)
            p = rho[:, None] * p * rho[None, :] + np.diag(q)
    return total, state, p


def filter_state(y, load, params):
    """Return exact stationary-initialized negative log likelihood and last filter."""
    y = np.ascontiguousarray(y, dtype=float)
    load = np.ascontiguousarray(load, dtype=float)
    params = np.asarray(params, dtype=float)
    if y.ndim != 2 or y.shape[1] != len(load) or load.shape[1] != 3:
        raise ValueError('expected T by D yields and D by 3 loadings')
    if len(params) != 9 + len(load) or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError('invalid parameter length or input observations')
    return _filter(y, load, params)


@njit(cache=False)
def likelihood_gradient(y, load, params):
    """Forward sensitivity recursion: exact gradient of stationary Gaussian NLL."""
    mu = params[:3]; rho = np.tanh(params[3:6])
    q = np.exp(params[6:9]); r = np.exp(params[9:])
    npar = len(params)
    state = mu.copy(); p = np.diag(q / (1-rho*rho))
    dm = np.zeros((npar, 3)); dp = np.zeros((npar, 3, 3))
    for j in range(3):
        dm[j, j] = 1.
        dp[3+j, j, j] = 2*rho[j]*p[j, j]
        dp[6+j, j, j] = p[j, j]
    total = 0.; gradient = np.zeros(npar); db = np.empty(3)
    for t in range(len(y)):
        for i in range(y.shape[1]):
            h = load[i]; b = p @ h
            variance = h @ b + r[i]
            innovation = y[t, i] - h @ state
            total += .5*(np.log(2*np.pi)+np.log(variance)+innovation**2/variance)
            for k in range(npar):
                for j in range(3):
                    db[j] = dp[k, j, 0]*h[0] + dp[k, j, 1]*h[1] + dp[k, j, 2]*h[2]
                dv = -(h[0]*dm[k, 0] + h[1]*dm[k, 1] + h[2]*dm[k, 2])
                df = h[0]*db[0] + h[1]*db[1] + h[2]*db[2] + (r[i] if k == 9+i else 0.)
                gradient[k] += .5*((1/variance-innovation**2/variance**2)*df + 2*innovation/variance*dv)
                for j in range(3):
                    dm[k, j] += db[j]*innovation/variance + b[j]*dv/variance - b[j]*innovation*df/variance**2
                    for l in range(3):
                        dp[k, j, l] += -(db[j]*b[l]+b[j]*db[l])/variance + b[j]*b[l]*df/variance**2
            state += b*(innovation/variance)
            p -= np.outer(b, b)/variance
        p = .5*(p+p.T)
        if t < len(y)-1:
            for k in range(npar):
                for j in range(3):
                    dm[k, j] *= rho[j]
                    if k == j:
                        dm[k, j] += 1-rho[j]
                    if k == 3+j:
                        dm[k, j] += (1-rho[j]**2)*(state[j]-mu[j])
                    for l in range(3):
                        dp[k, j, l] *= rho[j]*rho[l]
                        if k == 3+j:
                            dp[k, j, l] += (1-rho[j]**2)*p[j, l]*rho[l]
                        if k == 3+l:
                            dp[k, j, l] += rho[j]*p[j, l]*(1-rho[l]**2)
                        if j == l and k == 6+j:
                            dp[k, j, l] += q[j]
            state = mu+rho*(state-mu)
            p = rho[:, None]*p*rho[None, :]+np.diag(q)
    return total, gradient


def unpack(params):
    params = np.asarray(params, dtype=float)
    return params[:3], np.tanh(params[3:6]), np.exp(params[6:9]), np.exp(params[9:])


def forecast_moments(state, covariance, load, params, horizon):
    """Conditional Gaussian moments, without Monte Carlo or eigenvalue clipping."""
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    mu, rho, q, r = unpack(params)
    ah = rho ** horizon
    mean_f = mu + ah * (state - mu)
    cov_f = ah[:, None] * covariance * ah[None, :]
    cov_f += np.diag(q * sum(rho ** (2 * k) for k in range(horizon)))
    covariance_y = load @ cov_f @ load.T + np.diag(r)
    return load @ mean_f, (covariance_y + covariance_y.T) / 2


def boundary_profiles(y, load, params):
    """One-coordinate inward/outward checks, diagnostic only; no change to bounds."""
    bounds = parameter_bounds(y.shape[1]); value = filter_state(y, load, params)[0]
    active = np.flatnonzero((params <= bounds[:, 0]+1e-6) | (params >= bounds[:, 1]-1e-6))
    profiles = []
    for j in active:
        inward = params.copy(); outward = params.copy(); step = .05 if j < 6 else .5
        direction = 1 if params[j] <= bounds[j, 0]+1e-6 else -1
        inward[j] += direction*step; outward[j] -= direction*step
        profiles.append({'parameter_index': int(j), 'inward_step': direction*step,
                         'inward_delta_nll': float(filter_state(y, load, inward)[0]-value),
                         'outward_delta_nll': float(filter_state(y, load, outward)[0]-value)})
    return profiles


def run_experiment(phase='full', output_dir=None, origin_positions=None, n_starts=3, maxiter=1000):
    """Run training-only expanding origins; outputs are isolated from accepted M0--M2."""
    import csv, hashlib, json, sys
    from pathlib import Path
    from datetime import datetime, timezone
    from src.experiment import read_panel, write_csv
    import scipy, numba
    root = Path(__file__).resolve().parents[1]
    allowed = root/'results/state_space_revised'
    out = Path(output_dir) if output_dir is not None else (allowed/'pilot' if phase == 'pilot' else allowed)
    if not out.resolve().is_relative_to(allowed.resolve()):
        raise ValueError('output must stay inside results/state_space_revised')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); utc = datetime.now(timezone.utc).isoformat()
    config_path = root/'config/experiment.json'; data_path = root/'data/processed/canada_weekly.csv'
    config = json.loads(config_path.read_text()); dates, y = read_panel(data_path)
    design = prepare_design(dates, y, config); load = design['load']; origins = design['origin_indices']
    if origin_positions is not None:
        origins = [origins[k] for k in origin_positions]
    elif phase == 'pilot':
        origins = [origins[0], origins[len(origins)//2], origins[-1]]
    baseline_paths = sorted((root/'results/validated_baselines').glob('*'))
    baseline_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_paths if p.is_file()}
    fits = []; forecast_list = []; scores = []; maturity_rows = []; risk_rows = []; audit = []; keys = []; target_dates = []
    previous = None
    for counter, t in enumerate(origins, 1):
        train = np.ascontiguousarray(y[design['first_training_index']:t+1])
        begin = time.perf_counter()
        # Pilots have independent starts; full rolling fits add a previous-origin warm start,
        # but always retain OLS and (with 3 starts) an independently perturbed candidate.
        fit = fit_mle(train, load, n_starts=n_starts, maxiter=maxiter,
                      warm_start=previous if phase == 'full' else None)
        params = fit['params']; previous = params.copy()
        nll, state, p = filter_state(train, load, params)
        fit.update(origin_date=dates[t].isoformat(), n_training=len(train),
                   latest_fit_date=dates[t].isoformat(), boundary_profiles=boundary_profiles(train, load, params),
                   filtered_state=state.tolist(), filtered_covariance=p.tolist(), elapsed_s=time.perf_counter()-begin)
        fit['params'] = params.tolist(); fits.append(fit)
        # Incremental durable diagnostics survive interruption.
        (out/'optimizer_diagnostics.json').write_text(json.dumps(fits, indent=2), encoding='utf-8')
        for h in config['horizons_weeks']:
            target = t+h
            assert target < len(dates) and dates[target].isoformat() <= config['oos_end']
            mean, covariance = forecast_moments(state, p, load, params, h)
            record = evaluate_forecast(mean, covariance, y[target], y[t], 20261027+t*17+h)
            forecast_list.append(record); keys.append([dates[t].isoformat(), str(h), 'M3'])
            target_dates.append(dates[target].isoformat())
            header = {'origin_date': dates[t].isoformat(), 'target_date': dates[target].isoformat(),
                      'horizon': h, 'model': 'M3'}
            scores.append({**header, **record['metrics']})
            maturity_rows.extend({**header, **row} for row in record['maturity'])
            risk_rows.append({**header, **record['risk']})
            audit.append({**header, 'n_training': len(train), 'latest_fit_date': dates[t].isoformat(),
                          'elapsed_calendar_days': (dates[target]-dates[t]).days,
                          'fit_accepted': fit['accepted'],
                          'min_covariance_eigenvalue': float(np.linalg.eigvalsh(covariance).min()),
                          'train_sha256': hashlib.sha256(train.tobytes()).hexdigest()})
        if counter % 10 == 0 or counter == len(origins) or phase == 'pilot':
            print(f'{phase} {counter}/{len(origins)} {dates[t]} accepted={fit["accepted"]} '
                  f'pg={fit["projected_gradient_per_observation"]:.3g} '
                  f'spread={fit["start_nll_spread"]:.3g} elapsed={time.perf_counter()-started:.2f}s', flush=True)
    for name, rows in [('scores', scores), ('maturity_by_origin', maturity_rows),
                       ('risk_scenarios', risk_rows), ('origin_audit', audit)]:
        write_csv(out/(name+'.csv'), rows)
    summary = []
    for h in config['horizons_weeks']:
        subset = [r for r in scores if r['horizon'] == h]
        summary.append({'horizon': h, 'model': 'M3', 'n': len(subset),
                        **{key: float(np.mean([r[key] for r in subset])) for key in
                           ['dss', 'gaussian_negative_log_score', 'energy_score', 'variogram_score',
                            'mae', 'coverage90', 'width90']},
                        'pooled_rmse': float(np.sqrt(np.mean([r['squared_error_average'] for r in subset])))})
    write_csv(out/'summary.csv', summary)
    arrays = {name: np.array([r[name] for r in forecast_list])
              for name in ['scenarios', 'means', 'covariances', 'targets']}
    np.savez_compressed(out/'forecasts.npz', **arrays, keys=np.array(keys), target_dates=np.array(target_dates),
                        maturities=np.array(config['maturities_years']), lambda_per_year=design['lambda_per_year'])
    fit_summary = []
    for fit in fits:
        chosen = fit['diagnostics'][fit['selected_start']]
        fit_summary.append({key: fit[key] for key in ['origin_date', 'n_training', 'nll', 'accepted',
                                                      'selected_start', 'start_nll_spread',
                                                      'projected_gradient_per_observation', 'elapsed_s']} |
                           {'active_bounds': ';'.join(map(str, chosen['active_bound_indices'])),
                            'total_evaluations': sum(d['evaluations'] for d in fit['diagnostics'])})
    write_csv(out/'fit_summary.csv', fit_summary)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_paths if p.is_file()}
    assert baseline_hashes == after, 'Accepted baselines changed during execution'
    accepted = all(f['accepted'] for f in fits)
    complete = len(origins) == len(design['origin_indices']) and len(scores) == len(design['selected'])
    status = ('accepted_restricted_full' if complete else 'accepted_restricted_partial_'+phase) if accepted else 'rejected_fit_diagnostics'
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__), root/'tests/test_state_space_revised.py', data_path, config_path]}
    metadata = {'status': status, 'phase': phase, 'started_utc': utc,
                'elapsed_s': time.perf_counter()-started, 'python': sys.executable,
                'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'numba': numba.__version__},
                'n_origins': len(origins), 'forecast_records': len(scores), 'common_design_origins': len(design['origin_indices']),
                'first_origin': dates[origins[0]].isoformat(), 'last_origin': dates[origins[-1]].isoformat(),
                'initial_training_n': design['initial_training_n'], 'lambda_per_year': design['lambda_per_year'],
                'initial_sse': design['initial_sse'], 'horizons_observations': config['horizons_weeks'],
                'scenarios_per_record': 1024, 'seed_rule': '20261027+origin_panel_index*17+horizon',
                'n_starts': n_starts, 'maxiter': maxiter, 'parameter_bounds': parameter_bounds(8).tolist(),
                'acceptance': {'all_starts_projected_gradient_per_observation_below': 1e-5,
                               'start_nll_spread_below': .02},
                'fit_accept_count': sum(f['accepted'] for f in fits),
                'optimizer_success_count': sum(d['success'] for f in fits for d in f['diagnostics']),
                'optimizer_attempts': sum(len(f['diagnostics']) for f in fits),
                'baseline_hashes_before': baseline_hashes, 'baseline_hashes_after': after,
                'source_and_input_sha256': source_hashes,
                'limitations': ['restricted stationary diagonal transition, Q and R; not unrestricted DNS VAR',
                                'finite mu/u/logvariance bounds retained from legacy benchmark; active bounds explicitly diagnosed',
                                'conditional on estimated parameters and fixed initial lambda; no parameter uncertainty',
                                'historical revised-vintage observation-time, NOT publication-time real-time',
                                'Bank of Canada two-week publication lag not applied as an availability filter',
                                'horizons count observations; calendar durations may vary',
                                '1024-scenario Energy Score U-statistic excludes diagonal self-distances',
                                'first-order equal-market-value duration shock, not actual bond returns; no backtest claim']}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(status, 'records', len(scores), 'seconds', metadata['elapsed_s'], flush=True)
    return metadata


def main():
    import argparse, json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['pilot', 'full', 'all'], default='all')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; pilot_path = root/'results/state_space_revised/pilot/metadata.json'
    if args.phase in ['pilot', 'all']:
        pilot = run_experiment(phase='pilot', n_starts=3)
        if pilot['fit_accept_count'] != 3:
            raise RuntimeError('Pilot rejected; no full forecast run permitted')
    if args.phase in ['full', 'all']:
        if not pilot_path.exists():
            raise RuntimeError('Run pilot before full')
        pilot = json.loads(pilot_path.read_text())
        if pilot['fit_accept_count'] != 3 or pilot['elapsed_s'] > 600:
            raise RuntimeError('Pilot diagnostics/runtime gate did not pass')
        run_experiment(phase='full', n_starts=3)


if __name__ == '__main__':
    main()
