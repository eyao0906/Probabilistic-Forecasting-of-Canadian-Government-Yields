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
    load = ns_loadings(np.array(config['maturities_years']), lam)
    measurement_variance = calibrate_measurement_covariance(
        y[initial], load, config.get('variance_floor_percent_squared', 1e-6))
    return {'lambda_per_year': lam, 'initial_sse': sse, 'initial_indices': initial,
            'initial_training_n': len(initial), 'first_training_index': first,
            'selected': selected, 'origin_indices': sorted({t for t, _, _ in selected}),
            'load': load, 'initial_measurement_variance': measurement_variance,
            'initial_training_end_index': initial[-1]}


def parameter_bounds(dimension, persistence_bound=3., measurement_mode='estimated'):
    """Bounds in mu/u/log(Q)/log(R) order; fixed-R models omit log(R)."""
    if measurement_mode not in {'estimated', 'fixed', 'regularized'}:
        raise ValueError(f'unknown measurement covariance mode: {measurement_mode}')
    if not np.isfinite(persistence_bound) or persistence_bound <= 0:
        raise ValueError('persistence_bound must be finite and positive')
    return np.array([(-20., 30.)]*3 + [(-persistence_bound, persistence_bound)]*3
                    + [(-15., 5.)]*(3 + (dimension if measurement_mode != 'fixed' else 0)))


def parameter_count(dimension, measurement_mode='estimated'):
    if measurement_mode not in {'estimated', 'fixed', 'regularized'}:
        raise ValueError(f'unknown measurement covariance mode: {measurement_mode}')
    return 9 if measurement_mode == 'fixed' else 9 + dimension


def calibrate_measurement_covariance(y, load, floor=1e-6):
    """Training-only diagonal R calibration from cross-sectional DNS residuals."""
    y = np.asarray(y, dtype=float); load = np.asarray(load, dtype=float)
    if y.ndim != 2 or load.shape != (y.shape[1], 3) or len(y) < 2:
        raise ValueError('expected at least two T by D yields and a D by 3 loading matrix')
    if not np.isfinite(y).all() or not np.isfinite(load).all():
        raise ValueError('calibration inputs must be finite')
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError('measurement variance floor must be finite and positive')
    factors = np.linalg.lstsq(load, y.T, rcond=None)[0].T
    residual = y - factors @ load.T
    variance = np.mean(residual**2, axis=0)
    if not np.isfinite(variance).all():
        raise ValueError('non-finite training measurement covariance calibration')
    return np.maximum(variance, floor)


def _stationary_variances(u, q):
    """Compute q/(1-tanh(u)^2) without subtractive cancellation."""
    return q * np.cosh(u)**2


def starting_values(y, load, n_starts=3, warm_start=None, persistence_bound=3.,
                    measurement_mode='estimated', fixed_measurement=None):
    factors = np.linalg.lstsq(load, y.T, rcond=None)[0].T
    mu = factors.mean(axis=0); rho = np.empty(3); q = np.empty(3)
    for j in range(3):
        lag = factors[:-1, j]; target = factors[1:, j]
        coef = np.linalg.lstsq(np.column_stack((np.ones(len(lag)), lag)), target, rcond=None)[0]
        rho[j] = np.clip(coef[1], -.98, .98)
        intercept = np.mean(target-rho[j]*lag)
        q[j] = max(np.mean((target-intercept-rho[j]*lag)**2), 1e-4)
    r = np.maximum(np.mean((y-factors@load.T)**2, axis=0), 1e-6)
    base = np.r_[mu, np.arctanh(rho), np.log(q)]
    if measurement_mode == 'fixed':
        if fixed_measurement is None:
            raise ValueError('fixed measurement mode requires fixed_measurement')
        fixed_measurement = np.asarray(fixed_measurement, dtype=float)
        if fixed_measurement.shape != (y.shape[1],) or not np.isfinite(fixed_measurement).all() or np.any(fixed_measurement <= 0):
            raise ValueError('fixed_measurement must be a positive finite variance per maturity')
    else:
        base = np.r_[base, np.log(r)]
    bounds = parameter_bounds(y.shape[1], persistence_bound, measurement_mode)
    starts = [base]
    for k in range(1, n_starts):
        candidate = base.copy()
        if k == 1 and warm_start is not None:
            candidate = np.asarray(warm_start).copy()
        elif k % 2:
            candidate[3:6] *= .65
            candidate[6:9] += np.log(2.)
            if measurement_mode != 'fixed':
                candidate[9:] += np.log(.5)
        else:
            candidate[:3] += np.array([.4, -.3, .5])
            candidate[3:6] *= 1.2
            candidate[6:9] += np.log(.5)
            if measurement_mode != 'fixed':
                candidate[9:] += np.log(2.)
        if candidate.shape != base.shape:
            raise ValueError('warm_start parameter dimension does not match model specification')
        starts.append(candidate)
    # Clipping applies only to initial points; all optimizer solutions remain
    # explicitly subject to the same declared box constraints.
    return [np.clip(p, bounds[:, 0], bounds[:, 1]) for p in starts]


def projected_gradient(params, gradient, bounds):
    """KKT residual: outward slopes at active bounds are permitted, not hidden."""
    projected = gradient.copy()
    projected[(params <= bounds[:, 0]+1e-7) & (gradient > 0)] = 0.
    projected[(params >= bounds[:, 1]-1e-7) & (gradient < 0)] = 0.
    return projected


def fit_mle(y, load, n_starts=3, maxiter=1000, warm_start=None, *,
            persistence_bound=3., measurement_mode='estimated', fixed_measurement=None,
            penalty_strength=0., measurement_target=None, restart_count=0,
            restart_seed=271026, ftol=1e-14, gtol=2e-7):
    """Optimize original starts, then deterministic local restarts if requested.

    The three original starts are always optimized and retained. ``accepted``
    preserves the original all-start, 0.02 raw-NLL-spread diagnostic;
    ``revised_accepted`` separately assesses the best refined solution.
    """
    y = np.ascontiguousarray(y, dtype=float); load = np.ascontiguousarray(load, dtype=float)
    if y.ndim != 2 or load.shape != (y.shape[1], 3) or len(y) == 0:
        raise ValueError('expected nonempty T by D yields and a D by 3 loading matrix')
    if n_starts < 1 or restart_count < 0 or maxiter < 1:
        raise ValueError('n_starts/maxiter must be positive and restart_count nonnegative')
    if not np.isfinite(penalty_strength) or penalty_strength < 0:
        raise ValueError('penalty_strength must be finite and nonnegative')
    if measurement_mode == 'regularized':
        if measurement_target is None or penalty_strength <= 0:
            raise ValueError('regularized mode requires measurement_target and positive penalty_strength')
        measurement_target = np.asarray(measurement_target, dtype=float)
        if measurement_target.shape != (y.shape[1],) or not np.isfinite(measurement_target).all():
            raise ValueError('measurement_target must be finite log variances per maturity')
    elif penalty_strength != 0:
        raise ValueError('a measurement penalty is only valid in regularized mode')
    bounds = parameter_bounds(y.shape[1], persistence_bound, measurement_mode)
    starts = starting_values(y, load, n_starts, warm_start, persistence_bound,
                             measurement_mode, fixed_measurement)
    fixed_r = np.asarray(fixed_measurement, dtype=float) if fixed_measurement is not None else None
    rng = np.random.default_rng(restart_seed)

    def terms(p):
        raw_nll, raw_gradient = likelihood_gradient(
            y, load, p, measurement_mode, fixed_r)
        objective_nll = float(raw_nll)
        objective_gradient = raw_gradient.copy()
        if penalty_strength:
            difference = p[9:] - measurement_target
            objective_nll += .5 * penalty_strength * float(difference @ difference)
            objective_gradient[9:] += penalty_strength * difference
        return float(raw_nll), objective_nll, raw_gradient, objective_gradient

    def optimizer_objective(p):
        _, objective_nll, _, objective_gradient = terms(p)
        return objective_nll/len(y), objective_gradient/len(y)

    def valid_solution(p, objective_nll):
        if not np.isfinite(objective_nll) or not np.isfinite(p).all():
            return False
        if np.any(p < bounds[:, 0]-1e-10) or np.any(p > bounds[:, 1]+1e-10):
            return False
        try:
            raw_nll, state, covariance = filter_state(
                y, load, p, measurement_mode, fixed_r)
            return bool(np.isfinite(raw_nll) and np.isfinite(state).all()
                        and np.isfinite(covariance).all()
                        and np.linalg.eigvalsh(covariance).min() >= -1e-9)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return False

    attempts = []
    def optimize(start, *, start_id, is_restart, restart_id=None,
                 restart_center_start_id=None, restart_strategy=None, attempt_seed=None):
        start = np.asarray(start, dtype=float)
        initial_raw = float(filter_state(y, load, start, measurement_mode, fixed_r)[0])
        initial_objective = terms(start)[1]
        before = time.perf_counter()
        result = minimize(
            optimizer_objective,
            start, method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                     'maxls': 40, 'maxcor': 20})
        raw_nll, objective_nll, raw_gradient, objective_gradient = terms(result.x)
        pg = float(np.max(np.abs(projected_gradient(
            result.x, objective_gradient, bounds)))/len(y))
        active = np.flatnonzero(
            (result.x <= bounds[:, 0]+1e-6) | (result.x >= bounds[:, 1]-1e-6))
        diagnostic = {
            'start_id': int(start_id), 'is_restart': bool(is_restart),
            'restart_id': restart_id, 'restart_seed': attempt_seed,
            'restart_master_seed': restart_seed if is_restart else None,
            'restart_center_start_id': restart_center_start_id,
            'perturbation_strategy': restart_strategy,
            'initial_params': start.tolist(), 'params': result.x.tolist(),
            'initial_nll': initial_raw, 'nll': raw_nll,
            'initial_objective': initial_objective, 'objective': objective_nll,
            'raw_log_likelihood': -raw_nll, 'penalty': objective_nll-raw_nll,
            'success': bool(result.success), 'status': int(result.status),
            'message': str(result.message), 'iterations': int(result.nit),
            'evaluations': int(result.nfev), 'elapsed_s': time.perf_counter()-before,
            'projected_gradient_per_observation': pg,
            'raw_gradient_inf': float(np.max(np.abs(raw_gradient))),
            'objective_gradient_inf': float(np.max(np.abs(objective_gradient))),
            'active_bound_indices': active.tolist(),
            'distance_to_nearest_bound': np.minimum(
                result.x-bounds[:, 0], bounds[:, 1]-result.x).tolist(),
            'numerically_valid': valid_solution(result.x, objective_nll),
            'gradient': objective_gradient.tolist(),
        }
        attempts.append(diagnostic)
        return diagnostic

    # Compile before recording per-start optimizer time.
    terms(starts[0])
    original = [optimize(start, start_id=k, is_restart=False)
                for k, start in enumerate(starts)]
    original_valid = [d for d in original if d['numerically_valid']]
    if not original_valid:
        raise RuntimeError('all original M3 starts produced invalid likelihood solutions')

    def distinct_centers(diagnostics):
        ordered = sorted((d for d in diagnostics if d['numerically_valid']),
                         key=lambda item: item['objective'])
        centers = []
        scale = bounds[:, 1]-bounds[:, 0]
        for item in ordered:
            p = np.asarray(item['params'])
            if not any(np.max(np.abs(p-np.asarray(center['params']))/scale) <= 1e-3
                       for center in centers):
                centers.append(item)
        return centers

    initial_centers = distinct_centers(original)
    for restart_id in range(restart_count):
        centers = initial_centers[:3]
        center = centers[restart_id % len(centers)]
        center_start = np.asarray(center['params'])
        step = np.ones(len(center_start))
        step[:3] = .15; step[3:6] = .10; step[6:9] = .30
        if measurement_mode != 'fixed':
            step[9:] = .30
        perturbation_seed = int(rng.integers(0, 2**32-1))
        local_rng = np.random.default_rng(perturbation_seed)
        restart = np.clip(center_start + local_rng.normal(size=len(step))*step,
                          bounds[:, 0], bounds[:, 1])
        optimize(restart, start_id=int(center['start_id']), is_restart=True,
                 restart_id=restart_id, restart_center_start_id=int(center['start_id']),
                 restart_strategy='normal_perturbation_round_robin_original_basins',
                 attempt_seed=perturbation_seed)

    for basin_id, center in enumerate(distinct_centers(attempts)):
        center_params = np.asarray(center['params'])
        scale = bounds[:, 1]-bounds[:, 0]
        for item in attempts:
            p = np.asarray(item['params'])
            if np.max(np.abs(p-center_params)/scale) <= 1e-3:
                item['basin_id'] = basin_id
                item['objective_gap_to_best'] = None
                item['raw_nll_gap_to_best'] = None

    valid = [d for d in attempts if d['numerically_valid']]
    chosen = min(valid, key=lambda item: (item['objective'], item['nll']))
    best_objective = chosen['objective']; best_nll = chosen['nll']
    for item in attempts:
        item['objective_gap_to_best'] = (
            float(item['objective']-best_objective) if item['numerically_valid'] else None)
        item['raw_nll_gap_to_best'] = (
            float(item['nll']-best_nll) if item['numerically_valid'] else None)

    original_nlls = [d['nll'] for d in original if np.isfinite(d['nll'])]
    original_objectives = [d['objective'] for d in original if np.isfinite(d['objective'])]
    original_nll_spread = float(np.ptp(original_nlls))
    original_objective_spread = float(np.ptp(original_objectives))
    legacy_diagnostic_applicable = (
        measurement_mode == 'estimated' and penalty_strength == 0.)
    original_acceptance = (
        len(original_valid) == len(original)
        and all(d['projected_gradient_per_observation'] < 1e-5 for d in original)
        and original_nll_spread < .02)
    original_basin_count = len(distinct_centers(original))
    chosen_basin_replicates = sum(
        d.get('basin_id') == chosen.get('basin_id') for d in attempts)
    revised_accepted = bool(
        chosen['numerically_valid']
        and chosen['projected_gradient_per_observation'] < 1e-5
        and chosen_basin_replicates >= 2)
    return {
        'params': np.asarray(chosen['params']), 'nll': chosen['nll'],
        'objective': chosen['objective'], 'raw_log_likelihood': chosen['raw_log_likelihood'],
        'diagnostics': attempts, 'original_diagnostics': original,
        'restart_diagnostics': attempts[len(original):],
        'selected_start': int(chosen['start_id']),
        'selected_restart_id': chosen['restart_id'],
        'selected_basin_id': chosen.get('basin_id'),
        'start_nll_spread': original_nll_spread,
        'original_start_nll_spread': original_nll_spread,
        'original_start_objective_spread': original_objective_spread,
        'original_acceptance': bool(original_acceptance),
        'legacy_original_acceptance_applicable': legacy_diagnostic_applicable,
        'accepted': bool(original_acceptance) if legacy_diagnostic_applicable else False,
        'revised_accepted': revised_accepted,
        'original_distinct_basin_count': original_basin_count,
        'distinct_basin_count': len(distinct_centers(attempts)),
        'selected_basin_replicates': int(chosen_basin_replicates),
        'projected_gradient_per_observation': chosen['projected_gradient_per_observation'],
        'persistence_bound': float(persistence_bound),
        'measurement_mode': measurement_mode,
        'penalty_strength': float(penalty_strength),
        'restart_seed': int(restart_seed),
        'restart_count': int(restart_count),
    }


@njit(cache=False)
def _filter(y, load, params, fixed_r):
    """Scalar innovation form is exact because R is diagonal."""
    mu = params[:3]
    rho = np.tanh(params[3:6])
    q = np.exp(params[6:9])
    r = fixed_r if len(params) == 9 else np.exp(params[9:])
    state = mu.copy(); p = np.diag(q * np.cosh(params[3:6])**2)
    total = 0.0
    for t in range(len(y)):
        for i in range(y.shape[1]):
            h = load[i]
            b = p @ h
            variance = h @ b + r[i]
            if not np.isfinite(variance) or variance <= 0.:
                return np.inf, np.full(3, np.nan), np.full((3, 3), np.nan)
            innovation = y[t, i] - h @ state
            total += .5 * (np.log(2*np.pi) + np.log(variance) + innovation**2 / variance)
            state += b * (innovation / variance)
            p -= np.outer(b, b) / variance
        p = .5 * (p + p.T)
        if t < len(y)-1:
            state = mu + rho * (state - mu)
            p = rho[:, None] * p * rho[None, :] + np.diag(q)
    return total, state, p


def filter_state(y, load, params, measurement_mode='estimated', fixed_measurement=None):
    """Return exact stationary-initialized negative log likelihood and last filter."""
    y = np.ascontiguousarray(y, dtype=float)
    load = np.ascontiguousarray(load, dtype=float)
    params = np.asarray(params, dtype=float)
    if measurement_mode not in {'estimated', 'fixed', 'regularized'}:
        raise ValueError(f'unknown measurement covariance mode: {measurement_mode}')
    if y.ndim != 2 or y.shape[1] != len(load) or load.shape[1] != 3:
        raise ValueError('expected T by D yields and D by 3 loadings')
    if len(params) != parameter_count(len(load), measurement_mode) or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError('invalid parameter length or input observations')
    if not np.isfinite(params).all():
        raise ValueError('parameters must be finite')
    if measurement_mode == 'fixed':
        fixed_measurement = np.asarray(fixed_measurement, dtype=float)
        if (fixed_measurement.shape != (len(load),) or not np.isfinite(fixed_measurement).all()
                or np.any(fixed_measurement <= 0)):
            raise ValueError('fixed mode requires positive finite measurement variances')
    else:
        fixed_measurement = np.empty(0, dtype=float)
    result = _filter(y, load, params, fixed_measurement)
    if not np.isfinite(result[0]) or not np.isfinite(result[1]).all() or not np.isfinite(result[2]).all():
        raise FloatingPointError('Kalman filter produced a non-finite likelihood or state')
    return result


@njit(cache=False)
def _likelihood_gradient(y, load, params, fixed_r):
    """Forward sensitivity recursion: exact gradient of stationary Gaussian NLL."""
    mu = params[:3]; rho = np.tanh(params[3:6])
    q = np.exp(params[6:9])
    fixed_measurement = len(params) == 9
    r = fixed_r if fixed_measurement else np.exp(params[9:])
    npar = len(params)
    state = mu.copy(); p = np.diag(q * np.cosh(params[3:6])**2)
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
                df = h[0]*db[0] + h[1]*db[1] + h[2]*db[2]
                if not fixed_measurement and k == 9+i:
                    df += r[i]
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


def likelihood_gradient(y, load, params, measurement_mode='estimated', fixed_measurement=None):
    """Return raw-NLL and gradient for estimated, fixed, or penalized-diagonal R."""
    params = np.asarray(params, dtype=float)
    if measurement_mode not in {'estimated', 'fixed', 'regularized'}:
        raise ValueError(f'unknown measurement covariance mode: {measurement_mode}')
    if len(params) != parameter_count(np.asarray(load).shape[0], measurement_mode):
        raise ValueError('parameter length does not match measurement covariance mode')
    if measurement_mode == 'fixed':
        fixed_r = np.asarray(fixed_measurement, dtype=float)
        if (fixed_r.shape != (np.asarray(load).shape[0],) or not np.isfinite(fixed_r).all()
                or np.any(fixed_r <= 0)):
            raise ValueError('fixed mode requires positive finite measurement variances')
    else:
        fixed_r = np.empty(0, dtype=float)
    return _likelihood_gradient(np.ascontiguousarray(y, dtype=float),
                                np.ascontiguousarray(load, dtype=float), params, fixed_r)


def unpack(params, measurement_mode='estimated', fixed_measurement=None):
    params = np.asarray(params, dtype=float)
    if params.ndim != 1 or len(params) < 9 or not np.isfinite(params).all():
        raise ValueError('parameters must be a finite vector with at least nine entries')
    if measurement_mode == 'fixed':
        r = np.asarray(fixed_measurement, dtype=float)
        if len(params) != 9 or r.ndim != 1 or not np.isfinite(r).all() or np.any(r <= 0):
            raise ValueError('fixed mode requires nine parameters and positive finite measurement variances')
    elif measurement_mode in {'estimated', 'regularized'}:
        if len(params) <= 9:
            raise ValueError('estimated measurement mode requires log-variance parameters')
        r = np.exp(params[9:])
    else:
        raise ValueError(f'unknown measurement covariance mode: {measurement_mode}')
    return params[:3], np.tanh(params[3:6]), np.exp(params[6:9]), r


def forecast_moments(state, covariance, load, params, horizon,
                     measurement_mode='estimated', fixed_measurement=None):
    """Conditional Gaussian moments, without Monte Carlo or eigenvalue clipping."""
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    mu, rho, q, r = unpack(params, measurement_mode, fixed_measurement)
    ah = rho ** horizon
    mean_f = mu + ah * (state - mu)
    cov_f = ah[:, None] * covariance * ah[None, :]
    cov_f += np.diag(q * sum(rho ** (2 * k) for k in range(horizon)))
    covariance_y = load @ cov_f @ load.T + np.diag(r)
    return load @ mean_f, (covariance_y + covariance_y.T) / 2


def boundary_profiles(y, load, params, persistence_bound=3., measurement_mode='estimated',
                      fixed_measurement=None):
    """One-coordinate inward/outward checks, diagnostic only; no change to bounds."""
    bounds = parameter_bounds(y.shape[1], persistence_bound, measurement_mode)
    value = filter_state(y, load, params, measurement_mode, fixed_measurement)[0]
    active = np.flatnonzero((params <= bounds[:, 0]+1e-6) | (params >= bounds[:, 1]-1e-6))
    profiles = []
    for j in active:
        inward = params.copy(); outward = params.copy()
        step = .05 if j < 6 else .5
        direction = 1 if params[j] <= bounds[j, 0]+1e-6 else -1
        inward[j] += direction*step; outward[j] -= direction*step
        profiles.append({'parameter_index': int(j), 'inward_step': direction*step,
                         'inward_delta_nll': float(filter_state(
                             y, load, inward, measurement_mode, fixed_measurement)[0]-value),
                         'outward_delta_nll': float(filter_state(
                             y, load, outward, measurement_mode, fixed_measurement)[0]-value)})
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
