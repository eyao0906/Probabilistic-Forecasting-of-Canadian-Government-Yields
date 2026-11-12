"""Independent saved-artifact audit; no calls to revised filter or forecasting.

From project root:
.venv/Scripts/python.exe -B results/state_space_revised/verify_outputs.py
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import time
from collections import Counter

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiment import read_panel, select_origins
from src.models import ns_loadings
from datetime import date


def batch_filter(y, load, params):
    """Reference multivariate innovations with Joseph covariance updates.

    Intentionally distinct from production scalar observations and sensitivities.
    """
    mu = params[:3]; a = np.diag(np.tanh(params[3:6]))
    q = np.diag(np.exp(params[6:9])); r = np.diag(np.exp(params[9:]))
    state = mu.copy(); covariance = np.diag(np.diag(q)/(1-np.diag(a)**2))
    nll = 0.
    for t, obs in enumerate(y):
        if t:
            state = mu+a@(state-mu)
            covariance = a@covariance@a.T+q
        innovation = obs-load@state
        variance = load@covariance@load.T+r
        sign, logdet = np.linalg.slogdet(variance)
        assert sign == 1
        nll += .5*(len(obs)*np.log(2*np.pi)+logdet+innovation@np.linalg.solve(variance, innovation))
        gain = np.linalg.solve(variance, load@covariance).T
        state = state+gain@innovation
        ikz = np.eye(3)-gain@load
        covariance = ikz@covariance@ikz.T+gain@r@gain.T
    return nll, state, covariance


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def verify(out=None, expected_origins=130):
    started = time.perf_counter()
    out = Path(out) if out is not None else ROOT/'results/state_space_revised'
    metadata = json.loads((out/'metadata.json').read_text())
    fits = json.loads((out/'optimizer_diagnostics.json').read_text())
    scores = read_csv(out/'scores.csv'); audit = read_csv(out/'origin_audit.csv')
    summary = read_csv(out/'summary.csv'); maturities = read_csv(out/'maturity_by_origin.csv')
    risk = read_csv(out/'risk_scenarios.csv')
    with np.load(out/'forecasts.npz', allow_pickle=False) as archive:
        saved = {k: archive[k].copy() for k in archive.files}
    assert len(fits) == expected_origins
    n = expected_origins*2
    assert metadata['n_origins'] == expected_origins and metadata['forecast_records'] == n
    assert saved['scenarios'].shape == (n, 1024, 8)
    assert saved['means'].shape == (n, 8) and saved['covariances'].shape == (n, 8, 8)
    assert saved['targets'].shape == (n, 8) and saved['keys'].shape == (n, 3)
    assert len(scores) == len(audit) == len(risk) == n
    assert len(maturities) == n*8
    assert len({tuple(k) for k in saved['keys']}) == n
    for name in ['means', 'covariances', 'targets', 'scenarios']:
        assert np.isfinite(saved[name]).all()
    for name, expected_hash in metadata['source_and_input_sha256'].items():
        # Tests may be expanded after estimation; report that separately, not a
        # false identity claim. Executed scientific source/input must match.
        if name.replace('\\', '/') != 'tests/test_state_space_revised.py':
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected_hash, name
    baseline_dir = ROOT/'results/validated_baselines'
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_dir.iterdir() if p.is_file()}
    assert hashes == metadata['baseline_hashes_before'] == metadata['baseline_hashes_after']
    dates, y = read_panel(ROOT/'data/processed/canada_weekly.csv')
    config = json.loads((ROOT/'config/experiment.json').read_text())
    common = select_origins(dates, date(2016, 1, 1), date(2025, 12, 31), [1, 4], 4)
    expected_keys = [(dates[t].isoformat(), str(h), 'M3') for t, h, target in common]
    if expected_origins == 130:
        assert [tuple(k) for k in saved['keys']] == expected_keys
    date_index = {d.isoformat(): i for i, d in enumerate(dates)}
    first = next(i for i, d in enumerate(dates) if d >= date(2000, 1, 1))
    load = ns_loadings(saved['maturities'], float(saved['lambda_per_year']))
    with np.load(baseline_dir/'forecasts.npz', allow_pickle=False) as baselines:
        common_targets = {(k[0], k[1]): target for k, target in zip(baselines['keys'], baselines['targets'])}
    nll_diff = []; mean_diff = []; cov_diff = []; active = Counter(); profiles = []
    maximum_gradient = 0.; max_start_spread = 0.; fit_by_date = {}; start_sensitivity = []
    key_index = {(key[0], int(key[1])): k for k, key in enumerate(saved['keys'])}
    for fit in fits:
        t = date_index[fit['origin_date']]; training = y[first:t+1]
        assert len(training) == fit['n_training']
        assert fit['latest_fit_date'] == fit['origin_date']
        params = np.array(fit['params'])
        nll, state, covariance = batch_filter(training, load, params)
        nll_diff.append(abs(nll-fit['nll']))
        np.testing.assert_allclose(state, fit['filtered_state'], atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(covariance, fit['filtered_covariance'], atol=2e-10, rtol=2e-10)
        assert abs(nll-fit['nll']) < 1e-7
        assert len(fit['diagnostics']) == metadata['n_starts']
        bounds = np.array(metadata['parameter_bounds'])
        for attempt in fit['diagnostics']:
            p = np.array(attempt['params']); gradient = np.array(attempt['gradient'])
            assert np.all(p >= bounds[:, 0]-1e-10) and np.all(p <= bounds[:, 1]+1e-10)
            projected = gradient.copy()
            projected[(p <= bounds[:, 0]+1e-7) & (gradient > 0)] = 0.
            projected[(p >= bounds[:, 1]-1e-7) & (gradient < 0)] = 0.
            pg = np.max(abs(projected))/len(training)
            assert pg < 1e-5
            maximum_gradient = max(maximum_gradient, float(pg))
            candidate_nll, candidate_state, candidate_p = batch_filter(training, load, p)
            assert abs(candidate_nll-attempt['nll']) < 1e-7
            nll_diff.append(abs(candidate_nll-attempt['nll']))
            sensitivity = {'origin_date': fit['origin_date'], 'start_id': attempt['start_id'],
                           'nll_gap_to_selected': float(attempt['nll']-fit['nll']),
                           'selected_start': fit['selected_start'], 'fit_accepted': fit['accepted']}
            for h in [1, 4]:
                m_candidate = candidate_state.copy(); p_candidate = candidate_p.copy()
                a_candidate = np.diag(np.tanh(p[3:6])); q_candidate = np.diag(np.exp(p[6:9]))
                for _ in range(h):
                    m_candidate = p[:3]+a_candidate@(m_candidate-p[:3])
                    p_candidate = a_candidate@p_candidate@a_candidate.T+q_candidate
                k = key_index[(fit['origin_date'], h)]
                sensitivity['h'+str(h)+'_mean_max_abs_difference'] = float(np.max(abs(load@m_candidate-saved['means'][k])))
                c_candidate = load@p_candidate@load.T+np.diag(np.exp(p[9:]))
                sensitivity['h'+str(h)+'_covariance_max_abs_difference'] = float(np.max(abs(c_candidate-saved['covariances'][k])))
            start_sensitivity.append(sensitivity)
        spread = np.ptp([d['nll'] for d in fit['diagnostics']])
        assert fit['accepted'] == bool(spread < .02)
        max_start_spread = max(max_start_spread, float(spread))
        active.update(fit['diagnostics'][fit['selected_start']]['active_bound_indices'])
        for profile in fit['boundary_profiles']:
            assert profile['inward_delta_nll'] > -1e-6
            profiles.append(profile)
        fit_by_date[fit['origin_date']] = (params, state, covariance)
    for k, row in enumerate(scores):
        origin, horizon, model = saved['keys'][k]; h = int(horizon)
        assert (row['origin_date'], row['horizon'], row['model']) == (origin, horizon, model)
        t = date_index[origin]; target = t+h
        assert saved['target_dates'][k] == dates[target].isoformat() == row['target_date']
        np.testing.assert_array_equal(saved['targets'][k], y[target])
        np.testing.assert_array_equal(saved['targets'][k], common_targets[(origin, horizon)])
        assert audit[k]['latest_fit_date'] == origin
        assert int(audit[k]['n_training']) == len(y[first:t+1])
        assert audit[k]['train_sha256'] == hashlib.sha256(y[first:t+1].tobytes()).hexdigest()
        params, m, p = fit_by_date[origin]; mu = params[:3]
        a = np.diag(np.tanh(params[3:6])); q = np.diag(np.exp(params[6:9])); r = np.diag(np.exp(params[9:]))
        for _ in range(h):
            m = mu+a@(m-mu); p = a@p@a.T+q
        mean = load@m; covariance = load@p@load.T+r
        mean_diff.append(float(np.max(abs(mean-saved['means'][k]))))
        cov_diff.append(float(np.max(abs(covariance-saved['covariances'][k]))))
        np.testing.assert_allclose(mean, saved['means'][k], atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(covariance, saved['covariances'][k], atol=2e-10, rtol=2e-10)
        # Reproduce exact saved scenarios from the specified conditional Gaussian.
        regenerated = np.random.default_rng(20261027+t*17+h).multivariate_normal(saved['means'][k], saved['covariances'][k], size=1024)
        np.testing.assert_array_equal(regenerated, saved['scenarios'][k])
        error = saved['targets'][k]-saved['means'][k]; covariance = saved['covariances'][k]
        _, logdet = np.linalg.slogdet(covariance)
        dss = logdet+error@np.linalg.solve(covariance, error)
        sd = np.sqrt(np.diag(covariance)); hits = abs(error) <= norm.ppf(.95)*sd
        scenarios = saved['scenarios'][k]
        es = np.linalg.norm(scenarios-saved['targets'][k], axis=1).mean()-.5*pdist(scenarios).mean()
        tri = np.triu_indices(8, 1)
        spread_observed = abs(saved['targets'][k, :, None]-saved['targets'][k, None, :])**.5
        spread_simulated = np.mean(abs(scenarios[:, :, None]-scenarios[:, None, :])**.5, axis=0)
        vs = np.sum((spread_observed[tri]-spread_simulated[tri])**2)
        expected = {'dss': dss, 'energy_score': es, 'variogram_score': vs,
                    'gaussian_negative_log_score': .5*(8*np.log(2*np.pi)+dss),
                    'squared_error_average': np.mean(error**2), 'mae': np.mean(abs(error)),
                    'coverage90': hits.mean(), 'width90': np.mean(2*norm.ppf(.95)*sd)}
        for name, value in expected.items():
            np.testing.assert_allclose(float(row[name]), value, rtol=1e-10, atol=1e-10)
        for j in range(8):
            mr = maturities[k*8+j]
            assert mr['origin_date'] == origin and mr['horizon'] == horizon
            np.testing.assert_allclose(float(mr['squared_error']), error[j]**2, atol=1e-12)
            np.testing.assert_allclose(float(mr['pit']), norm.cdf(error[j]/sd[j]), atol=1e-12)
        delta = ((scenarios-y[t])/100)@(saved['maturities']/8)
        rr = risk[k]
        for p_tail, suffix in [(.95, '95'), (.99, '99')]:
            threshold = np.quantile(delta, p_tail)
            np.testing.assert_allclose(float(rr['VaR'+suffix]), threshold, atol=1e-12)
            np.testing.assert_allclose(float(rr['ES'+suffix]), delta[delta >= threshold].mean(), atol=1e-12)
    for row in summary:
        subset = [s for s in scores if s['horizon'] == row['horizon']]
        assert int(row['n']) == len(subset)
        for key in ['dss', 'gaussian_negative_log_score', 'energy_score', 'variogram_score', 'mae', 'coverage90', 'width90']:
            np.testing.assert_allclose(float(row[key]), np.mean([float(s[key]) for s in subset]), atol=1e-11)
        rmse = np.sqrt(np.mean([float(s['squared_error_average']) for s in subset]))
        np.testing.assert_allclose(float(row['pooled_rmse']), rmse, atol=1e-12)
    # At representative full-length windows independently difference the batch
    # likelihood, including active persistence/variance boundaries. Also compare
    # conditional forecasts from every starting value, not just their NLLs.
    pilot_checks = []
    for position in sorted({0, len(fits)//2, len(fits)-1}):
        fit = fits[position]; t = date_index[fit['origin_date']]; training = y[first:t+1]
        params, chosen_state, chosen_p = fit_by_date[fit['origin_date']]
        chosen_diagnostic = fit['diagnostics'][fit['selected_start']]
        numerical = np.empty(len(params))
        for j in range(len(params)):
            step = 1e-5*max(1., abs(params[j])); direction = np.eye(len(params))[j]*step
            numerical[j] = (batch_filter(training, load, params+direction)[0] -
                            batch_filter(training, load, params-direction)[0])/(2*step)
        gradient_error = float(np.max(abs(numerical-np.array(chosen_diagnostic['gradient']))))
        assert gradient_error < 2e-4
        mean_spread = 0.; covariance_spread = 0.
        for attempt in fit['diagnostics']:
            assert attempt['nll'] <= attempt['initial_nll']+1e-8
            candidate = np.array(attempt['params'])
            _, candidate_state, candidate_p = batch_filter(training, load, candidate)
            mu = candidate[:3]; a = np.diag(np.tanh(candidate[3:6])); q = np.diag(np.exp(candidate[6:9]))
            for h in [1, 4]:
                m = candidate_state.copy(); p = candidate_p.copy()
                for _ in range(h):
                    m = mu+a@(m-mu); p = a@p@a.T+q
                record_index = next(k for k, key in enumerate(saved['keys'])
                                    if key[0] == fit['origin_date'] and key[1] == str(h))
                mean_spread = max(mean_spread, float(np.max(abs(load@m-saved['means'][record_index]))))
                c = load@p@load.T+np.diag(np.exp(candidate[9:]))
                covariance_spread = max(covariance_spread, float(np.max(abs(c-saved['covariances'][record_index]))))
        assert mean_spread < 1e-4 and covariance_spread < 1e-5
        pilot_checks.append({'origin_date': fit['origin_date'],
                             'independent_central_difference_gradient_max_abs_error': gradient_error,
                             'across_start_forecast_mean_max_abs_difference': mean_spread,
                             'across_start_forecast_covariance_max_abs_difference': covariance_spread})
    with (out/'start_sensitivity.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(start_sensitivity[0]))
        writer.writeheader(); writer.writerows(start_sensitivity)
    all_fits_accepted = all(f['accepted'] for f in fits)
    status = ('verified_restricted_full' if expected_origins == 130 else 'verified_restricted_partial') if all_fits_accepted else 'rejected_optimizer_start_sensitivity'
    report = {'status': status, 'numerical_implementation_checks': 'passed',
              'fit_accept_count': sum(f['accepted'] for f in fits),
              'fit_reject_origins': [f['origin_date'] for f in fits if not f['accepted']],
              'origins_verified': len(fits), 'records_verified': n, 'scenarios_per_record': 1024,
              'independent_batch_nll_max_abs_difference': float(max(nll_diff)),
              'independent_mean_max_abs_difference': max(mean_diff),
              'independent_covariance_max_abs_difference': max(cov_diff),
              'min_predictive_eigenvalue': float(np.linalg.eigvalsh(saved['covariances']).min()),
              'maximum_all_start_projected_gradient_per_observation': maximum_gradient,
              'start_likelihoods_independently_verified': len(start_sensitivity),
              'maximum_start_nll_spread': max_start_spread,
              'maximum_start_forecast_mean_difference': {str(h): max(r['h'+str(h)+'_mean_max_abs_difference'] for r in start_sensitivity) for h in [1, 4]},
              'maximum_start_forecast_covariance_difference': {str(h): max(r['h'+str(h)+'_covariance_max_abs_difference'] for r in start_sensitivity) for h in [1, 4]},
              'pilot_independent_gradient_and_start_checks': pilot_checks,
              'active_bound_origin_counts': dict(sorted(active.items())),
              'boundary_profiles_checked': len(profiles),
              'outward_profiles_improving_likelihood': sum(p['outward_delta_nll'] < 0 for p in profiles),
              'accepted_baselines_unchanged': True, 'common_targets_equal_accepted_baselines': True,
              'exact_scenarios_reproduced': True, 'all_scores_and_pooled_summaries_recomputed': True,
              'elapsed_s': time.perf_counter()-started,
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
                                if p.is_file() and p.suffix in ['.npz', '.csv']},
              'limitations': ['Validation applies to explicitly bounded stationary diagonal Gaussian model only.',
                              'Active-bound outward likelihood improvements rule out an unrestricted-MLE claim.',
                              'No publication-time real-time or live trading claim.']}
    (out/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
