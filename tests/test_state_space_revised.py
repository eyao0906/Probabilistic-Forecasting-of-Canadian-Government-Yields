"""Independent acceptance fixtures for the revised M3 only (stdlib unittest)."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
from tests.support import scratch_root


@contextmanager
def isolated_m3_workspace():
    """Run unchanged, output-confined M3 source against disposable fixture inputs.

    The production runner intentionally rejects output paths outside its own
    results tree. Load an exact source copy rooted in scratch instead of
    weakening that guard or patching the scientific implementation.
    """
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(dir=scratch_root(), prefix='mrp-m3-') as temp:
        workspace = Path(temp)
        inputs = [
            'src/state_space_revised.py',
            'tests/test_state_space_revised.py',
            'config/experiment.json',
            'data/processed/canada_weekly.csv',
            'results/state_space_revised/verify_outputs.py',
        ]
        inputs.extend(path.relative_to(root).as_posix()
                      for path in sorted((root/'results/validated_baselines').iterdir())
                      if path.is_file())
        for name in inputs:
            target = workspace/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/name, target)
        spec = importlib.util.spec_from_file_location(
            'isolated_revised_m3', workspace/'src/state_space_revised.py')
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        previous_path = sys.path[:]
        try:
            yield module, workspace/'results/state_space_revised/test_fixture'
        finally:
            # The independent verifier prepends its ROOT to sys.path on import.
            # Do not leave a deleted scratch root in the process import path.
            sys.path[:] = previous_path


class RevisedStateSpaceTests(unittest.TestCase):
    def test_exact_forecast_propagates_filtered_covariance_once_per_side(self):
        self.assertIsNotNone(importlib.util.find_spec('src.state_space_revised'),
                             'Revised independent M3 module has not been implemented')
        from src.state_space_revised import forecast_moments
        load = np.array([[1., .6, .2], [1., .3, .4]])
        mu = np.array([2., -.2, .1]); rho = np.array([.8, -.4, .6])
        q = np.array([.02, .03, .01]); r = np.array([.004, .006])
        params = np.r_[mu, np.arctanh(rho), np.log(q), np.log(r)]
        state = np.array([1., .5, -.3])
        p = np.array([[.3, .04, -.02], [.04, .2, .03], [-.02, .03, .1]])
        for h in [1, 4]:
            m = state.copy(); c = p.copy(); a = np.diag(rho)
            for _ in range(h):
                m = mu + a @ (m - mu)
                c = a @ c @ a.T + np.diag(q)
            actual_mean, actual_cov = forecast_moments(state, p, load, params, h)
            np.testing.assert_allclose(actual_mean, load @ m, atol=1e-14)
            np.testing.assert_allclose(actual_cov, load @ c @ load.T + np.diag(r), atol=1e-14)
            ah = np.linalg.matrix_power(a, h)
            wrong = ah @ ah @ p @ ah.T @ ah.T
            wrong += sum((np.linalg.matrix_power(a, k) @ np.diag(q) @ np.linalg.matrix_power(a.T, k)
                          for k in range(h)))
            self.assertGreater(np.max(abs(actual_cov - (load @ wrong @ load.T + np.diag(r)))), 1e-4)
        with self.assertRaises(ValueError):
            forecast_moments(state, p, load, params, 0)

    def test_filter_and_stationary_likelihood_match_joint_gaussian_conditioning(self):
        from src import state_space_revised as revised
        self.assertTrue(hasattr(revised, 'filter_state'), 'independent filter missing')
        rng = np.random.default_rng(401)
        load = np.array([[1., .6, .2], [1., .3, .4], [.2, -.1, .9]])
        mu = np.array([2., -.2, .1]); rho = np.array([.8, -.4, .6])
        q = np.array([.02, .03, .01]); r = np.array([.004, .006, .003])
        params = np.r_[mu, np.arctanh(rho), np.log(q), np.log(r)]
        y = rng.normal(size=(7, 3)) * .1 + load @ mu
        # Independent full joint Gaussian distribution, no recursive filter.
        stationary = np.diag(q / (1 - rho ** 2)); n = len(y); d = y.shape[1]
        joint = np.empty((n*d, n*d)); cross = np.empty((3, n*d))
        for i in range(n):
            cross[:, i*d:(i+1)*d] = np.diag(rho ** (n-1-i)) @ stationary @ load.T
            for j in range(n):
                block = load @ np.diag(rho ** abs(i-j)) @ stationary @ load.T
                joint[i*d:(i+1)*d, j*d:(j+1)*d] = block + (np.diag(r) if i == j else 0)
        error = (y - load @ mu).ravel()
        sign, logdet = np.linalg.slogdet(joint); self.assertEqual(sign, 1)
        expected_nll = .5 * (n*d*np.log(2*np.pi) + logdet + error @ np.linalg.solve(joint, error))
        expected_mean = mu + cross @ np.linalg.solve(joint, error)
        expected_cov = stationary - cross @ np.linalg.solve(joint, cross.T)
        actual_nll, actual_mean, actual_cov = revised.filter_state(y, load, params)
        self.assertAlmostEqual(actual_nll, expected_nll, places=10)
        np.testing.assert_allclose(actual_mean, expected_mean, atol=1e-12)
        np.testing.assert_allclose(actual_cov, expected_cov, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(actual_cov).min(), 0)

    def test_analytic_likelihood_gradient_matches_central_difference(self):
        from src import state_space_revised as revised
        self.assertTrue(hasattr(revised, 'likelihood_gradient'), 'likelihood gradient missing')
        rng = np.random.default_rng(99)
        load = rng.normal(size=(8, 3))
        params = np.r_[[2., -.2, .1], np.arctanh([.8, -.4, .96]),
                       np.log([.02, .03, .01]), np.log(np.linspace(.004, .01, 8))]
        y = rng.normal(size=(31, 8)) * .1 + load @ params[:3]
        value, gradient = revised.likelihood_gradient(y, load, params)
        self.assertAlmostEqual(value, revised.filter_state(y, load, params)[0], places=9)
        finite = np.empty(len(params))
        for j in range(len(params)):
            step = 1e-5 * max(1., abs(params[j])); direction = np.eye(len(params))[j] * step
            finite[j] = (revised.filter_state(y, load, params+direction)[0] -
                         revised.filter_state(y, load, params-direction)[0]) / (2*step)
        np.testing.assert_allclose(gradient, finite, rtol=2e-6, atol=2e-6)

    def test_multistart_mle_has_small_projected_gradient_not_just_success(self):
        from src import state_space_revised as revised
        self.assertTrue(hasattr(revised, 'fit_mle'), 'revised multistart MLE missing')
        rng = np.random.default_rng(117)
        load = np.array([[1., .9, .1], [1., .2, .4], [1., .01, .05],
                         [.2, -.3, .9]])
        mu = np.array([2., -.5, .2]); rho = np.array([.75, .55, .35])
        q = np.array([.02, .03, .025]); r = np.array([.005, .003, .004, .006])
        state = mu.copy(); y = []
        for _ in range(260):
            state = mu+rho*(state-mu)+rng.normal(size=3)*np.sqrt(q)
            y.append(load@state+rng.normal(size=4)*np.sqrt(r))
        y = np.array(y)
        fit = revised.fit_mle(y, load, n_starts=2, maxiter=600)
        self.assertTrue(fit['accepted'], fit['diagnostics'])
        self.assertEqual(len(fit['diagnostics']), 2)
        self.assertLess(fit['projected_gradient_per_observation'], 1e-5)
        self.assertLess(fit['start_nll_spread'], .02)
        self.assertLess(fit['nll'], min(d['initial_nll'] for d in fit['diagnostics']))
        self.assertTrue(all(d['elapsed_s'] > 0 for d in fit['diagnostics']))

    def test_design_uses_only_initial_lambda_and_common_expanding_windows(self):
        from src import state_space_revised as revised
        self.assertTrue(hasattr(revised, 'prepare_design'), 'training design missing')
        from src.experiment import read_panel
        from pathlib import Path
        import json
        root = Path(__file__).resolve().parents[1]
        dates, y = read_panel(root/'data/processed/canada_weekly.csv')
        config = json.loads((root/'config/experiment.json').read_text())
        design = revised.prepare_design(dates, y, config)
        self.assertEqual(len(design['origin_indices']), 130)
        self.assertEqual(len(design['selected']), 260)
        changed = y.copy()
        for t, day in enumerate(dates):
            if day.year >= 2016:
                changed[t] += 200.
        other = revised.prepare_design(dates, changed, config)
        self.assertEqual(design['lambda_per_year'], other['lambda_per_year'])
        self.assertEqual(design['initial_training_n'], 835)
        self.assertLessEqual(dates[design['initial_indices'][-1]].year, 2015)
        for t, h, target in design['selected']:
            self.assertEqual(target, t+h)
            self.assertLess(dates[t], dates[target])
            self.assertLessEqual(dates[target].year, 2025)

    def test_forecast_metrics_use_analytic_moments_and_1024_scenarios(self):
        from src import state_space_revised as revised
        self.assertTrue(hasattr(revised, 'evaluate_forecast'), 'analytic scoring missing')
        mean = np.linspace(1., 2., 8); covariance = np.eye(8)*.02
        observed = mean+.1; current = mean-.05
        result = revised.evaluate_forecast(mean, covariance, observed, current, 99)
        np.testing.assert_array_equal(result['means'], mean)
        np.testing.assert_array_equal(result['covariances'], covariance)
        self.assertEqual(result['scenarios'].shape, (1024, 8))
        self.assertAlmostEqual(result['metrics']['squared_error_average'], .01)
        expected_dss = 8*np.log(.02)+8*.01/.02
        self.assertAlmostEqual(result['metrics']['dss'], expected_dss)
        self.assertEqual(result['metrics']['coverage90'], 1.)
        self.assertAlmostEqual(result['metrics']['width90'], 2*1.6448536269514722*np.sqrt(.02))
        again = revised.evaluate_forecast(mean, covariance, observed, current, 99)
        np.testing.assert_array_equal(result['scenarios'], again['scenarios'])
        altered = revised.evaluate_forecast(mean, covariance, observed, current, 100)
        np.testing.assert_array_equal(result['means'], altered['means'])
        self.assertFalse(np.array_equal(result['scenarios'], altered['scenarios']))

    def test_end_to_end_smoke_saves_moments_keys_targets_and_fit_diagnostics(self):
        import json
        with isolated_m3_workspace() as (revised, out):
            self.assertTrue(hasattr(revised, 'run_experiment'), 'revised experiment runner missing')
            metadata = revised.run_experiment(phase='smoke', output_dir=out, origin_positions=[0], n_starts=2)
            self.assertEqual(metadata['status'], 'accepted_restricted_partial_smoke')
            self.assertEqual(metadata['forecast_records'], 2)
            # Close the NPZ archive before TemporaryDirectory cleanup on Windows.
            with np.load(out/'forecasts.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['scenarios'].shape, (2, 1024, 8))
                self.assertEqual(saved['keys'].tolist(), [['2016-01-08', '1', 'M3'], ['2016-01-08', '4', 'M3']])
                self.assertEqual(saved['targets'].shape, (2, 8))
                self.assertTrue(np.isfinite(saved['covariances']).all())
            diagnostics = json.loads((out/'optimizer_diagnostics.json').read_text())
            self.assertEqual(len(diagnostics), 1)
            self.assertEqual(len(diagnostics[0]['diagnostics']), 2)
            self.assertTrue((out/'scores.csv').exists())
            self.assertTrue((out/'origin_audit.csv').exists())
        self.assertFalse(out.parents[2].exists(), 'Temporary M3 workspace was not removed')

    def test_independent_saved_archive_verifier_checks_real_smoke(self):
        import importlib.util
        previous_path = sys.path[:]
        with isolated_m3_workspace() as (revised, fixture):
            path = fixture.parent/'verify_outputs.py'
            self.assertTrue(path.exists(), 'independent archive verifier not implemented')
            spec = importlib.util.spec_from_file_location('verify_revised_m3', path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            revised.run_experiment(phase='smoke', output_dir=fixture, origin_positions=[0], n_starts=2)
            report = module.verify(fixture, expected_origins=1)
            self.assertEqual(report['status'], 'verified_restricted_partial')
            self.assertEqual(report['records_verified'], 2)
            self.assertLess(report['independent_batch_nll_max_abs_difference'], 1e-7)
            self.assertTrue(report['accepted_baselines_unchanged'])
        self.assertFalse(fixture.parents[2].exists(), 'Temporary M3 workspace was not removed')
        self.assertEqual(sys.path, previous_path)


if __name__ == '__main__':
    unittest.main()
