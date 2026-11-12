"""Regression tests that expose scientific defects in the first executed run."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import digamma, polygamma
from src import models


class MathematicalRegressions(unittest.TestCase):
    def test_sv_intercept_draw_matches_stationary_initial_density(self):
        h = np.array([-2.0, -1.5, -2.5, -2.2])
        phi, q = 0.8, 0.3
        # Full conditional in c for N(c/(1-phi), q/(1-phi^2)).
        precision = 1 / 25 + (len(h)-1) / q + (1+phi) / (q*(1-phi))
        rhs = np.sum(h[1:] - phi*h[:-1]) / q + h[0]*(1+phi)/q
        expected_mean, expected_sd = rhs/precision, np.sqrt(1/precision)
        class RecordingRng:
            def normal(self, mean, sd):
                np.testing.assert_allclose([mean, sd], [expected_mean, expected_sd], rtol=1e-12)
                return mean
        models._draw_c(h, phi, q, RecordingRng())

    def test_sv_sampler_uses_stationary_variance_full_conditional(self):
        x = np.random.default_rng(3).normal(size=30)
        with patch.object(models, 'sigma2_posterior_parameters', wraps=models.sigma2_posterior_parameters) as call:
            models.fit_sv_chain(x, np.random.default_rng(4), warmup=1, draws=2)
        self.assertEqual(call.call_count, 3)

    def test_mixture_matches_log_chisquare_moments(self):
        mean = np.sum(models.MIX_W * models.MIX_M)
        variance = np.sum(models.MIX_W*(models.MIX_V+models.MIX_M**2))-mean**2
        self.assertAlmostEqual(float(mean), float(digamma(0.5)+np.log(2)), delta=0.02)
        self.assertAlmostEqual(float(variance), float(polygamma(1,0.5)), delta=0.08)

    def test_state_forecast_propagates_filtered_covariance_by_A_to_h(self):
        load = models.ns_loadings(np.array([.25,.5,1,2,5,7,10,30]), .5)
        rho = np.array([.8,.7,.6]); q=np.array([.01,.02,.03]); r=np.full(8,.001)
        params=np.r_[np.zeros(3),np.arctanh(rho),np.log(q),np.log(r)]
        y=np.random.default_rng(5).normal(size=(15,8))
        _,_,p=models.kalman_filter(y,load,params,return_state=True)
        h=4; a=np.diag(rho**h)
        expected=a@p@a.T+np.diag([q[j]*sum(rho[j]**(2*k) for k in range(h)) for j in range(3)])
        expected=models.project_covariance(load@expected@load.T+np.diag(r))
        _,cov=models.state_space_forecast(y,load,params,h,np.random.default_rng(6),10)
        np.testing.assert_allclose(cov,expected,rtol=1e-10,atol=1e-12)

    def test_constrained_ar_intercept_is_refit_after_slope_constraint(self):
        x=np.arange(40,dtype=float)
        a,b,q=models.ar1_fit(x)
        self.assertAlmostEqual(a, float(np.mean(x[1:]-b*x[:-1])), places=10)


if __name__=='__main__':
    unittest.main()
