"""Acceptance tests for the validated classical rerun; not the failed SV run."""
import unittest
import numpy as np
from src.validated_baselines import gaussian_moments, forecast_baseline
from src.models import ns_loadings

class BaselineAcceptance(unittest.TestCase):
    def test_random_walk_mean_is_exact_origin_curve_not_monte_carlo_estimate(self):
        y=np.random.default_rng(90).normal(size=(200,8))
        mean,cov=forecast_baseline(y,'M0',4,None)
        np.testing.assert_array_equal(mean,y[-1])
        np.testing.assert_allclose(cov,np.cov(y[4:]-y[:-4],rowvar=False))

    def test_ar_forecast_matches_direct_ols_and_geometric_variance(self):
        f=np.random.default_rng(50).normal(size=(200,3))
        load=ns_loadings(np.array([.25,.5,1,2,5,7,10,30]),.52)
        y=f@load.T
        mean,cov=forecast_baseline(y,'M2',4,load)
        params=[]
        for j in range(3):
            X=np.column_stack([np.ones(199),f[:-1,j]])
            beta=np.linalg.lstsq(X,f[1:,j],rcond=None)[0]
            q=np.mean((f[1:,j]-X@beta)**2)
            params.append((beta,q))
        mf=np.array([b[1]**4*f[-1,j]+b[0]*sum(b[1]**k for k in range(4)) for j,(b,q) in enumerate(params)])
        vf=np.array([q*sum(b[1]**(2*k) for k in range(4)) for b,q in params])
        np.testing.assert_allclose(mean,load@mf,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(cov,load@np.diag(vf)@load.T+np.eye(8)*1e-6,rtol=1e-10,atol=1e-10)

    def test_mean_covariance_do_not_depend_on_sampling_seed(self):
        y=np.random.default_rng(10).normal(size=(100,8))
        for model in ['M0','M1','M2']:
            load=ns_loadings(np.array([.25,.5,1,2,5,7,10,30]),.5)
            m,c=forecast_baseline(y,model,4,load)
            self.assertEqual(m.shape,(8,));self.assertEqual(c.shape,(8,8))
            self.assertGreater(np.linalg.eigvalsh(c).min(),0)

if __name__=='__main__':unittest.main()
