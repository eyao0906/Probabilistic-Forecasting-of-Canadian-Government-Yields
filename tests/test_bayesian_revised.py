"""Analytical and target-preservation regressions for the isolated Bayesian pilot."""
import unittest
import numpy as np
from scipy.special import digamma, polygamma


class MixtureSourceTests(unittest.TestCase):
    def test_omori_table_one_and_logchisquare_moments(self):
        from src import bayesian_revised as br
        # Independently transcribed from legal UCI author manuscript Table 1.
        expected = np.array([
            [.00609,1.92677,.11265],[.04775,1.34744,.17788],
            [.13057,.73504,.26768],[.20674,.02266,.40611],
            [.22715,-.85173,.62699],[.18842,-1.97278,.98583],
            [.12047,-3.46788,1.57469],[.05591,-5.55246,2.54498],
            [.01575,-8.68384,4.16591],[.00115,-14.65000,7.33342]])
        np.testing.assert_array_equal(np.column_stack([br.MIX_W,br.MIX_M,br.MIX_V]),expected)
        self.assertAlmostEqual(float(br.MIX_W.sum()),1.,places=12)
        m = float(br.MIX_W @ br.MIX_M)
        v = float(br.MIX_W @ (br.MIX_V+br.MIX_M**2)-m*m)
        self.assertAlmostEqual(m,float(digamma(.5)+np.log(2)),delta=.002)
        self.assertAlmostEqual(v,float(polygamma(1,.5)),delta=.03)


class StationaryConditionalTests(unittest.TestCase):
    def test_q_conditional_matches_joint_inverse_gamma(self):
        from src import bayesian_revised as br
        from scipy.stats import norm, invgamma
        h=np.array([1.7,-.6,.2]); w=-1.3; phi=.72; a=2.; b=.2
        shape,scale=br.q_conditional(h,w,phi,a,b)
        self.assertEqual(shape,a+len(h)/2)
        def joint(q):
            return (invgamma.logpdf(q,a,scale=b)
                    +norm.logpdf(h[0],w,np.sqrt(q/(1-phi**2)))
                    +norm.logpdf(h[1:],w+phi*(h[:-1]-w),np.sqrt(q)).sum())
        for q in [.1,.8,4.]:
            self.assertAlmostEqual(joint(q)-joint(1.),invgamma.logpdf(q,shape,scale=scale)
                                   -invgamma.logpdf(1.,shape,scale=scale),places=10)

    def test_omega_conditional_matches_stationary_joint_quadratic(self):
        from src import bayesian_revised as br
        from scipy.stats import norm
        h=np.array([-2.,-1.4,-2.3]); phi=.84; q=.37; m=-1.1; v=2.4
        mean,var=br.omega_conditional(h,phi,q,m,v)
        def joint(w):
            return (norm.logpdf(w,m,np.sqrt(v))
                    +norm.logpdf(h[0],w,np.sqrt(q/(1-phi**2)))
                    +norm.logpdf(h[1:],w+phi*(h[:-1]-w),np.sqrt(q)).sum())
        for w in [-4.,0.,2.]:
            self.assertAlmostEqual(joint(w)-joint(mean),-.5*(w-mean)**2/var,places=11)


class FFBSTests(unittest.TestCase):
    def test_ffbs_affine_draw_matches_dense_gaussian_without_clipping(self):
        from src import bayesian_revised as br
        n=5; obs=np.array([-25.,-19.,-17.,-20.,-21.]); mix=np.array([0,4,9,3,8])
        for phi,q in [(.72,.4),(.99999,.003)]:
            omega=-20.
            idx=np.arange(n)
            cov=q/(1-phi**2)*phi**np.abs(idx[:,None]-idx[None,:])
            r=br.MIX_V[mix]; y=obs-br.MIX_M[mix]
            post_cov=np.linalg.inv(np.linalg.inv(cov)+np.diag(1/r))
            post_mean=post_cov@(np.linalg.solve(cov,np.full(n,omega))+y/r)
            mean=br.ffbs(obs,mix,omega,phi,q,np.zeros(n))
            transform=np.column_stack([br.ffbs(obs,mix,omega,phi,q,np.eye(n)[i])-mean for i in range(n)])
            np.testing.assert_allclose(mean,post_mean,atol=1e-8)
            np.testing.assert_allclose(transform@transform.T,post_cov,atol=1e-8)
            self.assertLess(mean.min(),-15.)
        with self.assertRaises(ValueError):
            br.ffbs(obs,mix,omega,1.,q,np.zeros(n))


class ExactCorrectionTests(unittest.TestCase):
    def test_enumerated_auxiliary_kernel_detailed_balance_and_real_mh_step(self):
        from src import bayesian_revised as br
        from scipy.stats import multivariate_normal, norm
        from scipy.special import logsumexp
        from itertools import product
        obs=np.array([-2.1,.3]); h=np.array([-1.2,.4]); hp=np.array([-.2,-.6])
        w=-.8; phi=.65; q=.4
        cov=q/(1-phi**2)*np.array([[1.,phi],[phi,1.]])
        def kernel(old,new):
            probs=br.mixture_probabilities(obs,old)
            terms=[]
            for labels in product(range(10),repeat=2):
                s=np.array(labels); r=br.MIX_V[s]
                v=np.linalg.inv(np.linalg.inv(cov)+np.diag(1/r))
                m=v@(np.linalg.solve(cov,np.full(2,w))+(obs-br.MIX_M[s])/r)
                terms.append(np.log(probs[0,s[0]])+np.log(probs[1,s[1]])
                             +multivariate_normal.logpdf(new,m,v))
            return logsumexp(terms)
        def target(state):
            return (multivariate_normal.logpdf(state,np.full(2,w),cov)
                    +br.logchi2_logpdf(obs-state).sum())
        loga=br.correction_log_ratio(obs,h,hp)
        forward=target(h)+kernel(h,hp)+min(0.,loga)
        backward=target(hp)+kernel(hp,h)+min(0.,-loga)
        self.assertAlmostEqual(forward,backward,places=11)
        for seed in range(6):
            rng=np.random.default_rng(seed); clone=np.random.default_rng(seed)
            uniforms=clone.random(2); noise=clone.normal(size=2)
            probs=br.mixture_probabilities(obs,h)
            labels=(uniforms[:,None]>np.cumsum(probs,axis=1)).sum(axis=1)
            proposal=br.ffbs(obs,labels,w,phi,q,noise)
            accept=np.log(clone.random())<br.correction_log_ratio(obs,h,proposal)
            actual,flag,ratio=br.corrected_h_step(obs,h,w,phi,q,rng)
            np.testing.assert_array_equal(actual,proposal if accept else h)
            self.assertEqual(flag,accept)
            self.assertAlmostEqual(ratio,br.correction_log_ratio(obs,h,proposal))

    def test_correction_is_exact_over_marginal_mixture_not_selected_component(self):
        from src import bayesian_revised as br
        from scipy.stats import norm
        from scipy.special import logsumexp
        residual=np.array([1.e-200,-.24,1.3]); obs=br.log_squared_residual(residual)
        self.assertTrue(np.isfinite(obs).all())
        old=np.array([-30.,-3.,1.]); new=np.array([-28.,-1.,-2.])
        def exact(h):
            # First residual squared underflows; stable density contribution is zero.
            return norm.logpdf(residual,scale=np.exp(h/2)).sum()
        def approx(h):
            return logsumexp(np.log(br.MIX_W)+norm.logpdf(
                obs[:,None]-h[:,None],br.MIX_M,np.sqrt(br.MIX_V)),axis=1).sum()
        expected=exact(new)-exact(old)-approx(new)+approx(old)
        self.assertAlmostEqual(br.correction_log_ratio(obs,old,new),expected,places=9)
        self.assertAlmostEqual(br.correction_log_ratio(obs,new,old),-expected,places=9)
        with self.assertRaisesRegex(ValueError,'zero'):
            br.log_squared_residual(np.array([0.,1.]))
        np.testing.assert_allclose(br.logchi2_logpdf(np.array([-2.,0.,2.])),
                                   -.5*np.log(2*np.pi)+.5*np.array([-2.,0.,2.])
                                   -.5*np.exp(np.array([-2.,0.,2.])))


class MatchedPriorTests(unittest.TestCase):
    def test_initial_frozen_priors_and_unscaled_exact_gaussian_beta_conditional(self):
        from src import bayesian_revised as br
        from scipy.stats import norm, multivariate_normal
        initial=np.array([[1.,-1.,2.],[1.3,-.8,1.7],[.8,-1.1,2.4],[1.2,-.7,1.9]])
        priors=br.freeze_priors(initial)
        self.assertEqual(len(priors),3)
        p=priors[0]
        self.assertEqual(p.beta_mean,(0.,.7))
        self.assertEqual(p.beta_variance,(25.,.09))
        with self.assertRaises((AttributeError,TypeError)):
            p.omega_mean=99.
        lag=np.array([.2,.8,-.3]); y=np.array([.4,-.2,.1])
        design=np.column_stack([np.ones(3),lag])
        for h in [np.array([-2.,1.,.6]),np.full(3,20.)]:
            mean,cov=br.beta_conditional(y,lag,h,p)
            precision=np.diag(1/np.array(p.beta_variance))+design.T@(np.exp(-h)[:,None]*design)
            np.testing.assert_allclose(cov,np.linalg.inv(precision),rtol=1e-12,atol=1e-12)
            def joint(beta):
                return (multivariate_normal.logpdf(beta,p.beta_mean,np.diag(p.beta_variance))
                        +norm.logpdf(y,design@beta,np.exp(h/2)).sum())
            beta=np.array([-.12,.6])
            self.assertAlmostEqual(joint(beta)-joint(mean),
                                   multivariate_normal.logpdf(beta,mean,cov)
                                   -multivariate_normal.logpdf(mean,mean,cov),places=10)


class ChainExecutionTests(unittest.TestCase):
    def test_both_models_use_frozen_priors_and_stationary_parameter_support(self):
        from src import bayesian_revised as br
        self.assertTrue(hasattr(br, 'sample_chain'), 'Missing executable corrected sampler')
        rng=np.random.default_rng(99)
        x=np.empty(80); x[0]=1.
        for i in range(1,len(x)):
            x[i]=.04+.92*x[i-1]+rng.normal(0,.2)
        p=br.freeze_priors(np.column_stack([x,x,x]))[0]
        for model in ['M4','M5']:
            result=br.sample_chain(x,p,model,np.random.default_rng(4),warmup=8,draws=12)
            a=result['draws']
            self.assertEqual(a.shape[0],12)
            self.assertTrue(np.isfinite(a).all())
            self.assertTrue((abs(a[:,1])<1).all())
            self.assertEqual(result['prior_beta_mean'],list(p.beta_mean))
            self.assertEqual(result['prior_beta_variance'],list(p.beta_variance))
            if model=='M5':
                self.assertTrue((abs(a[:,3])<1).all())
                self.assertTrue((a[:,4]>0).all())
                self.assertGreaterEqual(result['h_acceptance'],0.)
                self.assertLessEqual(result['h_acceptance'],1.)


    def test_phi_target_includes_stationary_density_prior_and_jacobian(self):
        from src import bayesian_revised as br
        from scipy.stats import beta, norm
        h=np.array([-2.,-1.,-3.]); p=br.Prior(); omega=-1.5; q=.4
        def expected(u):
            phi=np.tanh(u)
            return (beta.logpdf((phi+1)/2,p.phi_alpha,p.phi_beta)
                    +np.log(1-phi**2)
                    +norm.logpdf(h[0],omega,np.sqrt(q/(1-phi**2)))
                    +norm.logpdf(h[1:],omega+phi*(h[:-1]-omega),np.sqrt(q)).sum())
        for u in [-.7,.3,2.]:
            self.assertAlmostEqual(br.phi_log_target(u,h,omega,q,p)-br.phi_log_target(0.,h,omega,q,p),expected(u)-expected(0.),places=10)

    def test_pilot_runner_keeps_incomplete_diagnostics_out_of_accepted_results(self):
        from src import bayesian_revised as br
        self.assertTrue(hasattr(br, 'run_pilot'), 'Missing reproducible pilot runner')
        import tempfile
        from pathlib import Path
        scratch=Path.home()/'AppData/Local/hermes/cache/scratch'
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            report=br.run_pilot(out=Path(folder),warmup=8,draws=12,chains=2)
            self.assertFalse(report['accepted_for_rolling'])
            self.assertEqual(report['training_n'],835)
            self.assertTrue((Path(folder)/'diagnostics.csv').exists())


if __name__ == '__main__':
    unittest.main()


