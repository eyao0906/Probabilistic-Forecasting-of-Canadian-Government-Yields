"""Target-preservation tests for Stage A mixing repairs (no production artifacts)."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import invgamma, norm


class InterweavingTests(unittest.TestCase):
    def test_noncentered_scale_target_matches_transformed_stationary_joint(self):
        from src import bayesian_revised as br
        self.assertTrue(hasattr(br, 'noncentered_scale_log_target'),
                        'Missing exact noncentered scale target')
        z = np.array([.3, -.7, .2, 1.1]); obs = np.array([-2., -4., .1, -1.])
        omega = -2.; phi = .82; p = br.Prior()
        def independent(logq):
            q = np.exp(logq); h = omega + np.sqrt(q)*z
            lp = norm.logpdf(h[0], omega, np.sqrt(q/(1-phi**2)))
            lp += norm.logpdf(h[1:], omega+phi*(h[:-1]-omega), np.sqrt(q)).sum()
            lp += invgamma.logpdf(q, p.q_shape, scale=p.q_scale)
            lp += br.logchi2_logpdf(obs-h).sum()
            return lp + (len(h)/2+1)*logq
        for logq in [-5., -2., .5]:
            self.assertAlmostEqual(
                br.noncentered_scale_log_target(logq,z,omega,obs,p)
                -br.noncentered_scale_log_target(-1.,z,omega,obs,p),
                independent(logq)-independent(-1.), places=10)

    def test_interweaving_is_used_without_changing_prior_or_support(self):
        from src import bayesian_revised as br
        self.assertTrue(hasattr(br, 'interweave_scale'), 'Missing executable scale step')
        rng=np.random.default_rng(119)
        x=np.empty(90); x[0]=0.
        for i in range(1,len(x)):
            x[i]=.05+.8*x[i-1]+rng.normal(0,.3)
        p=br.freeze_priors(np.column_stack([x,x,x]))[0]
        with patch.object(br, 'interweave_scale', wraps=br.interweave_scale) as spy:
            result=br.sample_chain(x,p,'M5',np.random.default_rng(14),
                                   warmup=8,draws=12,kernel='interweave')
            self.assertEqual(spy.call_count,20)
        self.assertTrue(np.isfinite(result['draws']).all())
        self.assertTrue((result['draws'][:,4]>0).all())
        self.assertTrue((abs(result['draws'][:,3])<1).all())
        self.assertEqual(result['prior_beta_variance'],list(p.beta_variance))
        self.assertEqual(result['kernel'],'interweave')
        obs=np.array([-2.,-3.,-1.]); h=np.array([-1.2,-2.4,-.9]); omega=-2.; q=.3
        hp,qp=br.interweave_scale(obs,h,omega,q,p,np.random.default_rng(77))
        np.testing.assert_allclose((hp-omega)/np.sqrt(qp),(h-omega)/np.sqrt(q),atol=1e-12)


class StageDesignTests(unittest.TestCase):
    def test_windows_are_predefined_on_existing_grid_with_frozen_initial_priors(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('src.bayesian_stage_a'),
                             'Missing staged diagnostic runner')
        from src.bayesian_stage_a import make_design
        design=make_design()
        self.assertEqual(design['initial_n'],835)
        self.assertEqual([w['name'] for w in design['windows']],['initial','stress','late'])
        self.assertEqual(design['windows'][0]['date'],'2016-01-08')
        self.assertEqual(design['windows'][-1]['date'],'2025-11-28')
        self.assertTrue(all(w['n_training']>=835 for w in design['windows']))
        self.assertEqual(design['lambda'],0.5160377358490565)
        self.assertEqual(design['initial_last_date'],'2015-12-31')

    def test_diagnostic_gate_fails_closed_for_nonfinite_or_insufficient_chains(self):
        from src import bayesian_stage_a as stage
        self.assertTrue(hasattr(stage,'diagnose'),'Missing fail-closed gate')
        x=np.random.default_rng(82).normal(size=(4,1200,2))
        rows=stage.diagnose(x,['a','b'])
        self.assertTrue(all(r['passes'] for r in rows))
        self.assertFalse(any(r['passes'] for r in stage.diagnose(x[:2],['a','b'])))
        x[0,0,0]=np.nan
        self.assertFalse(stage.diagnose(x,['a','b'])[0]['passes'])

    def test_runner_checkpoints_and_refuses_rolling_acceptance_for_short_trial(self):
        from src import bayesian_stage_a as stage
        self.assertTrue(hasattr(stage,'run_window'),'Missing executable window runner')
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=Path.home()/'AppData/Local/hermes/cache/scratch') as folder:
            out=Path(folder)/'run'
            report=stage.run_window(out,'initial','interweave',warmup=4,draws=12,models=['M5'])
            self.assertFalse(report['accepted_for_rolling'])
            self.assertFalse(report['parameter_gate_passed'])
            self.assertEqual(len(report['diagnostics']),24)
            self.assertEqual(len(list(out.glob('*_chain*.npz'))),12)
            resumed=stage.run_window(out,'initial','interweave',warmup=4,draws=12,models=['M5'])
            self.assertEqual(report['diagnostics'],resumed['diagnostics'])
            with self.assertRaisesRegex(ValueError,'configuration'):
                stage.run_window(out,'initial','interweave',warmup=4,draws=16,models=['M5'])


class CollapsedTests(unittest.TestCase):
    def test_collapsed_target_matches_dense_stationary_gaussian_with_jacobians(self):
        from src import bayesian_revised as br
        self.assertTrue(hasattr(br,'collapsed_log_target'),'Missing collapsed target')
        from scipy.stats import beta, multivariate_normal
        obs=np.array([-2.,-3.,-.5,1.]); labels=np.array([0,4,9,2]); p=br.Prior(omega_mean=-2.)
        def independent(theta):
            omega,u,logq=theta; phi=np.tanh(u); q=np.exp(logq)
            i=np.arange(len(obs))
            cov=q/(1-phi**2)*phi**abs(i[:,None]-i[None,:])+np.diag(br.MIX_V[labels])
            return (multivariate_normal.logpdf(obs-br.MIX_M[labels],np.full(len(obs),omega),cov)
                    +norm.logpdf(omega,p.omega_mean,np.sqrt(p.omega_variance))
                    +beta.logpdf((phi+1)/2,p.phi_alpha,p.phi_beta)+np.log(1-phi**2)
                    +invgamma.logpdf(q,p.q_shape,scale=p.q_scale)+logq)
        origin=np.array([-2.,.8,-2.])
        for theta in [np.array([-1.,1.7,-3.]), np.array([-3.,-.2,-1.]),np.array([-2.,4.,-6.])]:
            self.assertAlmostEqual(br.collapsed_log_target(theta,obs,labels,p)
                                   -br.collapsed_log_target(origin,obs,labels,p),
                                   independent(theta)-independent(origin),places=9)

    def test_joint_collapsed_kernel_exact_detailed_balance_and_sampler_use(self):
        from src import bayesian_revised as br
        self.assertTrue(hasattr(br,'collapsed_path_step'),'Missing exact joint proposal')
        from scipy.special import logsumexp
        from scipy.stats import beta, multivariate_normal
        from itertools import product
        obs=np.array([-2.1,.3]); h=np.array([-1.2,.4]); hp=np.array([-.2,-.6])
        theta=np.array([-.8,.7,-1.]); proposed=np.array([-.6,.9,-.8]); p=br.Prior()
        proposal_cov=np.diag([.15,.1,.2])**2
        def prior_path(t,state):
            w,u,lq=t; phi=np.tanh(u); q=np.exp(lq)
            cov=q/(1-phi**2)*np.array([[1.,phi],[phi,1.]])
            return (multivariate_normal.logpdf(state,np.full(2,w),cov)
                    +norm.logpdf(w,p.omega_mean,np.sqrt(p.omega_variance))
                    +beta.logpdf((phi+1)/2,p.phi_alpha,p.phi_beta)+np.log(1-phi**2)
                    +invgamma.logpdf(q,p.q_shape,scale=p.q_scale)+lq)
        def kernel(t,state,tp,statep):
            prob=br.mixture_probabilities(obs,state); terms=[]
            w,u,lq=tp; phi=np.tanh(u); q=np.exp(lq)
            cov=q/(1-phi**2)*np.array([[1.,phi],[phi,1.]])
            for labels in product(range(10),repeat=2):
                s=np.array(labels); r=br.MIX_V[s]
                v=np.linalg.inv(np.linalg.inv(cov)+np.diag(1/r))
                m=v@(np.linalg.solve(cov,np.full(2,w))+(obs-br.MIX_M[s])/r)
                terms.append(np.log(prob[0,s[0]])+np.log(prob[1,s[1]])
                             +min(0.,br.collapsed_log_target(tp,obs,s,p)-br.collapsed_log_target(t,obs,s,p))
                             +multivariate_normal.logpdf(statep,m,v))
            return float(logsumexp(terms)+multivariate_normal.logpdf(tp,t,proposal_cov))
        correction=br.correction_log_ratio(obs,h,hp)
        forward=prior_path(theta,h)+br.logchi2_logpdf(obs-h).sum()+kernel(theta,h,proposed,hp)+min(0.,correction)
        backward=prior_path(proposed,hp)+br.logchi2_logpdf(obs-hp).sum()+kernel(proposed,hp,theta,h)+min(0.,-correction)
        self.assertAlmostEqual(forward,backward,places=10)
        x=np.random.default_rng(42).normal(size=50)
        with patch.object(br,'collapsed_path_step',wraps=br.collapsed_path_step) as spy:
            result=br.sample_chain(x,p,'M5',np.random.default_rng(44),warmup=8,draws=12,kernel='collapsed')
            self.assertEqual(spy.call_count,20)
        self.assertTrue(np.isfinite(result['draws']).all())
        self.assertTrue((result['draws'][:,4]>0).all())
        self.assertGreaterEqual(result['collapsed_acceptance'],0.)
        self.assertLessEqual(result['collapsed_acceptance'],1.)


class MomentAuditTests(unittest.TestCase):
    def test_inverse_gamma_scale_blocks_true_predictive_covariance(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('src.bayesian_stage_a_checks'),
                             'Missing predictive-moment gate')
        from src.bayesian_stage_a_checks import moment_audit
        from src import bayesian_revised as br
        p=br.Prior()
        report=moment_audit(p,835)
        self.assertFalse(report['finite_predictive_second_moment'])
        self.assertFalse(report['stage_a_mathematical_gate_passed'])
        h=np.linspace(-3.,-1.,835)
        a,b=br.q_conditional(h,-2.,.9,p.q_shape,p.q_scale)
        self.assertEqual(report['conditional_ig_shape'],a)
        # Beyond this threshold the tilted IG density grows at least exp(q/4).
        for q in report['tail_threshold_q']*np.array([1.,10.,100.]):
            derivative=.5-(a+1)/q+b/q**2
            self.assertGreaterEqual(derivative,.25)

    def test_synthetic_generator_matches_declared_stationary_transition(self):
        from src import bayesian_stage_a_checks as checks
        self.assertTrue(hasattr(checks,'simulate_sv'),'Missing reproducible synthetic check')
        truth={'a':.04,'b':.85,'omega':-3.,'phi':.94,'q':.08}
        x,h=checks.simulate_sv(240,30261027,truth)
        rng=np.random.default_rng(30261027)
        expected_h=truth['omega']+np.sqrt(truth['q']/(1-truth['phi']**2))*rng.normal()
        self.assertAlmostEqual(h[0],expected_h)
        expected_x=truth['a']+truth['b']*x[0]+np.exp(expected_h/2)*rng.normal()
        self.assertAlmostEqual(x[1],expected_x)
        self.assertEqual(x.shape,(240,)); self.assertEqual(h.shape,(239,))
        xx,hh=checks.simulate_sv(240,30261027,truth)
        np.testing.assert_array_equal(x,xx); np.testing.assert_array_equal(h,hh)

    def test_synthetic_recovery_saves_real_chains_and_never_authorizes_forecasts(self):
        from src import bayesian_stage_a_checks as checks
        self.assertTrue(hasattr(checks,'run_synthetic'),'Missing executable recovery experiment')
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=Path.home()/'AppData/Local/hermes/cache/scratch') as folder:
            out=Path(folder)/'synthetic'
            report=checks.run_synthetic(out,warmup=4,draws=12)
            self.assertFalse(report['accepted_for_rolling'])
            self.assertEqual(report['chains'],4)
            with np.load(out/'draws.npz') as z:
                self.assertEqual(z['draws'].shape,(4,12,8))
            self.assertEqual(len(report['recovery']),5)
            with self.assertRaises(FileExistsError):
                checks.run_synthetic(out,warmup=4,draws=12)

    def test_pilot_does_not_accept_infinite_moments_when_rhat_and_ess_pass(self):
        from src import bayesian_revised as br
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=Path.home()/'AppData/Local/hermes/cache/scratch') as folder:
            # Force diagnostic pass to isolate the mathematical acceptance gate.
            with patch('arviz.rhat',return_value=1.), patch('arviz.ess',return_value=1000.):
                report=br.run_pilot(out=Path(folder),warmup=1,draws=8,chains=4)
            self.assertFalse(report['accepted_for_rolling'],
                             'Convergence cannot override infinite predictive covariance')
            self.assertTrue(report['parameter_gate_passed'])
            self.assertFalse(report['predictive_moment_audit']['finite_predictive_second_moment'])


if __name__ == '__main__':
    unittest.main()
