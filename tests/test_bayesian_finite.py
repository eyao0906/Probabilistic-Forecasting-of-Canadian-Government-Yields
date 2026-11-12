"""Finite-moment prior and sampler regressions; never touch accepted results."""
import importlib.util
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, beta as beta_dist, multivariate_normal
from src import bayesian_revised as br


class PriorTests(unittest.TestCase):
    def test_normalized_bounded_prior_and_full_coordinate_jacobian(self):
        self.assertIsNotNone(importlib.util.find_spec('src.bayesian_finite'),
                             'Missing finite-moment parameterization')
        from src import bayesian_finite as bf
        p=bf.FinitePrior(br.Prior(omega_mean=-2.))
        self.assertAlmostEqual(quad(lambda v:np.exp(bf.v_logpdf(v,p)),0,p.v_upper)[0],1.,places=12)
        for v in [-1.,0.,4.,5.]: self.assertEqual(bf.v_logpdf(v,p),-np.inf)
        theta=np.array([-2.,.8,np.log(.7)])
        m,phi,v,w,q=bf.decode(theta,p)
        self.assertAlmostEqual(q,v*(1-phi**2)); self.assertAlmostEqual(w,m-v/2)
        # Direct physical density and a numeric 3-D coordinate Jacobian.
        def physical(t):
            mm,pp,vv,ww,qq=bf.decode(t,p)
            return np.array([ww,pp,qq])
        eps=1e-5
        jac=np.column_stack([(physical(theta+eps*np.eye(3)[i])-physical(theta-eps*np.eye(3)[i]))/(2*eps) for i in range(3)])
        direct=(norm.logpdf(m,p.base.omega_mean,np.sqrt(p.base.omega_variance))
                +beta_dist.logpdf((phi+1)/2,p.base.phi_alpha,p.base.phi_beta)-np.log(2)
                +bf.v_logpdf(v,p)-np.log(1-phi**2)+np.log(abs(np.linalg.det(jac))))
        self.assertAlmostEqual(bf.log_prior(theta,p),direct,places=8)
        with self.assertRaises(ValueError): bf.FinitePrior(br.Prior(),v_upper=0.)


class TargetTests(unittest.TestCase):
    def test_centered_collapsed_and_noncentered_targets_match_dense_reference(self):
        from src import bayesian_finite as bf
        self.assertTrue(hasattr(bf,'collapsed_target'),'Missing rederived targets')
        p=bf.FinitePrior(br.Prior(omega_mean=-2.))
        obs=np.array([-2.,-3.,-.5,1.]); labels=np.array([0,4,9,2])
        h=np.array([-.5,-1.,-2.,.3]); z=np.array([.1,-.3,.5,.7])
        def independent(t,collapsed):
            m,phi,v,w,q=bf.decode(t,p); i=np.arange(4)
            cov=v*phi**abs(i[:,None]-i[None,:])
            return (bf.log_prior(t,p)+multivariate_normal.logpdf(
                obs-br.MIX_M[labels] if collapsed else h,np.full(4,w),
                cov+np.diag(br.MIX_V[labels]) if collapsed else cov))
        for t in [np.array([-2.,.8,-1.]),np.array([-1.,2.,.4]),np.array([-3.,-.3,-3.])]:
            self.assertAlmostEqual(bf.collapsed_target(t,obs,labels,p),independent(t,True),places=10)
            self.assertAlmostEqual(bf.centered_target(t,h,p),independent(t,False),places=10)
        # Scale coordinate changes h, omega and q jointly, at fixed m,phi,z.
        m=-2.; phi=.8
        def transformed(s):
            v=np.exp(s); hh=m-v/2+np.sqrt(v)*z
            return bf.centered_target(np.array([m,np.arctanh(phi),s]),hh,p)+br.logchi2_logpdf(obs-hh).sum()+len(hh)/2*s
        for s in [-4.,-1.,.8]:
            self.assertAlmostEqual(bf.noncentered_target(s,z,m,phi,obs,p)-bf.noncentered_target(-2.,z,m,phi,obs,p),transformed(s)-transformed(-2.),places=10)
        mean,var=bf.m_conditional(h,phi,.7,p)
        def logm(m): return bf.centered_target(np.array([m,np.arctanh(phi),np.log(.7)]),h,p)
        self.assertAlmostEqual(logm(-1.)-logm(-2.),norm.logpdf(-1.,mean,np.sqrt(var))-norm.logpdf(-2.,mean,np.sqrt(var)),places=10)


class SamplerTests(unittest.TestCase):
    def test_exact_joint_kernel_and_real_sampler_without_ig_conditionals(self):
        from src import bayesian_finite as bf
        self.assertTrue(hasattr(bf,'sample_chain'),'Missing finite-prior sampler')
        from unittest.mock import patch
        from itertools import product
        from scipy.special import logsumexp
        p=bf.FinitePrior(br.Prior())
        obs=np.array([-2.1,.3]); h=np.array([-1.2,.4]); hp=np.array([-.2,-.6])
        t=np.array([-.8,.7,-1.]); tp=np.array([-.6,.9,-.8]); prop=np.diag([.15,.1,.2])**2
        def target(theta,hh): return bf.centered_target(theta,hh,p)+br.logchi2_logpdf(obs-hh).sum()
        def kernel(theta,hh,thetap,hhp):
            prob=br.mixture_probabilities(obs,hh); terms=[]
            m,phi,v,w,q=bf.decode(thetap,p); cov=v*np.array([[1.,phi],[phi,1.]])
            for labels in product(range(10),repeat=2):
                labels=np.array(labels); r=br.MIX_V[labels]
                vc=np.linalg.inv(np.linalg.inv(cov)+np.diag(1/r))
                mc=vc@(np.linalg.solve(cov,np.full(2,w))+(obs-br.MIX_M[labels])/r)
                terms.append(np.log(prob[0,labels[0]])+np.log(prob[1,labels[1]])
                             +min(0.,bf.collapsed_target(thetap,obs,labels,p)-bf.collapsed_target(theta,obs,labels,p))
                             +multivariate_normal.logpdf(hhp,mc,vc))
            return logsumexp(terms)+multivariate_normal.logpdf(thetap,theta,prop)
        ratio=br.correction_log_ratio(obs,h,hp)
        self.assertAlmostEqual(target(t,h)+kernel(t,h,tp,hp)+min(0.,ratio),
                               target(tp,hp)+kernel(tp,hp,t,h)+min(0.,-ratio),places=10)
        x=np.random.default_rng(42).normal(size=50)
        with patch.object(br,'q_conditional',side_effect=AssertionError('Old IG update used')), \
             patch.object(br,'phi_log_target',side_effect=AssertionError('Old fixed-q target used')), \
             patch.object(bf,'joint_step',wraps=bf.joint_step) as joint, \
             patch.object(bf,'scale_step',wraps=bf.scale_step) as scale:
            r=bf.sample_chain(x,p,'M5',np.random.default_rng(44),warmup=8,draws=12)
            self.assertEqual(joint.call_count,20); self.assertEqual(scale.call_count,20)
        z=r['draws']; names=r['names']; self.assertEqual(z.shape,(12,10)); self.assertTrue(np.isfinite(z).all())
        v=z[:,names.index('v')]; phi=z[:,names.index('phi')]
        self.assertTrue(((v>0)&(v<p.v_upper)).all()); self.assertTrue((abs(phi)<1).all())
        np.testing.assert_allclose(z[:,names.index('q')],v*(1-phi**2))
        np.testing.assert_allclose(z[:,names.index('omega')],z[:,names.index('m')]-v/2)
        r2=bf.sample_chain(x,p,'M5',np.random.default_rng(44),warmup=8,draws=12)
        np.testing.assert_array_equal(z,r2['draws'])
        old=br.sample_chain(x,p.base,'M4',np.random.default_rng(5),warmup=8,draws=12,kernel='collapsed')
        new=bf.sample_chain(x,p,'M4',np.random.default_rng(5),warmup=8,draws=12)
        np.testing.assert_array_equal(old['draws'],new['draws'])


class MomentTests(unittest.TestCase):
    def test_propriety_bound_and_prior_predictive_moments(self):
        from src import bayesian_finite as bf
        self.assertTrue(hasattr(bf,'moment_audit'),'Missing mathematical gate for new prior')
        p=bf.FinitePrior(br.Prior(omega_mean=-3.))
        for upper in [2.,4.,8.]:
            pp=bf.FinitePrior(p.base,v_upper=upper)
            audit=bf.moment_audit(pp,835)
            self.assertTrue(audit['posterior_proper']); self.assertTrue(audit['finite_predictive_second_moment'])
            for k in [1.,2.,4.]:
                numeric=quad(lambda v:np.exp(.5*(k*k-k)*v+bf.v_logpdf(v,pp)),0,upper)[0]
                expected=k*pp.base.omega_mean+.5*k*k*pp.base.omega_variance+np.log(numeric)
                self.assertAlmostEqual(bf.log_prior_exp_h_moment(k,pp),expected,places=10)
                self.assertTrue(np.isfinite(bf.log_prior_exp_h_moment(k,pp)))
            # Gaussian MGF of a joint stationary path is uniformly bounded even near phi=1.
            n=8; weights=-.5*np.ones(n); ii=np.arange(n)
            for phi in [.1,.99,.999999]:
                v=.99*upper
                covariance=pp.base.omega_variance*np.ones((n,n))+v*phi**abs(ii[:,None]-ii[None,:])
                exact=weights.sum()*(pp.base.omega_mean-v/2)+.5*weights@covariance@weights
                self.assertLessEqual(exact-.5*n*np.log(2*np.pi),bf.moment_audit(pp,n)['log_likelihood_normalizer_upper_bound'])


class RunnerTests(unittest.TestCase):
    def test_frozen_checkpoint_resume_and_fail_closed_acceptance(self):
        self.assertIsNotNone(importlib.util.find_spec('src.bayesian_finite_runner'),'Missing finite-prior runner')
        from src.bayesian_finite_runner import run_window
        from src import bayesian_finite as bf
        from unittest.mock import patch
        from pathlib import Path
        import tempfile
        import json
        with tempfile.TemporaryDirectory(dir=Path.home()/'AppData/Local/hermes/cache/scratch') as temp:
            out=Path(temp)/'window'
            r=run_window(out,'initial',warmup=4,draws=12,models=['M5'])
            self.assertFalse(r['accepted_for_rolling']); self.assertTrue(r['moment_gate_passed'])
            self.assertEqual(len(r['diagnostics']),30); self.assertEqual(len(list(out.glob('*chain*.npz'))),12)
            with patch.object(bf,'sample_chain',side_effect=AssertionError('Recomputed completed chain')):
                resumed=run_window(out,'initial',warmup=4,draws=12,models=['M5'])
            self.assertEqual(r['diagnostics'],resumed['diagnostics'])
            with self.assertRaisesRegex(ValueError,'configuration'):
                run_window(out,'initial',warmup=4,draws=16,models=['M5'])
            self.assertFalse(json.loads((out/'summary.json').read_text())['accepted_for_rolling'])


class SyntheticTests(unittest.TestCase):
    def test_fixed_data_four_genuine_chains_and_transparent_recovery(self):
        self.assertIsNotNone(importlib.util.find_spec('src.bayesian_finite_validation'),'Missing finite-prior synthetic check')
        from src.bayesian_finite_validation import run_synthetic
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory(dir=Path.home()/'AppData/Local/hermes/cache/scratch') as temp:
            out=Path(temp)/'synthetic'
            r=run_synthetic(out,warmup=8,draws=12)
            self.assertEqual(len(r['recovery']),7); self.assertFalse(r['accepted_for_rolling'])
            with np.load(out/'draws.npz',allow_pickle=False) as z:
                self.assertEqual(z['draws'].shape,(4,12,10))
                self.assertFalse(np.array_equal(z['draws'][0],z['draws'][1]))
            with self.assertRaises(FileExistsError): run_synthetic(out,warmup=8,draws=12)


class GlobalPersistenceTests(unittest.TestCase):
    def test_prior_independence_ratio_and_palindromic_sampler_execution(self):
        from src import bayesian_finite as bf
        self.assertTrue(hasattr(bf,'prior_phi_move'),'Missing global collapsed persistence move')
        from unittest.mock import patch
        p=bf.FinitePrior(br.Prior()); obs=np.array([-2.,-1.,.1]); labels=np.array([0,2,6])
        theta=np.array([-1.,1.4,-1.]); target=bf.collapsed_target(theta,obs,labels,p)
        class FixedRng:
            def beta(self,*args): return .98
            def random(self): return .25
        candidate=theta.copy(); candidate[1]=np.arctanh(.96)
        def log_phi_u(u):
            ph=np.tanh(u)
            return beta_dist.logpdf((ph+1)/2,p.base.phi_alpha,p.base.phi_beta)-np.log(2)+np.log1p(-ph*ph)
        ratio=bf.collapsed_target(candidate,obs,labels,p)-target+log_phi_u(theta[1])-log_phi_u(candidate[1])
        new,new_target=bf.prior_phi_move(theta,target,obs,labels,p,FixedRng())
        expected=candidate if np.log(.25)<ratio else theta
        np.testing.assert_array_equal(new,expected); self.assertEqual(new_target,bf.collapsed_target(expected,obs,labels,p))
        x=np.random.default_rng(7).normal(size=40)
        with patch.object(bf,'prior_phi_move',wraps=bf.prior_phi_move) as glob:
            r=bf.sample_chain(x,p,'M5',np.random.default_rng(8),warmup=8,draws=12)
            self.assertEqual(glob.call_count,20)
        self.assertTrue(np.isfinite(r['draws']).all())


class ExperimentTests(unittest.TestCase):
    def test_representative_only_execution_cannot_accept_130_origin_experiment(self):
        from src import bayesian_finite_experiment as experiment
        from tempfile import TemporaryDirectory
        from pathlib import Path
        self.assertTrue(hasattr(experiment,'run_experiment'),'Missing checkpointed experiment execution')
        gate=Path('results/bayesian_finite_20261027T194239Z/phase2_gate.json')
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'rolling'
            report=experiment.run_experiment(gate,root,workers=1,representatives_only=True)
            self.assertFalse(report['accepted'])
            self.assertEqual(report['planned_origins'],130)
            self.assertEqual(report['completed_origins'],3)
            self.assertEqual(len(report['failed_origins']),0)
            paths=list((root/'origins').glob('*/forecast.json'))
            self.assertEqual(len(paths),3)
            before={str(p):p.stat().st_mtime_ns for p in paths}
            again=experiment.run_experiment(gate,root,workers=1,representatives_only=True)
            self.assertEqual(again['completed_origins'],3)
            self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in paths})

    def test_plan_enumerates_130_common_origins_with_three_reuses(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('src.bayesian_finite_experiment'),'Missing frozen experiment plan')
        from src.bayesian_finite_experiment import make_plan
        from pathlib import Path
        gate=Path('results/bayesian_finite_20261027T194239Z/phase2_gate.json')
        plan=make_plan(gate)
        self.assertEqual(len(plan['origins']),130)
        self.assertEqual(len({r['index'] for r in plan['origins']}),130)
        self.assertEqual(sum(r['chain_source'] is not None for r in plan['origins']),3)
        self.assertEqual(plan['horizons'],[1,4])
        self.assertEqual(plan['models'],['M4','M5'])
        self.assertEqual(len(plan['maturities']),8)


class ProductionTests(unittest.TestCase):
    def test_forecasting_refuses_before_creating_output(self):
        from src import bayesian_finite_production as production
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json
        self.assertTrue(hasattr(production,'forecast_origin'),'Missing gated forecast execution')
        with TemporaryDirectory() as tmp:
            root=Path(tmp); gate=root/'gate.json'; gate.write_text(json.dumps({'accepted':False}))
            out=root/'forecast'
            with self.assertRaisesRegex(ValueError,'Phase 2'):
                production.forecast_origin(gate,out,836)
            self.assertFalse(out.exists())

    def test_origin_checkpoints_resume_without_sampling(self):
        from src import bayesian_finite_production as production
        from src.bayesian_stage_a import make_design
        from tempfile import TemporaryDirectory
        from pathlib import Path
        self.assertTrue(hasattr(production,'checkpoint_origin'),'Missing origin checkpoints')
        d=make_design(); t=d['first_index']+36
        with TemporaryDirectory() as tmp:
            out=Path(tmp)/'origin'
            first=production.checkpoint_origin(d,out,t,warmup=4,draws=12,seed=271026)
            files=list(out.glob('*chain*.npz')); before={p.name:p.stat().st_mtime_ns for p in files}
            self.assertEqual(len(files),24)
            second=production.checkpoint_origin(d,out,t,warmup=4,draws=12,seed=271026)
            self.assertEqual(before,{p.name:p.stat().st_mtime_ns for p in files})
            self.assertEqual(first['configuration'],second['configuration'])
            self.assertFalse(first['accepted_for_rolling'])
            with self.assertRaisesRegex(ValueError,'configuration'):
                production.checkpoint_origin(d,out,t,warmup=4,draws=12,seed=271027)

    def test_production_gate_rejects_unaccepted_or_incomplete_evidence(self):
        import importlib.util,json
        from tempfile import TemporaryDirectory
        from pathlib import Path
        self.assertIsNotNone(importlib.util.find_spec('src.bayesian_finite_production'),'Missing production gate')
        from src.bayesian_finite_production import require_phase2
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'gate.json'
            for report in [{'accepted':False},{'accepted':True}]:
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'Phase 2'): require_phase2(path)


class GateTests(unittest.TestCase):
    def test_aggregate_gate_refuses_missing_evidence(self):
        from src import bayesian_finite_gate as gate
        self.assertTrue(hasattr(gate,'write_gate'),'Missing aggregate gate')
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError): gate.write_gate(tmp)
            self.assertFalse((Path(tmp)/'phase2_gate.json').exists())

    def test_representative_pair_requires_independent_seeds_and_all_forecasts(self):
        import importlib.util,copy
        self.assertIsNotNone(importlib.util.find_spec('src.bayesian_finite_gate'),'Missing independent predictive gate')
        from src.bayesian_finite_gate import compare_run_reports
        records=[]
        for model in ['M4','M5']:
            for h in [1,4]:
                records.append({'model':model,'horizon':h,'passes':True,'mean':[0.]*8,
                                'covariance':np.eye(8).tolist(),'factor_variance':[1.]*3,
                                'dss':1.,'energy_score':1.,'dss_mcse':.01,'energy_mcse':.01})
        cfg={'window':{'name':'initial','index':1},'input_sha256':'unit-fixture','lambda':.5,
             'priors':[{'v_upper':4.}],'chains':4,'draws':6000,'warmup':3000,'models':['M4','M5']}
        first={'configuration':cfg,'passes':True,'chain_seeds':list(range(24)),'records':records}
        second=copy.deepcopy(first); second['chain_seeds']=list(range(24,48))
        self.assertTrue(compare_run_reports(first,second)['passes'])
        second['chain_seeds'][0]=first['chain_seeds'][0]
        with self.assertRaisesRegex(ValueError,'independent'): compare_run_reports(first,second)
        second['chain_seeds']=list(range(24,48)); second['records'].pop()
        with self.assertRaisesRegex(ValueError,'complete'): compare_run_reports(first,second)


class PredictiveTests(unittest.TestCase):
    def test_saved_prediction_validation_refuses_unaccepted_chains(self):
        from src import bayesian_finite_validation as bv
        self.assertTrue(hasattr(bv,'validate_predictions'),'Missing saved predictive validation')
        import json
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'summary.json').write_text(json.dumps({'parameter_gate_passed':False,'moment_gate_passed':True}))
            with self.assertRaisesRegex(ValueError,'Unaccepted'): bv.validate_predictions(path)
            self.assertFalse((path/'predictive_validation').exists())

    def test_scoring_record_reconstruction_and_fail_closed_stability(self):
        from src import bayesian_finite_validation as bv
        self.assertTrue(hasattr(bv,'predictive_record'),'Missing predictive score-validation record')
        from src.scores import dss,energy_score
        rng=np.random.default_rng(24); factors=[]
        for j in range(3):
            a=rng.normal(.1,.001,(4,600)); b=rng.normal(.8,.001,(4,600)); w=rng.normal(-2,.002,(4,600))
            factors.append((np.stack([a,b,w],axis=-1),['a','b','log_variance']))
        load=np.random.default_rng(25).normal(size=(8,3)); x=np.ones(3); r=np.full(8,.01); y=np.zeros(8)
        first=bv.predictive_record(factors,x,load,r,y,'M4',4,np.random.default_rng(26))
        second=bv.predictive_record(factors,x,load,r,y,'M4',4,np.random.default_rng(27))
        self.assertTrue(first['passes']); self.assertEqual(first['scenarios'].shape,(2048,8))
        self.assertAlmostEqual(first['dss'],dss(y,first['mean'],first['covariance']),places=11)
        self.assertAlmostEqual(first['energy_score'],energy_score(first['scenarios'],y),places=11)
        self.assertTrue(bv.prediction_difference(first,second)['passes'])
        changed=dict(second); changed['covariance']=(np.asarray(changed['covariance'])*2).tolist()
        self.assertFalse(bv.prediction_difference(first,changed)['passes'])

    def test_conditional_moments_scenarios_and_total_variance(self):
        from src import bayesian_finite_validation as bv
        self.assertTrue(hasattr(bv,'factor_conditionals'),'Missing analytic predictive moments')
        names=['a','b','m','phi','v','omega','q','h_first','h_mid','h_last']
        a=.1; b=.8; m=-1.8; phi=.7; v=.4; w=m-v/2; q=v*(1-phi**2); last=-1.3
        value=np.array([a,b,m,phi,v,w,q,w,w,last]); values=np.tile(value,(4,12,1))
        for h in [1,4]:
            mean,var=bv.factor_conditionals(values,names,1.,h,'M5')
            expected_mean=b**h+a*sum(b**k for k in range(h))
            expected_var=sum(b**(2*(h-k))*np.exp(w+phi**k*(last-w)+v*(1-phi**(2*k))/2) for k in range(1,h+1))
            np.testing.assert_allclose(mean,expected_mean); np.testing.assert_allclose(var,expected_var)
            draws=bv.factor_scenarios(values,names,1.,h,'M5',np.random.default_rng(271026),64000)
            self.assertLess(abs(draws.mean()-expected_mean),.012)
            self.assertLess(abs(draws.var()/expected_var-1),.04)
        cm,cv=bv.factor_conditionals(values[...,[0,1,5]],['a','b','log_variance'],1.,4,'M4')
        np.testing.assert_allclose(cv,np.exp(w)*sum(b**(2*k) for k in range(4)))
        mean,cov,fm,fv=bv.yield_moments([np.array([0.,2.])]*3,[np.array([1.,1.])]*3,np.ones((8,3)),np.full(8,.01))
        np.testing.assert_allclose(fm,1.); np.testing.assert_allclose(fv,2.)
        np.testing.assert_allclose(cov,np.full((8,8),6.)+np.eye(8)*.01)
        self.assertGreater(np.linalg.eigvalsh(cov).min(),0.)


if __name__=='__main__': unittest.main()
