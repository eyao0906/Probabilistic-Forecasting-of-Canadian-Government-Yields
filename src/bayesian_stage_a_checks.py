"""Analytical predictive-moment gate and small synthetic Stage A checks."""
from __future__ import annotations
import numpy as np
from src import bayesian_revised as br


def simulate_sv(n,seed,truth):
    """Condition on x[0], stationary initial log volatility, no hidden clipping."""
    if n<3 or abs(truth['b'])>=1 or abs(truth['phi'])>=1 or truth['q']<=0:
        raise ValueError('Invalid synthetic stationary model')
    rng=np.random.default_rng(seed)
    x=np.empty(n); h=np.empty(n-1)
    x[0]=truth['a']/(1-truth['b'])
    h[0]=truth['omega']+np.sqrt(truth['q']/(1-truth['phi']**2))*rng.normal()
    for i in range(n-1):
        if i>0:
            h[i]=truth['omega']+truth['phi']*(h[i-1]-truth['omega'])+np.sqrt(truth['q'])*rng.normal()
        x[i+1]=truth['a']+truth['b']*x[i]+np.exp(h[i]/2)*rng.normal()
    return x,h


def run_synthetic(out,warmup=1000,draws=2000):
    """One fixed synthetic recovery check, not a calibration simulation study."""
    from pathlib import Path
    from dataclasses import asdict
    import json,hashlib,time
    from src.bayesian_stage_a import diagnose
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    truth={'a':.04,'b':.85,'omega':-3.,'phi':.94,'q':.08}
    x,h=simulate_sv(240,30261027,truth)
    prior=br.freeze_priors(np.column_stack([x,x,x]))[0]
    np.savez_compressed(out/'data.npz',x=x,h=h)
    results=[br.sample_chain(x,prior,'M5',np.random.default_rng(81261027+k),
                             warmup,draws,kernel='collapsed') for k in range(4)]
    values=np.stack([r.pop('draws') for r in results]); names=results[0]['names']
    np.savez_compressed(out/'draws.npz',draws=values,names=np.array(names))
    quantiles=np.quantile(values.reshape(-1,len(names)),[.025,.5,.975],axis=0)
    recovery=[{'parameter':name,'truth':truth[name],'q025':float(quantiles[0,i]),
               'median':float(quantiles[1,i]),'q975':float(quantiles[2,i]),
               'truth_in_95_interval':bool(quantiles[0,i]<=truth[name]<=quantiles[2,i])}
              for i,name in enumerate(names[:5])]
    rows=diagnose(values,names)
    source=Path(br.__file__).read_bytes()
    (out/'executed_sampler.py.txt').write_bytes(source)
    (out/'executed_checks.py.txt').write_bytes(Path(__file__).read_bytes())
    report={'truth':truth,'data_seed':30261027,'chain_seed_rule':'81261027+chain_index',
            'n':240,'warmup':warmup,'draws':draws,'chains':4,'prior':asdict(prior),
            'recovery':recovery,'diagnostics':rows,'chain_reports':results,
            'parameter_gate_passed':bool(rows and all(r['passes'] for r in rows)),
            'accepted_for_rolling':False,'moment_audit':moment_audit(prior,len(x)-1),
            'sampler_sha256':hashlib.sha256(source).hexdigest(),
            'elapsed_s':time.perf_counter()-started,
            'interpretation':'Single-data-set parameter recovery does not establish finite predictive moments or general calibration.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def moment_audit(prior,n_transitions):
    """Prove divergence using the operative stationary-path IG conditional.

    Conditional on any finite h, omega, phi and beta, q|rest is IG(A,B).
    E[exp(h_next)|q,rest] = exp(omega+phi*(h_last-omega)+q/2).
    An IG variable has no finite positive exponential moment. Nonnegative
    integration (Tonelli) implies infinite unconditional predictive variance.
    This is a property of the model, not a finite-chain convergence failure.
    """
    if n_transitions<1 or prior.q_shape<=0 or prior.q_scale<=0:
        raise ValueError('Need a proper positive IG prior and nonempty path')
    h=np.zeros(n_transitions)
    shape,_=br.q_conditional(h,0.,0.,prior.q_shape,prior.q_scale)
    return {'q_prior':'inverse_gamma','conditional_ig_shape':shape,
            'conditional_ig_scale_lower_bound':prior.q_scale,
            'tilt_coefficient':.5,'tail_threshold_q':4*(shape+1),
            'log_tilted_density_derivative_lower_bound':.25,
            'finite_predictive_second_moment':False,
            'stage_a_mathematical_gate_passed':False,
            'proof':'For q>=4(A+1), d/dq log[exp(q/2) IG(q;A,B)] = 1/2-(A+1)/q+B/q^2 >=1/4. The integral diverges for every A,B>0.',
            'consequence':'M5 true posterior-predictive covariance is not finite already at h=1; finite empirical covariance does not fix the target. h=4 also contains a final SV innovation with this property.',
            'm4_contrast':'M4 log variance has a Gaussian prior and no random inverse-gamma log-volatility innovation variance; this divergence argument does not apply to M4.'}
