"""Synthetic and predictive validation for the finite stationary-variance prior."""
from pathlib import Path
from dataclasses import asdict
import json
import time
import numpy as np
from src import bayesian_finite as bf
from src import bayesian_revised as br
from src.bayesian_stage_a import diagnose
from src.bayesian_stage_a_checks import simulate_sv


def factor_conditionals(values,names,x_last,horizon,model):
    """Exact finite-horizon conditional means/variances for each posterior draw."""
    if horizon not in [1,4] or model not in ['M4','M5']: raise ValueError('Invalid forecast specification')
    values=np.asarray(values); get=lambda name: values[...,names.index(name)]
    a,b=get('a'),get('b')
    mean=b**horizon*x_last+a*sum(b**k for k in range(horizon))
    variance=np.zeros_like(mean)
    with np.errstate(over='raise',invalid='raise'):
        for k in range(1,horizon+1):
            if model=='M4': expected=np.exp(get('log_variance'))
            else:
                ph=get('phi')**k; v=get('v'); w=get('m')-v/2
                expected=np.exp(w+ph*(get('h_last')-w)+v*(1-ph**2)/2)
            variance+=b**(2*(horizon-k))*expected
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance<=0):
        raise ValueError('Invalid analytic predictive moments; no clipping permitted')
    return mean,variance


def factor_scenarios(values,names,x_last,horizon,model,rng,size):
    """One independent posterior index per scenario, with parameters fixed over its path."""
    values=np.asarray(values).reshape(-1,len(names)); chosen=values[rng.integers(len(values),size=size)]
    get=lambda name: chosen[:,names.index(name)]
    a,b=get('a'),get('b'); x=np.full(size,x_last,dtype=float)
    if model=='M5':
        v=get('v'); w=get('m')-v/2; phi=get('phi'); q=v*(1-phi**2); h=get('h_last').copy()
    elif model=='M4': h=get('log_variance')
    else: raise ValueError('Invalid model')
    with np.errstate(over='raise',invalid='raise'):
        for _ in range(horizon):
            if model=='M5': h=w+phi*(h-w)+np.sqrt(q)*rng.normal(size=size)
            x=a+b*x+np.exp(h/2)*rng.normal(size=size)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite predictive scenarios')
    return x


def yield_moments(conditional_means,conditional_variances,load,reconstruction):
    """Law of total variance; separate factor posteriors, never empirical cross-factor covariance."""
    means=np.array([np.mean(x) for x in conditional_means])
    variances=np.array([np.mean(v)+np.var(m,ddof=0) for m,v in zip(conditional_means,conditional_variances)])
    load=np.asarray(load); reconstruction=np.asarray(reconstruction)
    return load@means,(load*variances)@load.T+np.diag(reconstruction),means,variances


def predictive_record(factors,x_last,load,reconstruction,observed,model,horizon,rng,size=2048,batches=80):
    """Diagnostic forecasts only: analytic moments, proper ensemble ES and block MCSEs."""
    from src.scores import dss,energy_score
    from scipy.spatial.distance import cdist
    cond=[factor_conditionals(v,n,x_last[j],horizon,model) for j,(v,n) in enumerate(factors)]
    cm,cv=map(list,zip(*cond)); mean,cov,fm,fv=yield_moments(cm,cv,load,reconstruction)
    score=dss(observed,mean,cov); error=np.asarray(observed)-mean
    sign,ld=np.linalg.slogdet(cov); independent=float(ld+error@np.linalg.solve(cov,error))
    eigmin=float(np.linalg.eigvalsh(cov).min())
    reconstruction_ok=bool(sign>0 and eigmin>0 and np.isclose(score,independent,rtol=1e-9,atol=1e-9))
    derived=[]
    for j,(m,v) in enumerate(cond):
        derived.extend({'factor':j,**row} for row in diagnose(np.stack([m,v],axis=-1),['conditional_mean','conditional_variance']))
    nchain,ndraw=cm[0].shape; per_chain=batches//nchain
    if batches%nchain or ndraw%per_chain: raise ValueError('Need equal contiguous chain batches')
    chunk=ndraw//per_chain; batch_dss=[]; batch_es=[]
    def scenarios(fs,count):
        fd=np.column_stack([factor_scenarios(v,n,x_last[j],horizon,model,rng,count) for j,(v,n) in enumerate(fs)])
        return fd@np.asarray(load).T+rng.normal(size=(count,len(reconstruction)))*np.sqrt(reconstruction)
    samples=scenarios(factors,size)
    es=energy_score(samples,observed)
    # Jackknife the actual U-statistic ensemble, not an iid standard error of pair distances.
    distances=cdist(samples,samples); total=distances.sum()/2
    g=np.linalg.norm(samples-observed,axis=1)
    leave=(g.sum()-g)/(size-1)-(total-distances.sum(axis=1))/((size-1)*(size-2))
    es_jackknife=float(np.sqrt((size-1)/size*np.sum((leave-leave.mean())**2)))
    for chain in range(nchain):
        for b in range(per_chain):
            sl=slice(b*chunk,(b+1)*chunk)
            bm,bc,_,_=yield_moments([m[chain,sl] for m in cm],[v[chain,sl] for v in cv],load,reconstruction)
            batch_dss.append(dss(observed,bm,bc))
            block=[(v[chain,sl],names) for v,names in factors]
            batch_es.append(energy_score(scenarios(block,32),observed))
    dss_mcse=float(np.std(batch_dss,ddof=1)/np.sqrt(batches))
    es_batch_mcse=float(np.std(batch_es,ddof=1)/np.sqrt(batches))
    # Conservative quadrature includes independent diagnostic-batch simulation plus actual-ensemble error.
    es_mcse=float(np.hypot(es_jackknife,es_batch_mcse))
    return {'model':model,'horizon':horizon,'mean':mean.tolist(),'covariance':cov.tolist(),
            'factor_mean':fm.tolist(),'factor_variance':fv.tolist(),'dss':score,'dss_reconstructed':independent,
            'energy_score':es,'dss_mcse':dss_mcse,'energy_mcse':es_mcse,
            'energy_jackknife_mcse':es_jackknife,'energy_batch_mcse':es_batch_mcse,
            'batch_count':batches,'batch_dss':batch_dss,'batch_energy':batch_es,
            'minimum_eigenvalue':eigmin,'score_reconstruction_passed':reconstruction_ok,
            'derived_diagnostics':derived,'passes':bool(reconstruction_ok and all(r['passes'] for r in derived)),
            'scenarios':samples}


def prediction_difference(first,second):
    """Predeclared seed-stability checks; provenance/seed independence checked by caller."""
    if (first['model'],first['horizon'])!=(second['model'],second['horizon']): raise ValueError('Unmatched predictions')
    c1,c2=np.asarray(first['covariance']),np.asarray(second['covariance'])
    mean_difference=float(np.max(abs(np.asarray(first['mean'])-second['mean'])/np.sqrt((np.diag(c1)+np.diag(c2))/2)))
    cov_difference=float(np.linalg.norm(c1-c2)/np.linalg.norm((c1+c2)/2))
    v1,v2=np.asarray(first['factor_variance']),np.asarray(second['factor_variance'])
    factor_difference=float(np.max(abs(v1-v2)/((v1+v2)/2)))
    dss_difference=abs(first['dss']-second['dss']); es_difference=abs(first['energy_score']-second['energy_score'])
    dss_limit=max(.25,3*np.hypot(first['dss_mcse'],second['dss_mcse']))
    es_limit=max(.01,3*np.hypot(first['energy_mcse'],second['energy_mcse']))
    passed=bool(first['passes'] and second['passes'] and mean_difference<=.05 and cov_difference<=.05
                and factor_difference<=.10 and dss_difference<=dss_limit and es_difference<=es_limit)
    return {'model':first['model'],'horizon':first['horizon'],'max_standardized_mean_difference':mean_difference,
            'relative_covariance_difference':cov_difference,'max_relative_factor_variance_difference':factor_difference,
            'absolute_dss_difference':dss_difference,'dss_limit':float(dss_limit),
            'absolute_energy_difference':es_difference,'energy_limit':float(es_limit),'passes':passed}


def validate_predictions(chains_dir):
    """Read exact accepted checkpoints, save diagnostic predictions, resume without sampling."""
    import hashlib
    from src.bayesian_finite_runner import digest
    from src.bayesian_stage_a import make_design,FACTORS
    path=Path(chains_dir); summary=json.loads((path/'summary.json').read_text())
    if not (summary.get('parameter_gate_passed') and summary.get('moment_gate_passed')):
        raise ValueError('Unaccepted chains: diagnostic predictions forbidden')
    cfg=json.loads((path/'configuration.json').read_text())
    if summary['configuration']!=cfg or cfg['chains']!=4 or sorted(cfg['models'])!=['M4','M5']:
        raise ValueError('Invalid source configuration')
    if not all(r['passes'] for r in summary['diagnostics']): raise ValueError('Unaccepted parameter diagnostics')
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    for filename,key in [('executed_sampler.py.txt','sampler_sha256'),('executed_base_sampler.py.txt','base_sampler_sha256'),('executed_runner.py.txt','runner_sha256')]:
        if digest(path/filename)!=cfg[key]: raise ValueError('Executed source hash mismatch')
    checkpoints={f'{model}_{name}_chain{k}.npz':digest(path/f'{model}_{name}_chain{k}.npz')
                 for model in cfg['models'] for name in FACTORS for k in range(4)}
    provenance={'configuration_sha256':signature,'checkpoints':checkpoints,'validator_sha256':digest(__file__)}
    out=path/'predictive_validation'
    if out.exists():
        report=json.loads((out/'summary.json').read_text())
        if report['provenance']!=provenance: raise ValueError('Cannot resume changed predictive sources')
        if digest(out/'predictions.npz')!=report['prediction_archive_sha256']: raise ValueError('Changed predictions')
        return report
    d=make_design(); w=next(w for w in d['windows'] if w==cfg['window'])
    if cfg['input_sha256']!=d['input_sha256'] or cfg['lambda']!=d['lambda']: raise ValueError('Changed data/design')
    expected=[asdict(bf.FinitePrior(p,v_upper=cfg['priors'][j]['v_upper'],v_rate=cfg['priors'][j]['v_rate'])) for j,p in enumerate(d['priors'])]
    if json.loads(json.dumps(expected))!=cfg['priors']: raise ValueError('Changed frozen priors')
    t=w['index']; start=d['first_index']; yy=d['y'][start:t+1]; ff=d['factors'][start:t+1]
    reconstruction=np.maximum(np.mean((yy-ff@d['load'].T)**2,axis=0),1e-6)
    records=[]; archive={}; seeds=[]
    for model in cfg['models']:
        factors=[]
        for name in FACTORS:
            values=[]; names=None
            for k in range(4):
                with np.load(path/f'{model}_{name}_chain{k}.npz',allow_pickle=False) as z:
                    cr=json.loads(str(z['report'])); value=z['draws'].copy()
                if cr['signature']!=signature or value.shape!=(cfg['draws'],len(cr['names'])):
                    raise ValueError('Checkpoint configuration/shape mismatch')
                if not np.isfinite(value).all(): raise ValueError('Nonfinite posterior draws')
                if names is not None and names!=cr['names']: raise ValueError('Inconsistent parameter order')
                names=cr['names']; values.append(value); seeds.append(cr['seed'])
            factors.append((np.stack(values),names))
        for h in [1,4]:
            scenario_seed=cfg['seed']+700000+10000*['initial','stress','late'].index(w['name'])+1000*(model=='M5')+h*100
            observed=d['y'][t+h]
            r=predictive_record(factors,ff[-1],d['load'],reconstruction,observed,model,h,np.random.default_rng(scenario_seed))
            tag=f'{model}_h{h}'; archive[f'{tag}_scenarios']=r.pop('scenarios')
            archive[f'{tag}_mean']=np.array(r['mean']); archive[f'{tag}_covariance']=np.array(r['covariance'])
            archive[f'{tag}_observed']=observed; r['scenario_seed']=scenario_seed; records.append(r)
    if len(set(seeds))!=24: raise ValueError('Non-independent factor/chain seeds')
    out.mkdir(); np.savez_compressed(out/'predictions.npz',**archive)
    report={'provenance':provenance,'chain_directory':str(path.resolve()),'configuration':cfg,
            'chain_seeds':seeds,'records':records,'reconstruction_variance':reconstruction.tolist(),
            'passes':bool(len(records)==4 and all(r['passes'] for r in records)),
            'prediction_archive_sha256':digest(out/'predictions.npz'),
            'status':'representative-window numerical diagnostics only; not rolling forecast evidence'}
    (out/'executed_validator.py.txt').write_bytes(Path(__file__).read_bytes())
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def run_synthetic(out,warmup=1000,draws=2000,v_upper=4.):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    truth={'a':.04,'b':.85,'omega':-3.,'phi':.94,'q':.08}
    truth['v']=truth['q']/(1-truth['phi']**2); truth['m']=truth['omega']+truth['v']/2
    x,h=simulate_sv(n=240,seed=30261027,truth=truth)
    prior=bf.FinitePrior(br.freeze_priors(np.column_stack([x,x,x]))[0],v_upper=v_upper)
    np.savez_compressed(out/'data.npz',x=x,h=h)
    from src.bayesian_finite_runner import digest
    cfg={'warmup':warmup,'draws':draws,'chains':4,'truth':truth,'data_seed':30261027,
         'chain_seeds':[81261027+k for k in range(4)],'prior':asdict(prior),
         'sampler_sha256':digest(bf.__file__),'base_sampler_sha256':digest(br.__file__),
         'generator_sha256':digest(Path(__file__).with_name('bayesian_stage_a_checks.py'))}
    (out/'configuration.json').write_text(json.dumps(cfg,indent=2))
    (out/'executed_sampler.py.txt').write_bytes(Path(bf.__file__).read_bytes())
    start=time.perf_counter(); values=[]; chain_reports=[]
    for seed in cfg['chain_seeds']:
        r=bf.sample_chain(x,prior,'M5',np.random.default_rng(seed),warmup,draws)
        values.append(r.pop('draws')); chain_reports.append({'seed':seed,**r})
    values=np.stack(values); names=chain_reports[0]['names']
    np.savez_compressed(out/'draws.npz',draws=values,names=np.array(names))
    diagnostics=diagnose(values,names); recovery=[]
    for name,value in truth.items():
        q=np.quantile(values[:,:,names.index(name)],[.025,.5,.975])
        recovery.append({'parameter':name,'truth':value,'lower':float(q[0]),'median':float(q[1]),
                         'upper':float(q[2]),'contains_truth':bool(q[0]<=value<=q[2])})
    report={'configuration':cfg,'diagnostics':diagnostics,'recovery':recovery,'chain_reports':chain_reports,
            'parameter_gate_passed':all(r['passes'] for r in diagnostics),'accepted_for_rolling':False,
            'moment_audit':bf.moment_audit(prior,len(x)-1),
            'status':'one fixed dataset; interval misses do not establish bias or justify tuning',
            'elapsed_s':time.perf_counter()-start}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report
