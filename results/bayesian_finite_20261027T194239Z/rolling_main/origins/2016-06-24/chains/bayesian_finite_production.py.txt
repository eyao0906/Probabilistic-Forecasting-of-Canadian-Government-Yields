"""Fail-closed entry gate for checkpointed finite-prior forecasts."""
from __future__ import annotations
import json
from pathlib import Path
from .bayesian_finite_runner import digest


def require_phase2(path):
    """Reject missing, incomplete, rejected or changed acceptance evidence."""
    report=json.loads(Path(path).read_text())
    required={'accepted','representative_pairs','synthetic_accepted','max_rhat','min_bulk_ess',
              'min_tail_ess','source_sha256','evidence_sha256','protected_changed'}
    if not required.issubset(report) or not report.get('accepted'):
        raise ValueError('Phase 2 has not been accepted')
    pairs=report['representative_pairs']
    if (len(pairs)!=3 or {p['window']['name'] for p in pairs}!={'initial','stress','late'}
        or not all(p['passes'] and p['seed_sets_are_disjoint'] for p in pairs)
        or not report['synthetic_accepted'] or report['max_rhat']>1.01
        or report['min_bulk_ess']<400 or report['min_tail_ess']<400 or report['protected_changed']):
        raise ValueError('Phase 2 diagnostic requirements are not met')
    for mapping in [report['source_sha256'],report['evidence_sha256']]:
        if not mapping: raise ValueError('Phase 2 has no source evidence')
        if any(not Path(p).is_file() or digest(p)!=sha for p,sha in mapping.items()):
            raise ValueError('Phase 2 sources or evidence have changed')
    return report


def checkpoint_origin(d,output,t,warmup=3000,draws=6000,seed=9102701,v_upper=4.,gate_sha256=None):
    """Checkpoint genuine chains; small test runs cannot authorize forecasts."""
    import hashlib,time,os
    from dataclasses import asdict
    import numpy as np
    from . import bayesian_finite as bf
    from .bayesian_stage_a import FACTORS,diagnose
    out=Path(output); start=time.perf_counter()
    priors=[bf.FinitePrior(p,v_upper=v_upper) for p in d['priors']]
    cfg=json.loads(json.dumps({'origin_index':int(t),'origin':str(d['dates'][t]),
        'input_sha256':d['input_sha256'],'lambda':float(d['lambda']),'first_index':int(d['first_index']),
        'priors':[asdict(p) for p in priors],'warmup':int(warmup),'draws':int(draws),'chains':4,
        'seed':int(seed),'gate_sha256':gate_sha256,'source_sha256':{p:digest(p) for p in
        ['src/bayesian_finite.py','src/bayesian_revised.py','src/bayesian_finite_production.py']}}))
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    configuration=out/'configuration.json'
    if configuration.exists():
        if json.loads(configuration.read_text())!=cfg: raise ValueError('Changed origin configuration')
    else:
        if out.exists() and any(out.iterdir()): raise ValueError('Unidentified nonempty output directory')
        out.mkdir(parents=True,exist_ok=True); configuration.write_text(json.dumps(cfg,indent=2))
        for p in cfg['source_sha256']:
            (out/(Path(p).name+'.txt')).write_bytes(Path(p).read_bytes())
    previous=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None
    diagnostics=[]; checkpoints={}; seeds=[]
    ff=d['factors'][d['first_index']:t+1]
    for mi,model in enumerate(['M4','M5']):
        for j,name in enumerate(FACTORS):
            chains=[]; names=None
            for k in range(4):
                path=out/f'{model}_{name}_chain{k}.npz'; chain_seed=seed+mi*100000+j*10000+k*100
                if previous is not None and digest(path)!=previous['checkpoints'][path.name]:
                    raise ValueError('Changed completed chain checkpoint')
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:
                        values=z['draws'].copy(); report=json.loads(str(z['report']))
                    if report['signature']!=signature or report['seed']!=chain_seed:
                        raise ValueError('Changed chain configuration')
                else:
                    report=bf.sample_chain(ff[:,j],priors[j],model,np.random.default_rng(chain_seed),warmup,draws)
                    values=report.pop('draws'); report.update(signature=signature,seed=chain_seed)
                    partial=out/(path.name+'.partial.npz')
                    np.savez_compressed(partial,draws=values,report=np.array(json.dumps(report,allow_nan=False)))
                    os.replace(partial,path)
                if values.shape!=(draws,len(report['names'])) or not np.isfinite(values).all():
                    raise ValueError('Invalid chain array')
                if names is not None and names!=report['names']: raise ValueError('Changed parameter names')
                names=report['names']; chains.append(values); checkpoints[path.name]=digest(path); seeds.append(chain_seed)
            records=diagnose(np.stack(chains),names)
            diagnostics.extend(dict(r,model=model,factor=name) for r in records)
    if len(seeds)!=24 or len(set(seeds))!=24: raise ValueError('Nonindependent origin seeds')
    audits=[bf.moment_audit(p,len(ff)-1) for p in priors]
    report={'configuration':cfg,'signature':signature,'checkpoints':checkpoints,'chain_seeds':seeds,
        'diagnostics':diagnostics,'parameter_gate_passed':all(r['passes'] for r in diagnostics),
        'moment_gate_passed':all(r['stage_a_mathematical_gate_passed'] for r in audits),
        'moment_audits':audits,'accepted_for_rolling':False,'elapsed_s':time.perf_counter()-start}
    if previous is not None: return previous
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def forecast_origin(gate_path,output,t,seed=None,chain_source=None):
    """Create/resume a single main-prior forecast only after Phase 2 acceptance."""
    import os,shutil
    from datetime import date
    import numpy as np
    from .bayesian_stage_a import make_design,FACTORS
    from .experiment import select_origins
    from .bayesian_finite_validation import predictive_record,validate_predictions
    gate=require_phase2(gate_path); gate_sha=digest(gate_path)
    d=make_design(); origins=sorted({i for i,h,target in select_origins(d['dates'],date(2016,1,1),date(2025,12,31),[1,4],4)})
    if len(origins)!=130 or t not in origins: raise ValueError('Not an original common forecast origin')
    if seed is None: seed=9102701+1000000*origins.index(t)
    out=Path(output); source=Path(chain_source) if chain_source is not None else out/'chains'
    spec={'origin_index':int(t),'origin':str(d['dates'][t]),'gate_sha256':gate_sha,
          'seed':seed,'input_sha256':d['input_sha256'],'lambda':d['lambda'],'v_upper':4.,
          'warmup':3000,'draws':6000,'chains':4,'scenario_count':2048,'horizons':[1,4],
          'chain_source':str(source.resolve()),'reused':chain_source is not None,
          'production_sha256':digest(__file__),'validator_sha256':digest('src/bayesian_finite_validation.py')}
    metadata=out/'forecast_specification.json'
    if metadata.exists():
        if json.loads(metadata.read_text())!=spec: raise ValueError('Changed forecast specification')
    else:
        if out.exists() and any(out.iterdir()): raise ValueError('Unidentified forecast directory')
        out.mkdir(parents=True,exist_ok=True); metadata.write_text(json.dumps(spec,indent=2))
        (out/'executed_production.py.txt').write_bytes(Path(__file__).read_bytes())
    completed=out/'forecast.json'
    if completed.exists():
        report=json.loads(completed.read_text())
        if report['specification']!=spec or digest(out/'predictions.npz')!=report['prediction_archive_sha256']:
            raise ValueError('Changed completed forecast')
        if any(digest(p)!=sha for p,sha in report['chain_checkpoint_sha256'].items()):
            raise ValueError('Changed completed posterior checkpoint')
        return report
    reused=None
    if chain_source is not None:
        reused=validate_predictions(source); cfg=reused['configuration']
        if (cfg['window']['index']!=t or cfg['draws']!=6000 or cfg['warmup']!=3000 or not reused['passes']
            or any(p['v_upper']!=4. for p in cfg['priors'])):
            raise ValueError('Incompatible representative checkpoint')
        summary=json.loads((source/'summary.json').read_text())
    else:
        summary=checkpoint_origin(d,source,t,seed=seed,gate_sha256=gate_sha)
    if not summary['parameter_gate_passed'] or not summary['moment_gate_passed']:
        raise ValueError('Origin parameter or mathematical gate failed; forecasts forbidden')
    ff=d['factors'][d['first_index']:t+1]; yy=d['y'][d['first_index']:t+1]
    reconstruction=np.maximum(np.mean((yy-ff@d['load'].T)**2,axis=0),1e-6)
    records=[]; archive={}; checkpoint_hashes={}
    for model in ['M4','M5']:
        factors=[]
        for factor in FACTORS:
            values=[]; names=None
            for k in range(4):
                path=source/f'{model}_{factor}_chain{k}.npz'
                with np.load(path,allow_pickle=False) as z:
                    value=z['draws'].copy(); cr=json.loads(str(z['report']))
                if value.shape!=(6000,len(cr['names'])) or not np.isfinite(value).all(): raise ValueError('Invalid posterior checkpoint')
                if names is not None and names!=cr['names']: raise ValueError('Changed names')
                names=cr['names']; values.append(value); checkpoint_hashes[str(path.resolve())]=digest(path)
            factors.append((np.stack(values),names))
        for horizon in [1,4]:
            tag=f'{model}_h{horizon}'; observed=d['y'][t+horizon]
            if reused is not None:
                record=dict(next(r for r in reused['records'] if r['model']==model and r['horizon']==horizon))
                with np.load(source/'predictive_validation'/'predictions.npz',allow_pickle=False) as z:
                    samples=z[f'{tag}_scenarios'].copy()
            else:
                scenario_seed=seed+700000+1000*(model=='M5')+horizon*100
                record=predictive_record(factors,ff[-1],d['load'],reconstruction,observed,model,horizon,np.random.default_rng(scenario_seed))
                samples=record.pop('scenarios'); record['scenario_seed']=scenario_seed
            if not record['passes']: raise ValueError('Origin derived-prediction diagnostic gate failed')
            lower,upper=np.quantile(samples,[.05,.95],axis=0)
            record.update(origin=str(d['dates'][t]),target=str(d['dates'][t+horizon]),
                          squared_errors=((observed-np.asarray(record['mean']))**2).tolist(),
                          interval_lower_90=lower.tolist(),interval_upper_90=upper.tolist(),
                          coverage_90=((observed>=lower)&(observed<=upper)).tolist())
            archive[f'{tag}_scenarios']=samples; archive[f'{tag}_mean']=np.asarray(record['mean'])
            archive[f'{tag}_covariance']=np.asarray(record['covariance']); archive[f'{tag}_observed']=observed
            records.append(record)
    partial=out/'predictions.partial.npz'; np.savez_compressed(partial,**archive); os.replace(partial,out/'predictions.npz')
    report={'accepted':True,'specification':spec,'records':records,'reconstruction_variance':reconstruction.tolist(),
            'prediction_archive_sha256':digest(out/'predictions.npz'),'chain_checkpoint_sha256':checkpoint_hashes,
            'parameter_summary_sha256':digest(source/'summary.json'),
            'parameter_diagnostics':summary['diagnostics'],'chain_source':str(source.resolve())}
    temporary=out/'forecast.partial.json'; temporary.write_text(json.dumps(report,indent=2,allow_nan=False)); os.replace(temporary,completed)
    return report
