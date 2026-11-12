"""Limited, prespecified sensitivity planning; no sampler runs at import or planning."""
from __future__ import annotations


def sensitivity_cases(plan):
    """Six representative-only M5 cases; retain the cached matched M4 forecasts."""
    origins=plan['origins']
    representatives=sorted([row for row in origins if row['chain_source']],key=lambda row:row['index'])
    if len(origins)!=130 or len({row['index'] for row in origins})!=130 or len(representatives)!=3:
        raise ValueError('Require 130 origins and exactly three representative sources')
    frozen={'warmup':3000,'draws':6000,'chains':4,'scenario_count':2048}
    if any(plan[key]!=value for key,value in frozen.items()):
        raise ValueError('Changed main sampling specification')
    return [dict(index=row['index'],date=row['date'],v_upper=upper,
                 seed=73102700+int(upper)*10000+row['index'],
                 name=f"U{int(upper)}_{row['date']}",models=['M5'],reuse_m4_predictive=True,
                 matched_main_chain_source=row['chain_source'],**frozen)
            for upper in [2.,8.] for row in representatives]


def require_sensitivity_main(output,gate_path=None):
    """Fail closed before any sensitivity side effects; return read-only archives."""
    from .bayesian_finite_analysis import load_complete_experiment
    return load_complete_experiment(output,gate_path)


def source_predictions(source,v_upper,scenario_seeds=None):
    """Read accepted M5 checkpoints; regenerate predictions, never sample parameters.

    U=4 is supported here only for exact replay testing against main archives.
    The actual sensitivity case plan contains only U=2 and U=8.
    """
    from dataclasses import asdict
    import hashlib,json
    from pathlib import Path
    import numpy as np
    from .bayesian_finite import FinitePrior,moment_audit
    from .bayesian_finite_runner import digest
    from .bayesian_finite_validation import predictive_record
    from .bayesian_stage_a import FACTORS,diagnose,make_design
    path=Path(source); cfg=json.loads((path/'configuration.json').read_text())
    summary=json.loads((path/'summary.json').read_text()); design=make_design()
    if (summary['configuration']!=cfg or not summary['parameter_gate_passed'] or not summary['moment_gate_passed']
        or cfg['chains']!=4 or cfg['draws']!=6000 or cfg['warmup']!=3000 or 'M5' not in cfg['models']):
        raise ValueError('Unaccepted sensitivity parameter source')
    expected=[asdict(FinitePrior(p,v_upper=v_upper)) for p in design['priors']]
    if (json.loads(json.dumps(expected))!=cfg['priors']
        or cfg['input_sha256']!=design['input_sha256'] or cfg['lambda']!=design['lambda']):
        raise ValueError('Changed sensitivity prior or training design')
    window=next((w for w in design['windows'] if w==cfg['window']),None)
    if window is None: raise ValueError('Unmatched representative window')
    for filename,key in [('executed_sampler.py.txt','sampler_sha256'),('executed_base_sampler.py.txt','base_sampler_sha256'),('executed_runner.py.txt','runner_sha256')]:
        if digest(path/filename)!=cfg[key]: raise ValueError('Changed executed sensitivity source')
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    factors=[]; seeds=[]
    for j,name in enumerate(FACTORS):
        values=[]; names=None
        for k in range(4):
            with np.load(path/f'M5_{name}_chain{k}.npz',allow_pickle=False) as archive:
                cr=json.loads(str(archive['report'])); value=archive['draws'].copy()
            if (cr['signature']!=signature or cr['seed']!=cfg['seed']+100000+1000*j+k
                or value.shape!=(6000,len(cr['names'])) or not np.isfinite(value).all()
                or (names is not None and names!=cr['names'])):
                raise ValueError('Changed M5 checkpoint specification')
            seeds.append(cr['seed']); names=cr['names']; values.append(value)
        stacked=np.stack(values); diagnostics=diagnose(stacked,names)
        if any(not r['passes'] or not np.isfinite([r['rhat'],r['ess_bulk'],r['ess_tail']]).all()
               or r['rhat']>1.01 or min(r['ess_bulk'],r['ess_tail'])<400 for r in diagnostics):
            raise ValueError('Sensitivity parameter convergence failed')
        factors.append((stacked,names))
    if len(set(seeds))!=12: raise ValueError('Nonindependent M5 sensitivity chains')
    if not all(moment_audit(FinitePrior(p,v_upper=v_upper),window['n_training']-1)['stage_a_mathematical_gate_passed'] for p in design['priors']):
        raise ValueError('Sensitivity moment gate failed')
    t=window['index']; start=design['first_index']; ff=design['factors'][start:t+1]
    reconstruction=np.maximum(np.mean((design['y'][start:t+1]-ff@design['load'].T)**2,axis=0),1e-6)
    records=[]; arrays={}
    for h in [1,4]:
        seed=scenario_seeds[h] if scenario_seeds is not None else cfg['seed']+701000+100*h
        observed=design['y'][t+h]
        row=predictive_record(factors,ff[-1],design['load'],reconstruction,observed,'M5',h,np.random.default_rng(seed))
        samples=row.pop('scenarios')
        if not row['passes']: raise ValueError('Sensitivity derived-prediction convergence failed')
        lower,upper=np.quantile(samples,[.05,.95],axis=0)
        row.update(scenario_seed=seed,origin=str(design['dates'][t]),target=str(design['dates'][t+h]),
                   squared_errors=((observed-np.asarray(row['mean']))**2).tolist(),
                   interval_lower_90=lower.tolist(),interval_upper_90=upper.tolist(),
                   coverage_90=((observed>=lower)&(observed<=upper)).tolist())
        tag=f'M5_h{h}'; arrays.update({tag+'_scenarios':samples,tag+'_mean':np.asarray(row['mean']),
                                    tag+'_covariance':np.asarray(row['covariance']),tag+'_observed':observed})
        records.append(row)
    return records,arrays


def _assemble_case(output,case,source,main_folder):
    """Archive an accepted case; caller gates production on complete main evidence.

    Tests may replay U=4 into scratch to exercise this path without new sampling.
    """
    import json
    from pathlib import Path
    import numpy as np
    from .bayesian_finite_analysis import audit_origin
    from .bayesian_finite_runner import digest
    from .bayesian_stage_a import FACTORS,make_design
    out=Path(output); source=Path(source); main_folder=Path(main_folder); design=make_design()
    snapshots={'executed_sampler.py.txt':'sampler_sha256',
               'executed_base_sampler.py.txt':'base_sampler_sha256','executed_runner.py.txt':'runner_sha256'}
    validation_hashes={name:digest(source/name) for name in ['configuration.json',*snapshots]}
    cfg=json.loads((source/'configuration.json').read_text())
    summary=json.loads((source/'summary.json').read_text())
    main=json.loads((main_folder/'forecast.json').read_text())
    if (not main['accepted'] or not summary['parameter_gate_passed'] or not summary['moment_gate_passed']
        or cfg!=summary['configuration'] or any(validation_hashes[name]!=cfg[key] for name,key in snapshots.items())
        or cfg['window']['index']!=case['index'] or cfg['window']['date']!=case['date']
        or cfg['seed']!=case['seed'] or any(p['v_upper']!=case['v_upper'] for p in cfg['priors'])
        or main['specification']['origin_index']!=case['index']):
        raise ValueError('Changed or unaccepted sensitivity case specification')
    audit_origin(main_folder,main,design,energy_check=True)
    hashes={str((source/f'M5_{factor}_chain{k}.npz').resolve()):digest(source/f'M5_{factor}_chain{k}.npz') for factor in FACTORS for k in range(4)}
    spec={'origin_index':case['index'],'case':case,'source':str(source.resolve()),
          'source_validation_sha256':validation_hashes,
          'source_summary_sha256':digest(source/'summary.json'),'main_forecast_sha256':digest(main_folder/'forecast.json'),
          'main_archive_sha256':digest(main_folder/'predictions.npz'),
          'module_sha256':digest(__file__),'validator_sha256':digest('src/bayesian_finite_validation.py')}
    def comparison(records):
        def values(row):
            return {'dss':float(row['dss']),'energy_score':float(row['energy_score']),
                    'rmse':float(np.sqrt(np.mean(row['squared_errors']))),'coverage_90':float(np.mean(row['coverage_90']))}
        result={}
        for h in [1,4]:
            m4=values(next(r for r in main['records'] if r['model']=='M4' and r['horizon']==h))
            u4=values(next(r for r in main['records'] if r['model']=='M5' and r['horizon']==h))
            bound=values(next(r for r in records if r['model']=='M5' and r['horizon']==h))
            result[str(h)]={'M4':m4,'M5_main_U4':u4,'M5_bound':bound,
                            'M5_minus_M4':{k:bound[k]-m4[k] for k in bound},
                            'bound_minus_main_U4':{k:bound[k]-u4[k] for k in bound}}
        return result
    destination=out/'forecast.json'
    if destination.exists():
        previous=json.loads(destination.read_text())
        if (not previous['accepted'] or previous['specification']!=spec or previous['chain_checkpoint_sha256']!=hashes
            or previous['prediction_archive_sha256']!=digest(out/'predictions.npz')):
            raise ValueError('Changed completed sensitivity archive')
        audit=audit_origin(out,previous,design,energy_check=True)
        if previous['archive_audit']!=audit or previous['comparisons']!=comparison(previous['records']):
            raise ValueError('Sensitivity reconstruction: changed audit or comparison')
        return previous
    if out.exists() and any(p.name!='chains' for p in out.iterdir()):
        raise ValueError('Incomplete unidentified sensitivity archive; preserve before repair')
    records,arrays=source_predictions(source,case['v_upper'])
    records=[dict(r) for r in main['records'] if r['model']=='M4']+records
    with np.load(main_folder/'predictions.npz',allow_pickle=False) as archive:
        arrays.update({key:archive[key].copy() for key in archive.files if key.startswith('M4_')})
    if (any(digest(p)!=h for p,h in hashes.items()) or digest(main_folder/'forecast.json')!=spec['main_forecast_sha256']
        or any(digest(source/name)!=sha for name,sha in validation_hashes.items())
        or digest(source/'summary.json')!=spec['source_summary_sha256']):
        raise ValueError('Sensitivity sources changed during prediction')
    out.mkdir(parents=True,exist_ok=True)
    partial=out/'predictions.partial.npz'; np.savez_compressed(partial,**arrays); partial.replace(out/'predictions.npz')
    report={'accepted':True,'scope':'limited representative-window bound sensitivity, not full-sample robustness',
            'specification':spec,'chain_checkpoint_sha256':hashes,'records':records,
            'parameter_diagnostics':[r for r in summary['diagnostics'] if r['model']=='M5'],
            'prediction_archive_sha256':digest(out/'predictions.npz'),'comparisons':comparison(records),
            'yield_units':'percentage points','rmse_basis_point_multiplier':100}
    report['archive_audit']=audit_origin(out,report,design,energy_check=True)
    partial=out/'forecast.partial.json'; partial.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8'); partial.replace(destination)
    return report


def _sensitivity_job(main_output,case,main_sha,module_sha,window_name):
    """Numerical worker; preserve failures and never silently extend a failed run."""
    import contextlib,json,traceback
    from pathlib import Path
    from .bayesian_finite_runner import digest,run_window
    main=Path(main_output); root=main.parent/'bound_sensitivity'; out=root/case['name']; source=out/'chains'
    failure=out/'failure.json'
    if failure.exists(): return json.loads(failure.read_text())
    try:
        if digest(main/'summary.json')!=main_sha or digest(__file__)!=module_sha:
            raise ValueError('Changed main evidence or sensitivity implementation')
        with (root/'logs'/f"{case['name']}.log").open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            if not (out/'forecast.json').exists() and not (source/'summary.json').exists():
                result=run_window(source,window_name,case['warmup'],case['draws'],['M5'],case['seed'],case['v_upper'])
                if not result['parameter_gate_passed'] or not result['moment_gate_passed']:
                    raise ValueError('Limited-bound parameter or moment gate failed; no automatic repair')
            report=_assemble_case(out,case,source,main/'origins'/case['date'])
        return {'name':case['name'],'accepted':bool(report['accepted']),
                'forecast_sha256':digest(out/'forecast.json')}
    except Exception as exc:
        result={'name':case['name'],'accepted':False,'error':str(exc),'traceback':traceback.format_exc()}
        out.mkdir(parents=True,exist_ok=True); failure.write_text(json.dumps(result,indent=2))
        return result


def run_limited_sensitivity(main_output,gate_path=None,workers=2):
    """Six prespecified M5-only runs after complete main acceptance; bounded resume."""
    import csv,hashlib,inspect,json
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    from .bayesian_finite_analysis import analyze_experiment
    from .bayesian_finite_runner import digest
    from .bayesian_stage_a import make_design
    if not isinstance(workers,int) or not 1<=workers<=4: raise ValueError('Need one through four numerical workers')
    main=Path(main_output).resolve()
    plan,main_summary,archives=require_sensitivity_main(main,gate_path)
    analyze_experiment(main,gate_path)
    cases:list[dict]=sensitivity_cases(plan); frozen_path=main.parent/'prior_bound_sensitivity_plan.json'
    frozen=json.loads(frozen_path.read_text())
    if (frozen['cases']!=cases or frozen['main_plan_sha256']!=digest(main/'plan.json')
        or not frozen['accepted_complete_main_required_before_sampling']
        or frozen['planner_definition_sha256']!=hashlib.sha256(inspect.getsource(sensitivity_cases).encode()).hexdigest()):
        raise ValueError('Changed prespecified bound-sensitivity plan')
    main_sha=digest(main/'summary.json'); module_sha=digest(__file__)
    spec={'cases':cases,'main_summary_sha256':main_sha,'main_analysis_sha256':digest(main/'analysis/summary.json'),
          'frozen_plan_sha256':digest(frozen_path),'module_sha256':module_sha,
          'scope':'Three representative windows only; not full-sample robustness and not OOS bound selection.'}
    out=main.parent/'bound_sensitivity'; declaration=out/'plan.json'
    if declaration.exists():
        if json.loads(declaration.read_text())!=spec: raise ValueError('Changed sensitivity experiment declaration')
    else:
        if out.exists() and any(out.iterdir()): raise ValueError('Unidentified sensitivity experiment directory')
        out.mkdir(parents=True,exist_ok=True); declaration.write_text(json.dumps(spec,indent=2))
        (out/'executed_sensitivity.py.txt').write_bytes(Path(__file__).read_bytes())
    (out/'logs').mkdir(exist_ok=True)
    design=make_design(); window_names={w['index']:w['name'] for w in design['windows']}
    prior_report=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
    frozen_forecasts=dict(prior_report.get('case_forecast_sha256',{}))
    completed={}; failures={}; verified_complete=False
    def save():
        accepted=bool(verified_complete and len(completed)==6 and not failures)
        report={'accepted_limited_sensitivity':accepted,'prior_bound_sensitivity_completed':accepted,
                'planned_cases':6,'completed_cases':len(completed),'completed_names':sorted(completed),
                'failed_cases':list(failures.values()),'main_summary_sha256':main_sha,
                'main_analysis_sha256':spec['main_analysis_sha256'],'plan_sha256':digest(declaration),
                'scope':spec['scope'],
                'case_forecast_sha256':dict(sorted(frozen_forecasts.items()))}
        partial=out/'summary.partial.json'; partial.write_text(json.dumps(report,indent=2,allow_nan=False)); partial.replace(out/'summary.json')
        return report
    def accept(result):
        if result['accepted']:
            expected=frozen_forecasts.setdefault(result['name'],result['forecast_sha256'])
            if result['forecast_sha256']!=expected:
                result={'name':result['name'],'accepted':False,'error':'Changed frozen sensitivity forecast digest'}
        if not result['accepted']:
            completed.pop(result['name'],None)
            folder=out/result['name']; folder.mkdir(parents=True,exist_ok=True)
            failure=folder/'failure.json'
            if failure.exists(): result=json.loads(failure.read_text())
            else:
                partial=folder/'failure.partial.json'; partial.write_text(json.dumps(result,indent=2,allow_nan=False))
                partial.replace(failure)
        (completed if result['accepted'] else failures)[result['name']]=result
        print(json.dumps({'case':result['name'],'accepted':result['accepted'],'completed_cases':len(completed)}),flush=True)
        save()
    # Revalidate completed cases without parameter sampling, and preserve prior failures.
    for result in prior_report.get('failed_cases',[]):
        if result.get('name') not in {case['name'] for case in cases} or result.get('accepted') is not False:
            raise ValueError('Invalid prior sensitivity failure record')
        accept(result)
    pending=[]
    for case in cases:
        forecast=out/case['name']/'forecast.json'
        if case['name'] in frozen_forecasts and (not forecast.is_file() or digest(forecast)!=frozen_forecasts[case['name']]):
            accept({'name':case['name'],'accepted':False,'error':'Changed frozen sensitivity forecast digest'})
        elif forecast.exists() or (out/case['name']/'failure.json').exists():
            accept(_sensitivity_job(str(main),case,main_sha,module_sha,window_names[case['index']]))
        else: pending.append(case)
    save()
    if failures: return save()
    if workers==1:
        for case in pending:
            accept(_sensitivity_job(str(main),case,main_sha,module_sha,window_names[case['index']]))
            if failures: break
    elif pending:
        iterator=iter(pending)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            inflight={}
            def submit(case):
                try: future=pool.submit(_sensitivity_job,str(main),case,main_sha,module_sha,window_names[case['index']])
                except Exception as exc:
                    accept({'name':case['name'],'accepted':False,'error':repr(exc)})
                    return False
                inflight[future]=case
                return True
            for _ in range(min(workers,len(pending))):
                if not submit(next(iterator)): break
            while inflight:
                done,_=wait(inflight,return_when=FIRST_COMPLETED)
                for future in done:
                    case=inflight.pop(future)
                    try: result=future.result()
                    except Exception as exc: result={'name':case['name'],'accepted':False,'error':repr(exc)}
                    accept(result)
                if not failures:
                    for _ in range(workers-len(inflight)):
                        case=next(iterator,None)
                        if case is None: break
                        if not submit(case): break
    if len(completed)==6 and not failures:
        require_sensitivity_main(main,gate_path)
        if digest(main/'summary.json')!=main_sha or digest(main/'analysis/summary.json')!=spec['main_analysis_sha256']:
            raise ValueError('Main evidence changed during sensitivity experiment')
        reports={}
        for case in cases:
            folder=out/case['name']
            try:
                if digest(folder/'forecast.json')!=frozen_forecasts[case['name']]:
                    raise ValueError('Changed frozen sensitivity forecast digest')
                reports[case['name']]=_assemble_case(folder,case,folder/'chains',main/'origins'/case['date'])
                if digest(folder/'forecast.json')!=frozen_forecasts[case['name']]:
                    raise ValueError('Sensitivity forecast changed during final revalidation')
            except Exception as exc:
                accept({'name':case['name'],'accepted':False,'error':repr(exc)})
                return save()
        # Never build comparisons from an unvalidated reread, or refresh a digest.
        for case in cases:
            if digest(out/case['name']/'forecast.json')!=frozen_forecasts[case['name']]:
                accept({'name':case['name'],'accepted':False,'error':'Sensitivity forecast changed during final revalidation'})
                return save()
        rows=[]
        for case in cases:
            report=reports[case['name']]
            for horizon,values in report['comparisons'].items():
                for metric in values['M5_bound']:
                    rows.append(dict(v_upper=case['v_upper'],origin=case['date'],horizon=int(horizon),metric=metric,
                                     **{key:value[metric] for key,value in values.items()}))
        with (out/'comparisons.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        verified_complete=True
    return save()


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--main',required=True)
    parser.add_argument('--gate'); parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args(); result=run_limited_sensitivity(args.main,args.gate,args.workers)
    print(json.dumps({key:result[key] for key in ['accepted_limited_sensitivity','planned_cases','completed_cases','failed_cases']}))
    raise SystemExit(0 if result['accepted_limited_sensitivity'] else 1)
