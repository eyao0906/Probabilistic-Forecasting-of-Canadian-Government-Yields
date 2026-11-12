"""Paired dependence-aware analysis of accepted finite-prior forecast archives."""
from __future__ import annotations
import numpy as np


def stationary_blocks(n,block_length,reps,rng):
    """Stationary bootstrap: geometric block lengths on a circular sample."""
    from .scores import stationary_bootstrap_indices
    if not isinstance(n,int) or n<2 or not 1<=block_length<=n or reps<2:
        raise ValueError('Invalid stationary block specification')
    return np.stack([stationary_bootstrap_indices(n,block_length,rng) for _ in range(reps)])


def paired_summary(m4,m5,reps=9999,block_length=8,seed=271026):
    """Percentile intervals for paired score means, pooled RMSE and coverage."""
    keys=['dss','energy_score','squared_errors','coverage_90']
    values=[]
    n=len(m4['dss'])
    for data in [m4,m5]:
        arrays={key:np.asarray(data[key],dtype=float) for key in keys}
        if (any(a.shape[0]!=n or not np.isfinite(a).all() for a in arrays.values())
            or arrays['dss'].shape!=(n,) or arrays['energy_score'].shape!=(n,)
            or arrays['squared_errors'].ndim!=2 or arrays['coverage_90'].shape!=arrays['squared_errors'].shape
            or np.any(arrays['squared_errors']<0)
            or np.any((arrays['coverage_90']!=0)&(arrays['coverage_90']!=1))):
            raise ValueError('Invalid or unmatched paired arrays')
        values.append(arrays)
    if values[0]['squared_errors'].shape!=values[1]['squared_errors'].shape:
        raise ValueError('Unmatched maturity counts')
    indices=stationary_blocks(n,block_length,reps,np.random.default_rng(seed))
    metrics={}
    for metric,key in [('dss','dss'),('energy_score','energy_score'),('rmse','squared_errors'),('coverage_90','coverage_90')]:
        estimates=[]; bootstrap=[]
        for arrays in values:
            a=arrays[key]
            estimate=float(a.mean()); sampled=a[indices]; sampled=sampled.mean(axis=tuple(range(1,sampled.ndim)))
            if metric=='rmse': estimate=float(np.sqrt(estimate)); sampled=np.sqrt(sampled)
            estimates.append(estimate); bootstrap.append(sampled)
        difference=bootstrap[1]-bootstrap[0]
        metrics[metric]={'M4':estimates[0],'M5':estimates[1],
                         'M5_minus_M4':estimates[1]-estimates[0],
                         'ci95':np.quantile(difference,[.025,.975]).tolist(),
                         'bootstrap_se':float(np.std(difference,ddof=1))}
    return {'n_origins':n,'maturities':values[0]['squared_errors'].shape[1],
            'method':'paired stationary percentile bootstrap','block_length_origins':block_length,
            'repetitions':reps,'seed':seed,'confidence':.95,'metrics':metrics}


def read_origin(folder,row,plan):
    """Read-only checks: analysis must never call a sampling/resume entry point."""
    import json
    from pathlib import Path
    from .bayesian_finite_runner import digest
    from .bayesian_stage_a import FACTORS
    folder=Path(folder); report=json.loads((folder/'forecast.json').read_text())
    spec=report['specification']; declared=json.loads((folder/'forecast_specification.json').read_text())
    source=Path(row['chain_source']) if row['chain_source'] else folder/'chains'
    shared=['gate_sha256','input_sha256','lambda','v_upper','warmup','draws','chains','scenario_count','horizons','production_sha256']
    if (spec!=declared or any(spec[k]!=plan[k] for k in shared)
        or spec['origin_index']!=row['index'] or spec['origin']!=row['date'] or spec['seed']!=row['seed']
        or Path(spec['chain_source']).resolve()!=source.resolve() or Path(report['chain_source']).resolve()!=source.resolve()
        or spec['reused']!=bool(row['chain_source'])
        or spec['validator_sha256']!=digest('src/bayesian_finite_validation.py')):
        raise ValueError('Changed forecast specification')
    expected={str((source/f'{m}_{f}_chain{k}.npz').resolve()) for m in ['M4','M5'] for f in FACTORS for k in range(4)}
    checkpoints=report['chain_checkpoint_sha256']
    summary=json.loads((source/'summary.json').read_text())
    if (not report['accepted'] or not summary['parameter_gate_passed'] or not summary['moment_gate_passed']
        or digest(source/'summary.json')!=report['parameter_summary_sha256']
        or digest(folder/'predictions.npz')!=report['prediction_archive_sha256']
        or set(checkpoints)!=expected or any(digest(source/name)!=sha for name,sha in checkpoints.items())
        or summary['diagnostics']!=report['parameter_diagnostics']):
        raise ValueError('Changed or rejected posterior/predictive archive')
    diags=report['parameter_diagnostics']
    if (not diags or any(not d['passes'] or not np.isfinite([d['rhat'],d['ess_bulk'],d['ess_tail']]).all()
                       or d['rhat']>1.01 or min(d['ess_bulk'],d['ess_tail'])<400 for d in diags)):
        raise ValueError('Unaccepted parameter diagnostics')
    return report


def load_complete_experiment(output,gate_path=None):
    """Read only a complete, frozen, accepted 130-origin main experiment."""
    import json
    from pathlib import Path
    from .bayesian_finite_experiment import make_plan

    from .bayesian_finite_runner import digest
    root=Path(output); summary=json.loads((root/'summary.json').read_text())
    if (not summary.get('accepted') or summary.get('planned_origins')!=130
        or summary.get('completed_origins')!=130 or summary.get('failed_origins')
        or summary.get('not_completed_dates')):
        raise ValueError('Incomplete or unaccepted 130-origin experiment')
    gate_path=Path(gate_path) if gate_path is not None else root.parent/'phase2_gate.json'
    plan=make_plan(gate_path)
    if (json.loads((root/'plan.json').read_text())!=plan
        or summary['plan_sha256']!=digest(root/'plan.json')
        or set(summary['completed_dates'])!={row['date'] for row in plan['origins']}
        or len(summary['completed_dates'])!=130
        or set(summary['forecasts_sha256'])!=set(summary['completed_dates'])):
        raise ValueError('Changed or incomplete frozen experiment')
    records=[]
    for row in plan['origins']:
        folder=root/'origins'/row['date']
        if summary['forecasts_sha256'][row['date']]!=digest(folder/'forecast.json'):
            raise ValueError('Changed accepted forecast record')
        # Existing forecasts only; absence must never trigger new sampling here.
        if not (folder/'forecast.json').is_file():
            raise ValueError('Incomplete 130-origin forecast archives')
        report=read_origin(folder,row,plan)
        if (not report['accepted'] or len(report['records'])!=4
            or {(r['model'],r['horizon']) for r in report['records']}!={('M4',1),('M4',4),('M5',1),('M5',4)}
            or any(not d['passes'] or not np.isfinite([d['rhat'],d['ess_bulk'],d['ess_tail']]).all()
                   or d['rhat']>1.01 or min(d['ess_bulk'],d['ess_tail'])<400 for d in report['parameter_diagnostics'])
            or not report['parameter_diagnostics']):
            raise ValueError('Unaccepted or unmatched origin evidence')
        source=Path(report['chain_source'])
        if digest(source/'summary.json')!=report['parameter_summary_sha256']:
            raise ValueError('Changed parameter summary')
        records.append((row,folder,report))
    return plan,summary,records


def audit_origin(folder,report,design,energy_check=False):
    """Independently reconstruct saved predictions; do not generate any new draws."""
    from pathlib import Path
    from scipy.spatial.distance import pdist
    t=report['specification']['origin_index']; eigmin=np.inf; dss_error=0.; energy_error=0.
    expected={(m,h) for m in ['M4','M5'] for h in [1,4]}
    if len(report['records'])!=4 or {(r['model'],r['horizon']) for r in report['records']}!=expected:
        raise ValueError('Archive reconstruction: unmatched forecast set')
    with np.load(Path(folder)/'predictions.npz',allow_pickle=False) as archive:
        for row in report['records']:
            tag=f"{row['model']}_h{row['horizon']}"
            samples=archive[tag+'_scenarios']; mean=archive[tag+'_mean']; cov=archive[tag+'_covariance']; y=archive[tag+'_observed']
            arrays=[samples,mean,cov,y]
            if (samples.shape!=(2048,8) or mean.shape!=(8,) or cov.shape!=(8,8) or y.shape!=(8,)
                or not all(np.isfinite(a).all() for a in arrays)
                or not np.array_equal(mean,row['mean']) or not np.array_equal(cov,row['covariance'])
                or not np.array_equal(y,design['y'][t+row['horizon']])
                or row['origin']!=str(design['dates'][t]) or row['target']!=str(design['dates'][t+row['horizon']])
                or not np.allclose(cov,cov.T,rtol=1e-12,atol=1e-12)):
                raise ValueError('Archive reconstruction: changed dimensions, values, targets or moments')
            np.linalg.cholesky(cov); minimum=float(np.linalg.eigvalsh(cov).min()); eigmin=min(eigmin,minimum)
            sign,ld=np.linalg.slogdet(cov); error=y-mean
            score=float(ld+error@np.linalg.solve(cov,error)); dss_error=max(dss_error,abs(score-row['dss']))
            lo,hi=np.quantile(samples,[.05,.95],axis=0); coverage=(y>=lo)&(y<=hi)
            derived=row['derived_diagnostics']
            if (not row['passes'] or not row['score_reconstruction_passed'] or sign<=0 or minimum<=0
                or not np.isclose(score,row['dss'],rtol=1e-9,atol=1e-9)
                or not np.isclose(score,row['dss_reconstructed'],rtol=1e-9,atol=1e-9)
                or not np.array_equal(error**2,row['squared_errors'])
                or not np.array_equal(lo,row['interval_lower_90']) or not np.array_equal(hi,row['interval_upper_90'])
                or not np.array_equal(coverage,row['coverage_90'])
                or len(derived)!=6 or any(not d['passes'] or not np.isfinite([d['rhat'],d['ess_bulk'],d['ess_tail']]).all()
                    or d['rhat']>1.01 or min(d['ess_bulk'],d['ess_tail'])<400 for d in derived)):
                raise ValueError('Archive reconstruction: scores, intervals or diagnostics failed')
            if energy_check:
                # pdist differs from production cdist + upper-triangle indexing.
                es=float(np.linalg.norm(samples-y,axis=1).mean()-.5*pdist(samples).mean())
                energy_error=max(energy_error,abs(es-row['energy_score']))
                if not np.isclose(es,row['energy_score'],rtol=1e-10,atol=1e-10):
                    raise ValueError('Archive reconstruction: Energy Score mismatch')
    return {'passes':True,'record_count':4,'energy_reconstruction_checked':bool(energy_check),
            'minimum_eigenvalue':eigmin,'maximum_dss_error':dss_error,'maximum_energy_error':energy_error}


def aggregate_records(records,reps=9999):
    """Pure aggregation; archive validation is the caller's separate prerequisite."""
    origins=sorted({r['origin'] for r in records})
    identities={(r['origin'],r['model'],r['horizon']) for r in records}
    required={(o,m,h) for o in origins for m in ['M4','M5'] for h in [1,4]}
    if len(origins)!=130 or len(records)!=520 or identities!=required:
        raise ValueError('Require all 130 matched origins, both models and horizons')
    result={'n_origins':130,'yield_units':'percentage points','rmse_basis_point_multiplier':100,
            'maturities_years':[.25,.5,1,2,5,7,10,30],'horizons':{}}
    for h in [1,4]:
        data={}
        for model in ['M4','M5']:
            rows=sorted([r for r in records if r['model']==model and r['horizon']==h],key=lambda r:r['origin'])
            data[model]={key:np.asarray([r[key] for r in rows],dtype=float)
                for key in ['dss','energy_score','squared_errors','coverage_90','interval_lower_90','interval_upper_90']}
            if any(data[model][k].shape!=(130,8) for k in ['squared_errors','coverage_90','interval_lower_90','interval_upper_90']):
                raise ValueError('Require 130 origins by eight maturities')
        horizon={'main':paired_summary(data['M4'],data['M5'],reps=reps),
                 'dependence_sensitivity':{str(b):paired_summary(data['M4'],data['M5'],reps=reps,block_length=b) for b in [4,12]},
                 'per_maturity':[]}
        for j,maturity in enumerate(result['maturities_years']):
            row={'maturity_years':maturity}
            for model in ['M4','M5']:
                a=data[model]
                row[model]={'rmse':float(np.sqrt(a['squared_errors'][:,j].mean())),
                            'coverage_90':float(a['coverage_90'][:,j].mean()),
                            'mean_interval_width_90':float((a['interval_upper_90'][:,j]-a['interval_lower_90'][:,j]).mean())}
            horizon['per_maturity'].append(row)
        result['horizons'][str(h)]=horizon
    return result


def analyze_experiment(output,gate_path=None):
    """No scientific outputs are written until all 130 immutable archives pass."""
    import csv,json
    from pathlib import Path
    from .bayesian_stage_a import make_design
    from .bayesian_finite_runner import digest
    root=Path(output); plan,summary,archives=load_complete_experiment(root,gate_path)
    design=make_design(); audits=[]; records=[]; diagnostics=[]
    for row,folder,report in archives:
        audits.append(dict(origin=row['date'],**audit_origin(folder,report,design,energy_check=bool(row['chain_source']))))
        records.extend(report['records']); diagnostics.extend(report['parameter_diagnostics'])
        for record in report['records']: diagnostics.extend(record['derived_diagnostics'])
    result=aggregate_records(records)
    result.update(accepted_main_analysis=True,prior_bound_sensitivity_completed=False,
        disclaimer='Main U=4 evidence only; limited U=2/8 representative sensitivity is a separate gate.',
        max_rhat=max(d['rhat'] for d in diagnostics),min_bulk_ess=min(d['ess_bulk'] for d in diagnostics),
        min_tail_ess=min(d['ess_tail'] for d in diagnostics),archive_audits=audits,
        input_summary_sha256=digest(root/'summary.json'),plan_sha256=digest(root/'plan.json'),
        forecast_sha256=summary['forecasts_sha256'],
        source_sha256={p:digest(p) for p in ['src/bayesian_finite_analysis.py','src/scores.py']})
    if result['input_summary_sha256']!=digest(root/'summary.json'):
        raise ValueError('Experiment changed during analysis')
    out=root/'analysis'; destination=out/'summary.json'
    if out.exists() and any(out.iterdir()):
        previous=json.loads(destination.read_text())
        if previous!=result: raise ValueError('Refusing to overwrite changed scientific analysis')
        return previous
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; maturities=[]
    for h,horizon in result['horizons'].items():
        for metric,values in horizon['main']['metrics'].items():
            rows.append(dict(horizon=int(h),metric=metric,**{k:values[k] for k in ['M4','M5','M5_minus_M4']},
                             ci95_lower=values['ci95'][0],ci95_upper=values['ci95'][1],bootstrap_se=values['bootstrap_se']))
        for item in horizon['per_maturity']:
            for model in ['M4','M5']:
                maturities.append(dict(horizon=int(h),maturity_years=item['maturity_years'],model=model,**item[model]))
    for filename,table in [('paired_summary.csv',rows),('per_maturity.csv',maturities)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    partial=out/'summary.json.partial'
    partial.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8'); partial.replace(destination)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); parser.add_argument('--gate')
    args=parser.parse_args(); result=analyze_experiment(args.output,args.gate)
    print('Accepted main analysis:',result['n_origins'],'origins; bound sensitivity remains separate')
