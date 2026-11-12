"""Bounded Stage A diagnostics. Never runs baseline models or writes the paper."""
from __future__ import annotations
from pathlib import Path
from datetime import date
import hashlib
import json
import numpy as np
from src import bayesian_revised as br
from src.experiment import read_panel, select_origins
from src.models import ns_loadings

ROOT=Path(__file__).resolve().parents[1]
FACTORS=['level','slope','curvature']


def diagnose(values,names):
    import arviz as az
    values=np.asarray(values)
    if values.ndim!=3 or values.shape[2]!=len(names):
        raise ValueError('Expected chain/draw/parameter array')
    rows=[]
    for i,name in enumerate(names):
        v=values[:,:,i]
        rh=bulk=tail=None
        if np.isfinite(v).all() and v.shape[0]>=2 and v.shape[1]>=8:
            rh=float(az.rhat(v,method='rank'))
            bulk=float(az.ess(v,method='bulk'))
            tail=float(az.ess(v,method='tail',prob=(.05,.95)))
        finite=all(t is not None and np.isfinite(t) for t in [rh,bulk,tail])
        rows.append({'parameter':name,'rhat':rh if finite else None,
                     'ess_bulk':bulk if finite else None,'ess_tail':tail if finite else None,
                     'passes':bool(finite and v.shape[0]>=4 and rh<=1.01 and bulk>=400 and tail>=400)})
    return rows


def make_design():
    path=ROOT/'data/processed/canada_weekly.csv'
    dates,y=read_panel(path)
    initial=[i for i,d in enumerate(dates) if date(2000,1,1)<=d<=date(2015,12,31)]
    meta=json.loads((ROOT/'results/bayesian_revised_pilot/extended/metadata.json').read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=meta['input_sha256']:
        raise ValueError('Panel changed since saved fixed-lambda pilot')
    lam=meta['lambda_per_year']
    load=ns_loadings(np.array([.25,.5,1,2,5,7,10,30]),lam)
    factors=np.linalg.lstsq(load,y.T,rcond=None)[0].T
    priors=br.freeze_priors(factors[initial])
    origins=sorted({t for t,h,target in select_origins(dates,date(2016,1,1),date(2025,12,31),[1,4],4)})
    chosen=[origins[0],min(origins,key=lambda t:abs((dates[t]-date(2020,3,20)).days)),origins[-1]]
    first=initial[0]
    windows=[{'name':name,'index':t,'date':dates[t].isoformat(),'n_training':t-first+1,
              'targets':{str(h):dates[t+h].isoformat() for h in [1,4]}}
             for name,t in zip(['initial','stress','late'],chosen)]
    return {'dates':dates,'y':y,'factors':factors,'priors':priors,'load':load,
            'initial_n':len(initial),'initial_last_date':dates[initial[-1]].isoformat(),
            'first_index':first,'windows':windows,'lambda':lam,'input_sha256':digest}


def run_window(out,window,kernel,warmup=3000,draws=6000,models=None,seed=9102701):
    """Resume only identical per-chain checkpoints; never authorizes rolling work."""
    import time
    from dataclasses import asdict
    from src.experiment import write_csv
    d=make_design(); w=next(w for w in d['windows'] if w['name']==window)
    models=['M4','M5'] if models is None else list(models)
    if not models or len(set(models))!=len(models) or any(m not in ['M4','M5'] for m in models):
        raise ValueError('Invalid model subset')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    source=Path(br.__file__).read_bytes()
    config={'window':w,'kernel':kernel,'warmup':warmup,'draws':draws,'chains':4,
            'models':models,'seed':seed,'input_sha256':d['input_sha256'],
            'sampler_sha256':hashlib.sha256(source).hexdigest(),
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'priors':[asdict(p) for p in d['priors']],'lambda':d['lambda']}
    config_path=out/'configuration.json'
    if config_path.exists():
        if json.loads(config_path.read_text())!=json.loads(json.dumps(config)):
            raise ValueError('Cannot resume different configuration or source')
    else:
        with config_path.open('x',encoding='utf-8') as f: json.dump(config,f,indent=2)
        (out/'executed_sampler.py.txt').write_bytes(source)
        (out/'executed_runner.py.txt').write_bytes(Path(__file__).read_bytes())
    diagnostics=[]; timings=[]; start=time.perf_counter()
    for m in models:
        for j,name in enumerate(FACTORS):
            values=[]
            for chain in range(4):
                path=out/f'{m}_{name}_chain{chain}.npz'
                chain_seed=seed+100000*(m=='M5')+1000*j+chain
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:
                        result=json.loads(str(z['report']))
                        value=z['draws'].copy()
                    if result['seed']!=chain_seed or value.shape[0]!=draws:
                        raise ValueError('Invalid checkpoint')
                else:
                    result=br.sample_chain(d['factors'][d['first_index']:w['index']+1,j],
                                           d['priors'][j],m,np.random.default_rng(chain_seed),
                                           warmup,draws,kernel=kernel)
                    value=result.pop('draws'); result['seed']=chain_seed
                    with path.open('xb') as f:
                        np.savez_compressed(f,draws=value,report=json.dumps(result))
                values.append(value)
                timings.append({'model':m,'factor':name,'chain':chain,**result})
            rows=diagnose(np.stack(values),result['names'])
            diagnostics.extend({'model':m,'factor':name,**row} for row in rows)
            print(window,m,name,'gate',all(row['passes'] for row in rows),flush=True)
    report={'configuration':config,'diagnostics':diagnostics,'chain_reports':timings,
            'parameter_gate_passed':bool(diagnostics and all(r['passes'] for r in diagnostics)),
            'accepted_for_rolling':False,'status':'single-window parameter diagnostics only; predictive stability and other windows are separate gates',
            'elapsed_s_this_invocation':time.perf_counter()-start}
    write_csv(out/'diagnostics.csv',diagnostics)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--window',choices=['initial','stress','late'],required=True)
    parser.add_argument('--kernel',choices=['interweave','centered','collapsed'],required=True)
    parser.add_argument('--warmup',type=int,default=3000)
    parser.add_argument('--draws',type=int,default=6000)
    parser.add_argument('--models',nargs='+',choices=['M4','M5'],default=['M4','M5'])
    parser.add_argument('--seed',type=int,default=9102701)
    a=parser.parse_args()
    if a.warmup<0 or a.draws<8: parser.error('Need nonnegative warmup and >=8 draws')
    report=run_window(a.output,a.window,a.kernel,a.warmup,a.draws,a.models,a.seed)
    print(json.dumps({k:report[k] for k in ['parameter_gate_passed','accepted_for_rolling','elapsed_s_this_invocation']}))
    return 0 if report['parameter_gate_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())

