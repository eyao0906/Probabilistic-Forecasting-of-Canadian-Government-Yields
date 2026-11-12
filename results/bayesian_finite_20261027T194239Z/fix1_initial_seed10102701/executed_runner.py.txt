"""Checkpointed finite-prior validation; never authorizes production by itself."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import time
import ast
import numpy as np
from src import bayesian_finite as bf
from src import bayesian_revised as br
from src.bayesian_stage_a import ROOT, FACTORS, make_design, diagnose
from src.experiment import write_csv


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _m4_reuse(d,w,warmup,draws,seed):
    old=ROOT/'results/bayesian_stage_a_20261027T184429Z/repair2_initial'
    if w['name']!='initial' or not (old/'configuration.json').is_file(): return None
    cfg=json.loads((old/'configuration.json').read_text())
    expected={'window':w,'warmup':warmup,'draws':draws,'chains':4,'seed':seed,
              'input_sha256':d['input_sha256'],'lambda':d['lambda'],
              'priors':json.loads(json.dumps([asdict(p) for p in d['priors']]))}
    if any(cfg.get(k)!=v for k,v in expected.items()): return None
    previous=ast.parse((old/'executed_sampler.py.txt').read_text())
    current=ast.parse(Path(br.__file__).read_text())
    def functions(tree):
        return {n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='run_pilot'}
    if functions(previous)!=functions(current): return None
    return old


def run_window(out,window,warmup=3000,draws=6000,models=None,seed=9102701,v_upper=4.):
    d=make_design(); w=next(w for w in d['windows'] if w['name']==window)
    models=['M4','M5'] if models is None else list(models)
    if not models or len(set(models))!=len(models) or any(m not in ['M4','M5'] for m in models):
        raise ValueError('Invalid model subset')
    if warmup<0 or draws<8: raise ValueError('Invalid sampling length')
    priors=[bf.FinitePrior(p,v_upper=v_upper) for p in d['priors']]
    reuse=_m4_reuse(d,w,warmup,draws,seed) if 'M4' in models else None
    config={'window':w,'warmup':warmup,'draws':draws,'chains':4,'models':models,'seed':seed,
            'input_sha256':d['input_sha256'],'lambda':d['lambda'],'priors':[asdict(p) for p in priors],
            'sampler_sha256':digest(bf.__file__),'base_sampler_sha256':digest(br.__file__),
            'runner_sha256':digest(__file__),'design_sha256':digest(ROOT/'src/bayesian_stage_a.py'),
            'reused_m4':None if reuse is None else {p.name:digest(p) for p in sorted(reuse.glob('M4_*chain*.npz'))}}
    config=json.loads(json.dumps(config)); signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    out=Path(out); out.mkdir(parents=True,exist_ok=True); cfg_path=out/'configuration.json'
    if cfg_path.exists():
        if json.loads(cfg_path.read_text())!=config: raise ValueError('Cannot resume different configuration or source')
    else:
        with cfg_path.open('x',encoding='utf-8') as f: json.dump(config,f,indent=2)
        for name,path in [('executed_sampler.py.txt',bf.__file__),('executed_base_sampler.py.txt',br.__file__),('executed_runner.py.txt',__file__)]:
            (out/name).write_bytes(Path(path).read_bytes())
    diagnostics=[]; reports=[]; start=time.perf_counter()
    for model in models:
        for j,name in enumerate(FACTORS):
            values=[]
            for chain in range(4):
                path=out/f'{model}_{name}_chain{chain}.npz'
                chain_seed=seed+100000*(model=='M5')+1000*j+chain
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:
                        report=json.loads(str(z['report'])); value=z['draws'].copy()
                    if report.get('signature')!=signature or report.get('seed')!=chain_seed:
                        raise ValueError('Invalid checkpoint signature or seed')
                else:
                    if model=='M4' and reuse is not None:
                        src=reuse/path.name
                        with np.load(src,allow_pickle=False) as z:
                            report=json.loads(str(z['report'])); value=z['draws'].copy()
                        if report['seed']!=chain_seed: raise ValueError('M4 reuse seed mismatch')
                        report['reused_from']=str(src); report['reused_sha256']=digest(src)
                    else:
                        report=bf.sample_chain(d['factors'][d['first_index']:w['index']+1,j],priors[j],model,
                                               np.random.default_rng(chain_seed),warmup,draws)
                        value=report.pop('draws')
                    report['seed']=chain_seed; report['signature']=signature
                    temporary=path.with_suffix('.tmp.npz')
                    with temporary.open('xb') as f: np.savez_compressed(f,draws=value,report=json.dumps(report,allow_nan=False))
                    temporary.replace(path)
                if value.shape!=(draws,len(report['names'])) or not np.isfinite(value).all():
                    raise ValueError('Invalid checkpoint draw array')
                values.append(value); reports.append({'model':model,'factor':name,'chain':chain,**report})
            rows=diagnose(np.stack(values),report['names'])
            diagnostics.extend({'model':model,'factor':name,**r} for r in rows)
            print(window,model,name,'gate',all(r['passes'] for r in rows),flush=True)
    audits=[bf.moment_audit(p,w['n_training']-1) for p in priors]
    result={'configuration':config,'diagnostics':diagnostics,'chain_reports':reports,'moment_audits':audits,
            'parameter_gate_passed':bool(diagnostics and all(r['passes'] for r in diagnostics)),
            'moment_gate_passed':all(a['stage_a_mathematical_gate_passed'] for a in audits),
            'accepted_for_rolling':False,'status':'single window only; all windows and independent predictive stability still required',
            'elapsed_s_this_invocation':time.perf_counter()-start}
    write_csv(out/'diagnostics.csv',diagnostics)
    temporary=out/'summary.tmp.json'; temporary.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(out/'summary.json')
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True); parser.add_argument('--window',required=True,choices=['initial','stress','late'])
    parser.add_argument('--warmup',type=int,default=3000); parser.add_argument('--draws',type=int,default=6000)
    parser.add_argument('--seed',type=int,default=9102701); parser.add_argument('--v-upper',type=float,default=4.)
    parser.add_argument('--models',nargs='+',choices=['M4','M5'],default=['M4','M5'])
    a=parser.parse_args(); result=run_window(a.output,a.window,a.warmup,a.draws,a.models,a.seed,a.v_upper)
    print(json.dumps({k:result[k] for k in ['parameter_gate_passed','moment_gate_passed','accepted_for_rolling','elapsed_s_this_invocation']}))
    return 0 if result['parameter_gate_passed'] and result['moment_gate_passed'] else 1


if __name__=='__main__': raise SystemExit(main())
