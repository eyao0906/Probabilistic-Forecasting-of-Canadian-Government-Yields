"""Frozen main-experiment plan and bounded checkpoint orchestration."""
from __future__ import annotations
from pathlib import Path
from datetime import date
import json
from .bayesian_finite_production import require_phase2,forecast_origin
from .bayesian_finite_runner import digest
from .bayesian_stage_a import make_design
from .experiment import select_origins


def make_plan(gate_path):
    gate_path=Path(gate_path); gate=require_phase2(gate_path); d=make_design()
    origins=sorted({i for i,h,target in select_origins(d['dates'],date(2016,1,1),date(2025,12,31),[1,4],4)})
    if len(origins)!=130: raise ValueError('Common origin count changed')
    folders={'initial':'fix1_initial_seed10102701','stress':'fix1_stress_seed9102701','late':'fix1_late_seed9102701'}
    reuse={pair['window']['index']:str((gate_path.parent/folders[pair['window']['name']]).resolve())
           for pair in gate['representative_pairs']}
    return {'gate_sha256':digest(gate_path),'input_sha256':d['input_sha256'],'lambda':d['lambda'],
            'maturities':[.25,.5,1,2,5,7,10,30],'horizons':[1,4],'models':['M4','M5'],
            'v_upper':4.,'warmup':3000,'draws':6000,'chains':4,'scenario_count':2048,
            'production_sha256':digest('src/bayesian_finite_production.py'),
            'origins':[{'ordinal':j,'index':t,'date':str(d['dates'][t]),'seed':9102701+1000000*j,
                        'chain_source':reuse.get(t),'status':'pending'} for j,t in enumerate(origins)]}


def _origin_job(gate_path,output,row):
    """CPU worker, not an agent; preserve failures rather than retry silently."""
    import contextlib,traceback
    out=Path(output)/'origins'/row['date']; logs=Path(output)/'logs'; logs.mkdir(exist_ok=True)
    try:
        with (logs/(row['date']+'.log')).open('a') as handle, contextlib.redirect_stdout(handle),contextlib.redirect_stderr(handle):
            report=forecast_origin(gate_path,out,row['index'],row['seed'],row['chain_source'])
        return {'date':row['date'],'accepted':bool(report['accepted'])}
    except Exception as exc:
        failure={'date':row['date'],'accepted':False,'error':str(exc),'traceback':traceback.format_exc()}
        out.mkdir(parents=True,exist_ok=True); (out/'failure.json').write_text(json.dumps(failure,indent=2))
        return failure


def run_experiment(gate_path,output,workers=4,representatives_only=False):
    """Bound inflight jobs; stop new work at first failure; resume exact successes."""
    import os
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    if not isinstance(workers,int) or not 1<=workers<=8: raise ValueError('Need one through eight CPU workers')
    plan=make_plan(gate_path); out=Path(output); path=out/'plan.json'
    if path.exists():
        if json.loads(path.read_text())!=plan: raise ValueError('Changed frozen experiment plan')
    else:
        out.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(plan,indent=2))
        (out/'executed_orchestrator.py.txt').write_bytes(Path(__file__).read_bytes())
    rows=plan['origins']; completed={}; failures={}
    def save():
        report={'accepted':len(completed)==130 and not failures,'planned_origins':130,
                'completed_origins':len(completed),'completed_dates':sorted(completed),
                'failed_origins':list(failures.values()),'not_completed_dates':[r['date'] for r in rows if r['date'] not in completed],
                'plan_sha256':digest(path),'workers':workers,'representatives_only':representatives_only,
                'forecasts_sha256':{r['date']:digest(out/'origins'/r['date']/'forecast.json') for r in rows if r['date'] in completed}}
        partial=out/'summary.partial.json'; partial.write_text(json.dumps(report,indent=2)); os.replace(partial,out/'summary.json')
        return report
    pending=[]
    for row in rows:
        origin=out/'origins'/row['date']
        if (origin/'forecast.json').exists():
            result=_origin_job(gate_path,out,row)
            (completed if result['accepted'] else failures)[row['date']]=result
        elif (origin/'failure.json').exists():
            failures[row['date']]=json.loads((origin/'failure.json').read_text())
        elif not representatives_only or row['chain_source'] is not None:
            pending.append(row)
    save()
    if failures: return save()
    # Cheap accepted representatives first; no chain resampling.
    for row in [r for r in pending if r['chain_source'] is not None]:
        result=_origin_job(gate_path,out,row)
        (completed if result['accepted'] else failures)[row['date']]=result
        print(json.dumps({'origin':row['date'],'accepted':result['accepted'],'completed':len(completed)}),flush=True)
        save()
        if failures: return save()
    pending=[r for r in pending if r['chain_source'] is None]
    if not pending: return save()
    if workers==1:
        for row in pending:
            result=_origin_job(gate_path,out,row)
            (completed if result['accepted'] else failures)[row['date']]=result
            print(json.dumps({'origin':row['date'],'accepted':result['accepted'],'completed':len(completed)}),flush=True)
            save()
            if failures: break
    else:
        iterator=iter(pending)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            inflight={}
            for _ in range(min(workers,len(pending))):
                row=next(iterator); inflight[pool.submit(_origin_job,str(gate_path),str(out),row)]=row
            while inflight:
                done,_=wait(inflight,return_when=FIRST_COMPLETED)
                for future in done:
                    row=inflight.pop(future)
                    try: result=future.result()
                    except Exception as exc: result={'date':row['date'],'accepted':False,'error':repr(exc)}
                    (completed if result['accepted'] else failures)[row['date']]=result
                    print(json.dumps({'origin':row['date'],'accepted':result['accepted'],'completed':len(completed)}),flush=True)
                    save()
                if not failures:
                    for _ in range(workers-len(inflight)):
                        row=next(iterator,None)
                        if row is None: break
                        inflight[pool.submit(_origin_job,str(gate_path),str(out),row)]=row
    return save()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--gate',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=4); parser.add_argument('--representatives-only',action='store_true')
    args=parser.parse_args()
    report=run_experiment(args.gate,args.output,args.workers,args.representatives_only)
    print(json.dumps({key:report[key] for key in ['accepted','planned_origins','completed_origins','failed_origins']}),flush=True)
    raise SystemExit(0 if report['accepted'] or (args.representatives_only and not report['failed_origins']) else 1)
