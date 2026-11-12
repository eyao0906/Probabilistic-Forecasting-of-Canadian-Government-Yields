"""Inventory and independently replay the research without altering saved outputs.

Canonical CLI: python -m src.reproduce --help
Execution and mathematical reproducibility do not establish scientific acceptance.
"""
from __future__ import annotations
import ast
import hashlib
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'.git','.venv','__pycache__','reproduction','node_modules','site-packages'}
SUFFIXES={'.py','.r','.sh','.ps1','.ipynb','.qmd'}
ACQUISITION={'src/acquisition.py','src/source_catalog.py','src/source_data_check.py','src/verify_sources.py','src/verify_data_sources.py'}
LEGACY={'fetch_raw.py','prep_data.py','src/experiment.py','src/data_pipeline.py','src/finalize_paper.py','src/update_paper_results.py','src/generate_figures.py','src/summarize_results.py','src/config.py','src/data.py','src/evaluation.py'}


def inventory(root=ROOT):
    root=Path(root).resolve();entries=[]
    for parent,dirs,files in os.walk(root):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDE)
        for filename in sorted(files):
            path=Path(parent)/filename
            if path.suffix.lower() not in SUFFIXES and not path.name.endswith('.py.txt'):continue
            relative=path.relative_to(root).as_posix();raw=path.read_bytes()
            historical=('archive_before_validation' in path.parts or 'source_snapshot' in path.parts or 'archive_pre_resumption' in path.parts)
            if historical:role='historical_snapshot'
            elif relative.startswith('tests/'):role='test'
            elif relative in ACQUISITION or (relative.startswith('docs/sources/') and path.suffix=='.py'):role='source_acquisition'
            elif relative in LEGACY:role='historical_not_canonical'
            elif path.suffix.lower()=='.qmd':role='document_source'
            else:role='canonical_code'
            row={'path':relative,'role':role,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                 'syntax_ok':None,'verification':'document render' if path.suffix.lower()=='.qmd' else 'inventory only'}
            if path.suffix.lower()=='.py' or path.name.endswith('.py.txt'):
                try:
                    compile(ast.parse(raw,filename=relative),relative,'exec')
                    row.update(syntax_ok=True,verification='AST compilation; execution status recorded separately')
                except (SyntaxError,ValueError,UnicodeError) as error:
                    row.update(syntax_ok=False,error=str(error))
            entries.append(row)
    entries.sort(key=lambda r:r['path'])
    return {'schema_version':1,'passed':not any(x['syntax_ok'] is False for x in entries),
            'code_count':len(entries),'python_count':sum(x['path'].endswith(('.py','.py.txt')) for x in entries),
            'excluded_generated_directories':sorted(EXCLUDE),'files':entries,
            'scope_note':'All project source outside environments, Git internals and derivative reproduction copies. Historical code is preserved, syntax-checked, not claimed scientifically accepted.'}


def stage_workspace(root, destination):
    """Copy declared inputs/code only into a NEW confined reproduction workspace."""
    import json,shutil
    root=Path(root).resolve();destination=Path(destination).resolve()
    if not destination.is_relative_to(root/'reproduction') or destination==root/'reproduction':
        raise ValueError('Workspace must be a new child of project/reproduction')
    if destination.exists(): raise FileExistsError('Reproduction workspace already exists')
    inputs=[]
    for parent,dirs,files in os.walk(root):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDE)
        for name in sorted(files):
            path=Path(parent)/name;rel=path.relative_to(root).as_posix()
            code=path.suffix.lower() in SUFFIXES or path.name.endswith('.py.txt')
            raw=rel.startswith('data/raw/')
            source=rel.startswith('docs/sources/')
            historical=rel.startswith('results/archive_before_validation/')
            settings=rel.startswith('config/') or rel in ['references.bib','requirements-lock.txt','requirements.txt','requirements-validated.txt']
            document=path.suffix.lower()=='.csl' or (path.parent==root and path.name in ['MQF MRP Guidelines 2026.pdf','MRP_Proposal.pdf','MRP_Proposal_Revised.pdf'])
            if not (code or raw or source or historical or settings or document):continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('Refusing symlink/external input: '+rel)
            inputs.append((path,rel,'raw_data' if raw else 'source_evidence' if source else 'historical_evidence' if historical else 'source_or_configuration'))
    destination.mkdir(parents=True,exist_ok=False)
    manifest=[]
    for path,rel,role in inputs:
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        after=hashlib.sha256(target.read_bytes()).hexdigest()
        if before!=after: raise RuntimeError('Input copy checksum mismatch: '+rel)
        manifest.append({'path':rel,'role':role,'sha256':before,'bytes':target.stat().st_size})
    report={'copied_files':len(manifest),'destination':str(destination),'files':manifest,
            'note':'No existing predictive results, generated figures, rendered PDF, environment or old logs copied. Quarantined historical evidence is input, not an accepted forecast.'}
    (destination/'staged_inputs.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def compare_numeric_archive(original, replayed, rtol=1e-9, atol=1e-10):
    """Compare semantic arrays, not timestamp-dependent NPZ container bytes."""
    import numpy as np
    result={'original':str(original),'replayed':str(replayed),'passed':False,'rtol':rtol,'atol':atol,'arrays':{}}
    try:
        with np.load(original,allow_pickle=False) as a, np.load(replayed,allow_pickle=False) as b:
            if set(a.files)!=set(b.files):
                result['error']='Archive key sets differ';return result
            for key in a.files:
                x,y=a[key],b[key]
                row={'shape_equal':x.shape==y.shape,'passed':False}
                if x.shape==y.shape and x.size>0:
                    if x.dtype.kind in 'iufc' and y.dtype.kind in 'iufc':
                        finite=bool(np.isfinite(x).all() and np.isfinite(y).all())
                        row.update(finite=finite,max_absolute_difference=float(np.max(np.abs(x-y))) if x.size else 0.,
                                   passed=bool(finite and np.allclose(x,y,rtol=rtol,atol=atol)))
                    else: row['passed']=bool(np.array_equal(x,y))
                result['arrays'][key]=row
            result['passed']=bool(result['arrays']) and all(x['passed'] for x in result['arrays'].values())
    except (OSError,ValueError,KeyError) as error: result['error']=str(error)
    return result


def build_plan(workspace, python):
    """Explicit full-size replay; rejection diagnostics are outputs, not approval."""
    python=str(python)
    def module(name,*args):return [python,'-B','-m',name,*args]
    extended="from pathlib import Path; from src.bayesian_revised import run_pilot; import json; print(json.dumps(run_pilot(out=Path('results/bayesian_revised_pilot/extended'),warmup=3000,draws=6000,chains=4),indent=2))"
    rows=[
        ('acquisition',module('src.acquisition','--extract','--report','logs/acquisition_offline.json'),300),
        ('prepare',module('src.prepare_data'),120),
        ('baselines',module('src.validated_baselines'),600),
        ('availability',module('src.robustness'),600),
        ('mcs',module('src.mcs_baselines'),300),
        ('bayesian_default',module('src.bayesian_revised'),900),
        ('bayesian_extended',[python,'-B','-c',extended],1800),
        ('m3',module('src.state_space_revised','--phase','all'),1800),
        ('m3_verify',[python,'-B','results/state_space_revised/verify_outputs.py'],600),
        ('artifacts',module('src.build_artifacts'),300),
        ('supplements',module('src.revision_artifacts'),300),
        ('render',['quarto','render','paper/MRP_Draft.qmd','--to','pdf'],600),
        ('pdf_verify',module('src.verify_paper','--no-previews'),120),
        ('tests',module('unittest','discover','-s','tests','-v'),600),
    ]
    return [{'name':name,'command':command,'timeout_s':timeout} for name,command,timeout in rows]


def run_step(workspace, spec):
    import subprocess,time
    from datetime import datetime,timezone
    workspace=Path(workspace).resolve()
    logs=workspace/'logs';logs.mkdir(parents=True,exist_ok=True)
    target=logs/(spec['name']+'.log')
    started=time.perf_counter()
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUTF8='1',MPLBACKEND='Agg')
    env.pop('PYTHONPATH',None)
    if os.name=='nt': scratch=Path.home()/'AppData/Local/hermes/cache/scratch'
    else: scratch=Path.home()/'.hermes/cache/scratch'
    scratch.mkdir(parents=True,exist_ok=True);env['TMPDIR']=str(scratch)
    row={**spec,'started_utc':datetime.now(timezone.utc).isoformat(),'log':target.relative_to(workspace).as_posix()}
    with target.open('w',encoding='utf-8') as handle:
        try:
            result=subprocess.run(spec['command'],cwd=workspace,env=env,stdout=handle,
                                  stderr=subprocess.STDOUT,timeout=spec['timeout_s'],check=False)
            row['returncode']=result.returncode
        except (OSError,subprocess.TimeoutExpired) as error:
            row.update(returncode=124 if isinstance(error,subprocess.TimeoutExpired) else 127,error=str(error))
    row.update(passed=row['returncode']==0,elapsed_s=time.perf_counter()-started)
    print(spec['name'],'PASS' if row['passed'] else 'FAIL',f"{row['elapsed_s']:.2f}s",flush=True)
    return row


def run_pipeline(root, destination, python):
    """Run all declared stages in a fresh copy and save fail-closed evidence."""
    import json
    from datetime import datetime,timezone
    root=Path(root).resolve();destination=Path(destination).resolve();python=Path(python).resolve()
    def protected():
        files=[]
        for prefix in ['data/raw','docs/sources','results']:
            base=root/prefix
            if base.exists(): files.extend(p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        pdf=root/'paper/MRP_Draft.pdf'
        if pdf.exists():files.append(pdf)
        return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    before=protected()
    staged=stage_workspace(root,destination)
    plan=build_plan(destination,python)
    report={'schema_version':1,'started_utc':datetime.now(timezone.utc).isoformat(),
            'root':str(root),'workspace':str(destination),'python':str(python),'planned_steps':len(plan),
            'staged_input_count':staged['copied_files'],'steps':[],'numeric_comparisons':[],
            'scope_note':'Execution/numeric replay is not scientific acceptance. M3 and the M4/M5 pair remain unaccepted unless independently cleared.'}
    target=destination/'reproduction_report.json'
    for spec in plan:
        row=run_step(destination,spec);report['steps'].append(row)
        target.write_text(json.dumps(report,indent=2),encoding='utf-8')
        if not row['passed']:break
    report['execution_passed']=len(report['steps'])==len(plan) and all(x['passed'] for x in report['steps'])
    if report['execution_passed']:
        for path in ['results/validated_baselines/forecasts.npz','results/availability_robustness/forecasts.npz',
                     'results/state_space_revised/forecasts.npz','results/bayesian_revised_pilot/chains.npz',
                     'results/bayesian_revised_pilot/extended/chains.npz']:
            report['numeric_comparisons'].append({'path':path,**compare_numeric_archive(root/path,destination/path)})
        for path in ['data/processed/canada_weekly.csv']:
            a=root/path;b=destination/path
            ok=a.is_file() and b.is_file() and a.read_bytes()==b.read_bytes()
            report['numeric_comparisons'].append({'path':path,'comparison':'exact CSV bytes','passed':bool(ok)})
        for path in ['results/state_space_revised/metadata.json','results/bayesian_revised_pilot/metadata.json',
                     'results/bayesian_revised_pilot/extended/metadata.json']:
            value=json.loads((destination/path).read_text(encoding='utf-8'))
            report.setdefault('scientific_status',{})[path]={k:value[k] for k in
                ['accepted','accepted_for_rolling','status','max_rhat','min_ess_bulk','min_ess_tail'] if k in value}
    after=protected()
    changes=[p for p in sorted(before.keys()|after.keys()) if before.get(p)!=after.get(p)]
    report.update(original_artifacts_preserved=not changes,changed_original_artifacts=changes,
                  original_protected_file_count=len(before),completed_utc=datetime.now(timezone.utc).isoformat())
    report['numeric_reproduction_passed']=bool(report['numeric_comparisons']) and all(x['passed'] for x in report['numeric_comparisons'])
    report['passed']=bool(report['execution_passed'] and report['numeric_reproduction_passed'] and report['original_artifacts_preserved'])
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main(argv=None):
    import argparse,json,sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--inventory-only',action='store_true',help='Hash and syntax-check every project script, without executing research')
    parser.add_argument('--workspace',type=Path,help='Run the full pipeline in this NEW child of project/reproduction')
    parser.add_argument('--python',type=Path,default=Path(sys.executable),help='Interpreter with requirements-lock.txt installed')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    if args.workspace:
        if args.inventory_only:parser.error('--workspace and --inventory-only are mutually exclusive')
        report=run_pipeline(args.root,args.workspace,args.python)
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:report[k] for k in ['passed','execution_passed','numeric_reproduction_passed','original_artifacts_preserved','workspace']},indent=2))
        return 0 if report['passed'] else 1
    report=inventory(args.root)
    target=args.report or args.root/'logs/code_integrity.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'code_count':report['code_count'],'report':str(target)},indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())


