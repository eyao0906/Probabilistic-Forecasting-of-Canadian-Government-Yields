"""Read-back audit and one exact-seed computational replay; no scientific acceptance."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from src import bayesian_revised as br
from src.bayesian_stage_a import make_design, diagnose

out=Path(__file__).resolve().parent
root=out.parents[1]
started=time.perf_counter()
protected=json.loads((out/'protected_before.json').read_text())
changed=[p for p,digest in protected.items()
         if not (root/p).is_file() or hashlib.sha256((root/p).read_bytes()).hexdigest()!=digest]
if changed:
    raise RuntimeError(f'Protected artifacts changed: {changed}')

reconstructed=[]
for label in ['repair1_initial','repair2_initial','synthetic']:
    folder=out/label
    saved=json.loads((folder/'summary.json').read_text())
    rows=[]
    if label=='synthetic':
        with np.load(folder/'draws.npz',allow_pickle=False) as z:
            rows=diagnose(z['draws'],z['names'].tolist())
    else:
        groups=sorted({(r['model'],r['factor']) for r in saved['diagnostics']})
        for model,factor in groups:
            values=[]
            for chain in range(4):
                with np.load(folder/f'{model}_{factor}_chain{chain}.npz',allow_pickle=False) as z:
                    values.append(z['draws'].copy())
                    names=json.loads(str(z['report']))['names']
            rows.extend({'model':model,'factor':factor,**r}
                        for r in diagnose(np.stack(values),names))
    key=lambda r:(r.get('model','M5'),r.get('factor','synthetic'),r['parameter'])
    expected=sorted(saved['diagnostics'],key=key)
    actual=sorted(rows,key=key)
    if len(actual)!=len(expected):
        raise RuntimeError('Diagnostic row count mismatch')
    for a,b in zip(actual,expected):
        if key(a)!=key(b) or a['passes']!=b['passes']:
            raise RuntimeError('Diagnostic identity or pass flag mismatch')
        for name in ['rhat','ess_bulk','ess_tail']:
            if a[name] is None or b[name] is None:
                if a[name]!=b[name]: raise RuntimeError('Nonfinite diagnostic mismatch')
            elif not np.isclose(a[name],b[name],rtol=1e-12,atol=1e-12):
                raise RuntimeError('Diagnostic value mismatch')
    reconstructed.append({'trial':label,'diagnostic_rows':len(rows),'matched':True})

# Same seed is a computational reproduction check, NOT an independent-run stability test.
d=make_design(); window=d['windows'][0]
path=out/'repair2_initial/M5_slope_chain0.npz'
with np.load(path,allow_pickle=False) as z:
    expected=z['draws'].copy(); config=json.loads(str(z['report']))
result=br.sample_chain(d['factors'][d['first_index']:window['index']+1,1],d['priors'][1],
                       'M5',np.random.default_rng(config['seed']),3000,6000,kernel='collapsed')
actual=result['draws']
matched=bool(np.array_equal(expected,actual))
report={'protected_files':len(protected),'protected_artifacts_preserved':not changed,
        'reconstructed_diagnostics':reconstructed,'exact_seed_chain_replayed':'M5_slope_chain0',
        'seed':config['seed'],'bitwise_draw_match':matched,
        'maximum_absolute_draw_difference':float(np.max(np.abs(actual-expected))),
        'elapsed_s':time.perf_counter()-started,
        'executed_trial_source_sha256':json.loads((out/'repair2_initial/configuration.json').read_text())['sampler_sha256'],
        'current_source_sha256':hashlib.sha256(Path(br.__file__).read_bytes()).hexdigest(),
        'source_difference_note':'After trial execution only run_pilot acceptance reporting was changed to fail the predictive-moment gate; sample_chain was unchanged.',
        'independent_run_predictive_stability_test':False,'stage_a_passed':False}
(out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
if not matched:
    raise SystemExit(1)
