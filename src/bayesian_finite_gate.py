"""Fail-closed independent-seed comparison for accepted representative forecasts."""
from __future__ import annotations
from .bayesian_finite_validation import prediction_difference


def compare_run_reports(first,second):
    for key in ['window','input_sha256','lambda','priors','chains','draws','warmup','models']:
        if first['configuration'][key]!=second['configuration'][key]:
            raise ValueError(f'Unmatched configuration: {key}')
    seeds1,seeds2=first['chain_seeds'],second['chain_seeds']
    if len(seeds1)!=24 or len(seeds2)!=24 or len(set(seeds1))!=24 or len(set(seeds2))!=24 or set(seeds1)&set(seeds2):
        raise ValueError('Chains must have independent seeds within/across runs')
    expected={(m,h) for m in ['M4','M5'] for h in [1,4]}
    for r in [first,second]:
        keys=[(x['model'],x['horizon']) for x in r['records']]
        if len(keys)!=4 or set(keys)!=expected: raise ValueError('Require complete matched forecast sets')
    a={(r['model'],r['horizon']):r for r in first['records']}
    b={(r['model'],r['horizon']):r for r in second['records']}
    comparisons=[prediction_difference(a[key],b[key]) for key in sorted(expected)]
    return {'window':first['configuration']['window'],'comparisons':comparisons,
            'passes':bool(first['passes'] and second['passes'] and all(r['passes'] for r in comparisons)),
            'seed_sets_are_disjoint':True}


def write_gate(root):
    """Aggregate existing accepted evidence only; never launch sampling."""
    import json
    from pathlib import Path
    from .bayesian_finite_validation import validate_predictions
    from .bayesian_finite_runner import digest
    root=Path(root)
    pairs=[('initial','main_initial_seed1','fix1_initial_seed10102701'),
           ('stress','fix1_stress_seed9102701','fix1_stress_seed10102701'),
           ('late','fix1_late_seed9102701','fix1_late_seed10102701')]
    comparisons=[]; diagnostics=[]; evidence={}
    for window,a,b in pairs:
        ra,rb=validate_predictions(root/a),validate_predictions(root/b)
        comparison=compare_run_reports(ra,rb)
        if comparison['window']['name']!=window: raise ValueError('Wrong representative window')
        comparisons.append(comparison)
        for folder in [root/a,root/b]:
            summary=json.loads((folder/'summary.json').read_text())
            diagnostics.extend(summary['diagnostics'])
            for path in [folder/'configuration.json',folder/'summary.json',
                         folder/'predictive_validation'/'summary.json']:
                evidence[str(path)]=digest(path)
    synthetic_path=root/'fix1_synthetic'/'summary.json'
    synthetic=json.loads(synthetic_path.read_text()); evidence[str(synthetic_path)]=digest(synthetic_path)
    synthetic_pass=bool(synthetic['parameter_gate_passed'] and synthetic['moment_audit']['stage_a_mathematical_gate_passed'])
    diagnostics.extend(synthetic['diagnostics'])
    protected=json.loads((root/'protected_before.json').read_text())
    changed=[path for path,sha in protected.items() if not Path(path).exists() or digest(path)!=sha]
    files=['src/bayesian_finite.py','src/bayesian_revised.py','src/bayesian_finite_runner.py',
           'src/bayesian_finite_validation.py','src/bayesian_finite_gate.py','src/bayesian_stage_a.py',
           'src/scores.py','src/validated_baselines.py']
    report={'accepted':bool(synthetic_pass and all(x['passes'] for x in comparisons) and not changed),
            'representative_pairs':comparisons,'synthetic_accepted':synthetic_pass,
            'max_rhat':max(x['rhat'] for x in diagnostics),
            'min_bulk_ess':min(x['ess_bulk'] for x in diagnostics),
            'min_tail_ess':min(x['ess_tail'] for x in diagnostics),
            'source_sha256':{path:digest(path) for path in files},'evidence_sha256':evidence,
            'protected_checked':len(protected),'protected_changed':changed,
            'main_v_upper':4.,'chain_settings':{'chains':4,'warmup':3000,'draws':6000},
            'diagnostic_repairs_used':1,'diagnostic_repairs_maximum':2}
    (root/'phase2_gate.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(); parser.add_argument('--root',required=True)
    args=parser.parse_args(); report=write_gate(args.root)
    print(json.dumps({key:report[key] for key in ['accepted','max_rhat','min_bulk_ess','min_tail_ess','protected_checked','protected_changed']}))
    raise SystemExit(0 if report['accepted'] else 1)
