"""Sampling-free planning, private U4 replay, and synthetic integrity tests."""
import importlib
import importlib.util
import unittest
from unittest.mock import patch


class SamplingFreeTestCase(unittest.TestCase):
    def setUp(self):
        # Fail closed even if a regression reaches a real parameter-sampling path.
        for target in ['src.bayesian_finite_runner.run_window',
                       'src.bayesian_finite.sample_chain',
                       'src.bayesian_revised.sample_chain',
                       'src.bayesian_stage_a.run_window']:
            guard=patch(target,side_effect=AssertionError('Parameter sampling forbidden in sensitivity tests'))
            guard.start(); self.addCleanup(guard.stop)


class SensitivityPlanTests(SamplingFreeTestCase):
    def test_six_prespecified_cases_reuse_only_representative_origins(self):
        self.assertIsNotNone(importlib.util.find_spec('src.bayesian_finite_sensitivity'),'Missing limited-bound sensitivity planner')
        module=importlib.import_module('src.bayesian_finite_sensitivity')
        origins=[{'index':i,'date':f'fixture-{i:03d}','chain_source':f'cached-{i}' if i in [0,50,129] else None} for i in range(130)]
        result=module.sensitivity_cases({'origins':origins,'warmup':3000,'draws':6000,'chains':4,'scenario_count':2048})
        self.assertEqual(len(result),6)
        self.assertEqual({r['v_upper'] for r in result},{2.,8.})
        self.assertEqual({r['index'] for r in result},{0,50,129})
        self.assertEqual(len({r['seed'] for r in result}),6)
        self.assertTrue(all(r['models']==['M5'] and r['reuse_m4_predictive'] for r in result))
        self.assertTrue(all(r['warmup']==3000 and r['draws']==6000 and r['chains']==4 and r['scenario_count']==2048 for r in result))
        origins[50]['chain_source']=None
        with self.assertRaisesRegex(ValueError,'three representative'):
            module.sensitivity_cases({'origins':origins})


class SensitivityGateTests(SamplingFreeTestCase):
    def test_entrypoint_cannot_start_before_complete_main(self):
        import json,tempfile
        from pathlib import Path
        from src import bayesian_finite_sensitivity as module
        self.assertTrue(hasattr(module,'run_limited_sensitivity'),'Missing gated sensitivity entrypoint')
        with tempfile.TemporaryDirectory() as temporary:
            main=Path(temporary)/'main'; main.mkdir()
            (main/'summary.json').write_text(json.dumps({'accepted':False,'planned_origins':130,'completed_origins':19,'failed_origins':[]}))
            with self.assertRaisesRegex(ValueError,'130-origin'):
                module.run_limited_sensitivity(main,workers=2)
            self.assertEqual({p.relative_to(temporary).as_posix() for p in Path(temporary).rglob('*')},{'main','main/summary.json'})

    def test_incomplete_main_refuses_sensitivity_before_outputs(self):
        import json
        import tempfile
        from pathlib import Path
        from src import bayesian_finite_sensitivity as module
        self.assertTrue(hasattr(module,'require_sensitivity_main'),'Missing sensitivity main-experiment gate')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'summary.json').write_text(json.dumps({'accepted':False,'planned_origins':130,'completed_origins':15,'failed_origins':[]}))
            before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.assertRaisesRegex(ValueError,'130-origin'):
                module.require_sensitivity_main(root)
            self.assertEqual(before,{p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            self.assertFalse((root/'analysis').exists())
            self.assertFalse((root.parent/'bound_sensitivity').exists())


class CachedSourcePredictionTests(SamplingFreeTestCase):
    def test_m5_prediction_replays_cached_representative_without_sampling(self):
        import json
        from pathlib import Path
        import numpy as np
        from src import bayesian_finite_sensitivity as module
        self.assertTrue(hasattr(module,'source_predictions'),'Missing read-only M5 sensitivity predictions')
        root=Path(__file__).resolve().parents[1]/'results/bayesian_finite_20261027T194239Z'
        source=root/'fix1_initial_seed10102701'
        cached=json.loads((source/'predictive_validation/summary.json').read_text())
        seeds={r['horizon']:r['scenario_seed'] for r in cached['records'] if r['model']=='M5'}
        before={p:p.stat().st_mtime_ns for p in source.rglob('*') if p.is_file()}
        records,arrays=module.source_predictions(source,4.,scenario_seeds=seeds)
        self.assertEqual(len(records),2)
        for row in records:
            expected=next(r for r in cached['records'] if r['model']=='M5' and r['horizon']==row['horizon'])
            for key in ['mean','covariance','dss','energy_score']:
                np.testing.assert_allclose(row[key],expected[key],rtol=1e-12,atol=1e-12)
            self.assertEqual(arrays[f"M5_h{row['horizon']}_scenarios"].shape,(2048,8))
        self.assertEqual(before,{p:p.stat().st_mtime_ns for p in source.rglob('*') if p.is_file()})
        with self.assertRaisesRegex(ValueError,'prior'):
            module.source_predictions(source,2.)


class SensitivityAssemblyTests(SamplingFreeTestCase):
    def test_cached_u4_fixture_assembles_resumes_and_rejects_tampering(self):
        import json,tempfile
        from pathlib import Path
        import numpy as np
        from src import bayesian_finite_sensitivity as module
        self.assertTrue(hasattr(module,'_assemble_case'),'Missing sensitivity case archive assembler')
        root=Path(__file__).resolve().parents[1]/'results/bayesian_finite_20261027T194239Z'
        source=root/'fix1_initial_seed10102701'; main=root/'rolling_main/origins/2016-01-08'
        cfg=json.loads((source/'configuration.json').read_text())
        case={'index':cfg['window']['index'],'date':cfg['window']['date'],'v_upper':4.,'seed':cfg['seed'],'name':'fixture_only_U4'}
        before={p:p.stat().st_mtime_ns for folder in [source,main] for p in folder.rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'fixture'
            report=module._assemble_case(out,case,source,main)
            self.assertTrue(report['accepted'])
            self.assertEqual(report['archive_audit']['record_count'],4)
            for values in report['comparisons'].values():
                np.testing.assert_allclose(list(values['bound_minus_main_U4'].values()),0.,atol=1e-12)
            saved={p:p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}
            self.assertEqual(module._assemble_case(out,case,source,main),report)
            self.assertEqual(saved,{p:p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()})
            changed=json.loads((out/'forecast.json').read_text()); changed['records'][0]['dss']+=.1
            (out/'forecast.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'reconstruction'):
                module._assemble_case(out,case,source,main)
        self.assertEqual(before,{p:p.stat().st_mtime_ns for folder in [source,main] for p in folder.rglob('*') if p.is_file()})


class SensitivitySourceIntegrityTests(SamplingFreeTestCase):
    """Private U4 replay in scratch; accepted research sources are read-only."""
    def copied_source(self,root):
        import json,shutil
        from pathlib import Path
        accepted=Path(__file__).resolve().parents[1]/'results/bayesian_finite_20261027T194239Z'
        original=accepted/'fix1_initial_seed10102701'
        source=root/'source'; source.mkdir()
        names=['configuration.json','summary.json','executed_sampler.py.txt',
               'executed_base_sampler.py.txt','executed_runner.py.txt']
        for path in list(original.glob('M5_*_chain*.npz'))+[original/name for name in names]:
            shutil.copy2(path,source/path.name)
        cfg=json.loads((source/'configuration.json').read_text())
        case={'index':cfg['window']['index'],'date':cfg['window']['date'],
              'v_upper':4.,'seed':cfg['seed'],'name':'private_U4_replay'}
        return source,case,accepted/'rolling_main/origins/2016-01-08'

    @staticmethod
    def mutate(path):
        import json
        if path.name=='configuration.json':
            cfg=json.loads(path.read_text()); cfg['warmup']+=1
            path.write_text(json.dumps(cfg))
        else:
            path.write_bytes(path.read_bytes()+b'\n# changed scratch snapshot\n')

    def test_completed_resume_rejects_configuration_or_executed_snapshot_changes(self):
        import tempfile
        from pathlib import Path
        from src import bayesian_finite_sensitivity as module
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source,case,main=self.copied_source(root); out=root/'case'
            report=module._assemble_case(out,case,source,main)
            for name in ['configuration.json','executed_sampler.py.txt',
                         'executed_base_sampler.py.txt','executed_runner.py.txt']:
                with self.subTest(name=name):
                    path=source/name; original=path.read_bytes()
                    try:
                        self.mutate(path)
                        with self.assertRaises(ValueError): module._assemble_case(out,case,source,main)
                    finally: path.write_bytes(original)
            with patch.object(module,'source_predictions',side_effect=AssertionError('Resume must not predict again')):
                self.assertEqual(module._assemble_case(out,case,source,main),report)

    def test_first_prediction_rejects_configuration_or_executed_snapshot_changes(self):
        import tempfile
        from pathlib import Path
        from src import bayesian_finite_sensitivity as module
        predict=module.source_predictions
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source,case,main=self.copied_source(root)
            for i,name in enumerate(['configuration.json','executed_sampler.py.txt',
                                     'executed_base_sampler.py.txt','executed_runner.py.txt']):
                with self.subTest(name=name):
                    path=source/name; original=path.read_bytes(); out=root/f'case-{i}'
                    def changed_after_prediction(*args,**kwargs):
                        result=predict(*args,**kwargs); self.mutate(path); return result
                    try:
                        with patch.object(module,'source_predictions',side_effect=changed_after_prediction):
                            with self.assertRaises(ValueError): module._assemble_case(out,case,source,main)
                        self.assertFalse((out/'forecast.json').exists())
                    finally: path.write_bytes(original)


class SyntheticDriverTestCase(SamplingFreeTestCase):
    """Synthetic unit-driver fixtures only: never scientific output or evidence."""
    def driver_fixture(self,root):
        import hashlib,inspect,json
        from contextlib import ExitStack
        from src import bayesian_finite_sensitivity as module
        from src.bayesian_finite_runner import digest
        main=root/'main'; main.mkdir(); (main/'analysis').mkdir()
        origins=[dict(index=i,date=f'synthetic-{i:03d}',chain_source='unit-only' if i in [0,50,129] else None)
                 for i in range(130)]
        plan=dict(origins=origins,warmup=3000,draws=6000,chains=4,scenario_count=2048)
        cases:list[dict]=module.sensitivity_cases(plan)
        (main/'plan.json').write_text(json.dumps(plan))
        (main/'summary.json').write_text(json.dumps({'synthetic_unit_driver':True}))
        (main/'analysis/summary.json').write_text(json.dumps({'synthetic_unit_driver':True}))
        frozen=dict(cases=cases,main_plan_sha256=digest(main/'plan.json'),
                    accepted_complete_main_required_before_sampling=True,
                    planner_definition_sha256=hashlib.sha256(inspect.getsource(module.sensitivity_cases).encode()).hexdigest())
        (root/'prior_bound_sensitivity_plan.json').write_text(json.dumps(frozen))
        stack=ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(module,'require_sensitivity_main',return_value=(plan,{},[])))
        stack.enter_context(patch('src.bayesian_finite_analysis.analyze_experiment'))
        stack.enter_context(patch('src.bayesian_stage_a.make_design',return_value={
            'windows':[dict(index=i,name=f'unit-{i}') for i in [0,50,129]]}))
        return module,main,cases

    @staticmethod
    def failure_pool(submissions,fail_on_submit=False):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        class UnitFailurePool:
            def __init__(self,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def submit(self,fn,main,case,*args):
                submissions.append(case['name'])
                error=BrokenProcessPool('synthetic worker exited; no numerical work ran')
                if fail_on_submit: raise error
                future=Future(); future.set_exception(error); return future
        return UnitFailurePool


class SensitivityProcessFailureTests(SyntheticDriverTestCase):
    def test_parent_observed_process_failure_is_durable_before_return_and_resume(self):
        import json,tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); module,main,cases=self.driver_fixture(root); submissions=[]
            with patch('concurrent.futures.ProcessPoolExecutor',self.failure_pool(submissions)):
                report=module.run_limited_sensitivity(main,workers=2)
                self.assertFalse(report['accepted_limited_sensitivity'])
                self.assertEqual(len(submissions),2,'No replacement submission after observed failure')
                for name in submissions:
                    failure=root/'bound_sensitivity'/name/'failure.json'
                    self.assertTrue(failure.is_file(),'Parent process failure must be durable per case')
                    saved=json.loads(failure.read_text())
                    self.assertEqual(saved['name'],name); self.assertFalse(saved['accepted'])
                    self.assertIn('synthetic worker exited',saved['error'])
                prior=list(submissions)
                resumed=module.run_limited_sensitivity(main,workers=2)
                self.assertEqual(submissions,prior,'A prior process failure must prevent automatic retry')
                self.assertEqual(resumed['failed_cases'],report['failed_cases'])

    def test_legacy_aggregate_only_process_failures_block_new_submissions(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); module,main,cases=self.driver_fixture(root); submissions=[]
            with patch('concurrent.futures.ProcessPoolExecutor',self.failure_pool(submissions)):
                prior=module.run_limited_sensitivity(main,workers=2)
                # Simulate evidence written by the older parent: aggregate only.
                for failure in (root/'bound_sensitivity').glob('*/failure.json'): failure.unlink()
                submissions.clear()
                resumed=module.run_limited_sensitivity(main,workers=2)
                self.assertEqual(submissions,[],'Legacy aggregate failures must stop resume before worker submission')
                self.assertEqual(resumed['failed_cases'],prior['failed_cases'])
                for result in prior['failed_cases']:
                    self.assertTrue((root/'bound_sensitivity'/result['name']/'failure.json').is_file())

    def test_parent_submission_failure_is_persisted_without_further_submission(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); module,main,cases=self.driver_fixture(root); submissions=[]
            observed=None; report=None
            with patch('concurrent.futures.ProcessPoolExecutor',self.failure_pool(submissions,fail_on_submit=True)):
                try: report=module.run_limited_sensitivity(main,workers=2)
                except Exception as exc: observed=exc
            self.assertIsNone(observed,'Parent submission failures must be persisted, not escape without evidence')
            self.assertIsNotNone(report)
            assert report is not None
            self.assertFalse(report['accepted_limited_sensitivity'])
            self.assertEqual(submissions,[cases[0]['name']])
            self.assertTrue((root/'bound_sensitivity'/cases[0]['name']/'failure.json').is_file())


class SensitivityFinalIntegrityTests(SyntheticDriverTestCase):
    def synthetic_archives(self,module,main,cases,after_job=None):
        """Replace only costly posterior generation; keep assembler/audits real."""
        import json
        from contextlib import ExitStack
        from pathlib import Path
        import numpy as np
        from src.bayesian_finite_runner import digest
        from src.bayesian_stage_a import FACTORS
        stack=ExitStack()
        dates=[f'synthetic-{i:03d}' for i in range(134)]
        design={'windows':[dict(index=i,name=f'unit-{i}') for i in [0,50,129]],
                'y':np.zeros((134,8)),'dates':dates}
        stack.enter_context(patch('src.bayesian_stage_a.make_design',return_value=design))
        def predictions(t,models):
            records=[]; arrays={}
            for model in models:
                for h in [1,4]:
                    row=dict(model=model,horizon=h,origin=dates[t],target=dates[t+h],
                             mean=[0.]*8,covariance=np.eye(8).tolist(),dss=0.,dss_reconstructed=0.,
                             energy_score=0.,squared_errors=[0.]*8,interval_lower_90=[0.]*8,
                             interval_upper_90=[0.]*8,coverage_90=[True]*8,passes=True,
                             score_reconstruction_passed=True,
                             derived_diagnostics=[dict(passes=True,rhat=1.,ess_bulk=1000.,ess_tail=1000.) for _ in range(6)])
                    tag=f'{model}_h{h}'; records.append(row)
                    arrays.update({tag+'_scenarios':np.zeros((2048,8)),tag+'_mean':np.zeros(8),
                                   tag+'_covariance':np.eye(8),tag+'_observed':np.zeros(8)})
            return records,arrays
        def synthetic_prediction(source,*args,**kwargs):
            cfg=json.loads((Path(source)/'configuration.json').read_text())
            return predictions(cfg['window']['index'],['M5'])
        stack.enter_context(patch.object(module,'source_predictions',side_effect=synthetic_prediction))
        real_job=module._sensitivity_job
        def job(main_output,case,*args):
            folder=main.parent/'bound_sensitivity'/case['name']; source=folder/'chains'
            if not source.exists():
                source.mkdir(parents=True)
                snapshots={'executed_sampler.py.txt':'sampler_sha256',
                           'executed_base_sampler.py.txt':'base_sampler_sha256','executed_runner.py.txt':'runner_sha256'}
                cfg:dict=dict(window={'index':case['index'],'date':case['date']},seed=case['seed'],
                         priors=[{'v_upper':case['v_upper']}],warmup=3000,draws=6000,chains=4,models=['M5'])
                for name,key in snapshots.items():
                    (source/name).write_bytes(b'# synthetic unit fixture, NOT executed research source\n')
                    cfg[key]=digest(source/name)
                (source/'configuration.json').write_text(json.dumps(cfg))
                (source/'summary.json').write_text(json.dumps(dict(configuration=cfg,parameter_gate_passed=True,
                                                                 moment_gate_passed=True,diagnostics=[])))
                for factor in FACTORS:
                    for k in range(4):
                        (source/f'M5_{factor}_chain{k}.npz').write_bytes(b'SYNTHETIC UNIT CHECKPOINT; NOT POSTERIOR DRAWS')
                origin=main/'origins'/case['date']; origin.mkdir(parents=True,exist_ok=True)
                records,arrays=predictions(case['index'],['M4','M5'])
                (origin/'forecast.json').write_text(json.dumps(dict(accepted=True,specification={'origin_index':case['index']},records=records)))
                np.savez_compressed(origin/'predictions.npz',**arrays)
            result=real_job(main_output,case,*args)
            if after_job is not None: after_job(case,result)
            return result
        stack.enter_context(patch.object(module,'_sensitivity_job',side_effect=job))
        return stack

    def test_later_case_cannot_refresh_an_earlier_accepted_forecast_digest(self):
        import json,tempfile
        from pathlib import Path
        from src.bayesian_finite_runner import digest
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); module,main,cases=self.driver_fixture(root); frozen={}
            first=cases[0]['name']; forecast=root/'bound_sensitivity'/first/'forecast.json'
            def mutate_after_job(case,result):
                if case['name']==first: frozen[first]=digest(forecast)
                elif case['name']==cases[1]['name']:
                    changed=json.loads(forecast.read_text()); changed['comparisons']['1']['M5_bound']['dss']=123.
                    forecast.write_text(json.dumps(changed))
            with self.synthetic_archives(module,main,cases,mutate_after_job):
                report=module.run_limited_sensitivity(main,workers=1)
            self.assertEqual(report['case_forecast_sha256'][first],frozen[first],
                             'save() must retain the digest frozen at individual completion')
            self.assertFalse(report['accepted_limited_sensitivity'])
            self.assertFalse((root/'bound_sensitivity/comparisons.csv').exists())

    def test_final_acceptance_rejects_between_case_archive_or_source_mutation(self):
        import tempfile
        from pathlib import Path
        from src.bayesian_stage_a import FACTORS
        names=['predictions.npz',f'chains/M5_{FACTORS[0]}_chain0.npz',
               'chains/configuration.json','chains/summary.json','chains/executed_sampler.py.txt',
               'chains/executed_base_sampler.py.txt','chains/executed_runner.py.txt']
        for name in names:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); module,main,cases=self.driver_fixture(root)
                def mutate_after_job(case,result):
                    if case['name']==cases[-1]['name']:
                        path=root/'bound_sensitivity'/cases[0]['name']/name
                        # Whitespace preserves JSON meaning; provenance must still fail.
                        path.write_bytes(path.read_bytes()+b'\n ')
                with self.synthetic_archives(module,main,cases,mutate_after_job):
                    report=module.run_limited_sensitivity(main,workers=1)
                self.assertFalse(report['accepted_limited_sensitivity'],f'Changed {name} must fail final revalidation')
                self.assertTrue(report['failed_cases'])
                self.assertFalse((root/'bound_sensitivity/comparisons.csv').exists())

    def test_completed_resume_missing_source_evidence_is_validation_only(self):
        import json,tempfile
        from pathlib import Path
        from src.bayesian_stage_a import FACTORS
        for missing_checkpoint in [False,True]:
            with self.subTest(missing_checkpoint=missing_checkpoint),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); module,main,cases=self.driver_fixture(root)
                with self.synthetic_archives(module,main,cases):
                    initial=module.run_limited_sensitivity(main,workers=1)
                    self.assertTrue(initial['accepted_limited_sensitivity'],'Synthetic six-case fixture must complete first')
                    self.assertEqual(initial['completed_cases'],6)
                    folder=root/'bound_sensitivity'/cases[0]['name']; source=folder/'chains'
                    missing=[source/'summary.json']
                    if missing_checkpoint: missing.append(source/f'M5_{FACTORS[0]}_chain0.npz')
                    for path in missing: path.unlink()
                    def protected_state():
                        paths=[p for p in main.rglob('*') if p.is_file()]
                        for case in cases:
                            case_folder=root/'bound_sensitivity'/case['name']
                            paths.extend([case_folder/'forecast.json',case_folder/'predictions.npz'])
                            paths.extend(p for p in (case_folder/'chains').rglob('*') if p.is_file())
                        return {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in paths}
                    before=protected_state(); saved_failure=None
                    with patch('src.bayesian_finite_runner.run_window',
                               side_effect=AssertionError('Completed resume must never run a parameter window')) as run_window, \
                         patch('concurrent.futures.ProcessPoolExecutor',
                               side_effect=AssertionError('Completed resume must not submit new jobs')) as pool, \
                         patch.object(module,'source_predictions',
                                      side_effect=AssertionError('Completed resume must not regenerate predictions')) as predict:
                        for attempt in range(2):
                            report=module.run_limited_sensitivity(main,workers=2)
                            run_window.assert_not_called()
                            pool.assert_not_called(); predict.assert_not_called()
                            self.assertFalse(report['accepted_limited_sensitivity'])
                            self.assertFalse(report['prior_bound_sensitivity_completed'])
                            self.assertEqual(report['completed_cases'],5)
                            self.assertEqual(report['case_forecast_sha256'],initial['case_forecast_sha256'])
                            self.assertEqual(len(report['failed_cases']),1)
                            failure=folder/'failure.json'
                            self.assertTrue(failure.is_file(),'Missing source evidence must produce a durable case failure')
                            evidence=json.loads(failure.read_text())
                            self.assertEqual(evidence,report['failed_cases'][0])
                            self.assertEqual(evidence['name'],cases[0]['name'])
                            self.assertFalse(evidence['accepted'])
                            self.assertIn('summary.json',evidence['error'])
                            state=(failure.read_bytes(),failure.stat().st_mtime_ns)
                            if attempt==0: saved_failure=state
                            else: self.assertEqual(state,saved_failure,'Resume must preserve the original durable failure')
                            self.assertEqual(protected_state(),before,'Resume must preserve source and completed archive bytes/mtimes')
                            for path in missing: self.assertFalse(path.exists(),'Missing source evidence must not be recreated')

    def test_all_six_archives_are_revalidated_before_comparisons_and_on_resume(self):
        import csv,json,tempfile
        from pathlib import Path
        from src.bayesian_finite_runner import digest
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); module,main,cases=self.driver_fixture(root)
            with self.synthetic_archives(module,main,cases):
                with patch.object(module,'_assemble_case',wraps=module._assemble_case) as assembler:
                    report=module.run_limited_sensitivity(main,workers=1)
                self.assertEqual(assembler.call_count,12,'Each case needs individual and final sampling-free validation')
                self.assertTrue(report['accepted_limited_sensitivity'])
                self.assertEqual(report['case_forecast_sha256'],
                                 {case['name']:digest(root/'bound_sensitivity'/case['name']/'forecast.json') for case in cases})
                with (root/'bound_sensitivity/comparisons.csv').open(newline='') as handle:
                    rows=list(csv.DictReader(handle))
                self.assertEqual(len(rows),48)
                for row in rows:
                    case=next(case for case in cases if case['date']==row['origin'] and case['v_upper']==float(row['v_upper']))
                    saved=json.loads((root/'bound_sensitivity'/case['name']/'forecast.json').read_text())
                    values=saved['comparisons'][row['horizon']]
                    for key,metrics in values.items(): self.assertEqual(float(row[key]),metrics[row['metric']])
                with patch.object(module,'source_predictions',side_effect=AssertionError('Completed resume cannot regenerate predictions')):
                    self.assertEqual(module.run_limited_sensitivity(main,workers=1),report)


if __name__=='__main__': unittest.main()
