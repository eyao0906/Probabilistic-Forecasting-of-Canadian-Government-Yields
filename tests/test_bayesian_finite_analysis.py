"""Tests for accepted-experiment aggregation, never synthetic scientific evidence."""
import importlib
import importlib.util
import unittest
import numpy as np


class PairedAnalysisTests(unittest.TestCase):
    def test_paired_constant_fixture_has_exact_difference_and_interval(self):
        self.assertIsNotNone(importlib.util.find_spec('src.bayesian_finite_analysis'),
                             'Missing paired finite-experiment analysis')
        module=importlib.import_module('src.bayesian_finite_analysis')
        m4={'dss':np.full(12,1.),'energy_score':np.full(12,.1),
            'squared_errors':np.full((12,8),4.),'coverage_90':np.ones((12,8))}
        m5={'dss':np.full(12,2.),'energy_score':np.full(12,.2),
            'squared_errors':np.full((12,8),9.),'coverage_90':np.zeros((12,8))}
        result=module.paired_summary(m4,m5,reps=40,block_length=3,seed=271026)
        for metric,expected in [('dss',1.),('energy_score',.1),('rmse',1.),('coverage_90',-1.)]:
            row=result['metrics'][metric]
            self.assertAlmostEqual(row['M5_minus_M4'],expected)
            np.testing.assert_allclose(row['ci95'],[expected,expected],atol=1e-14)
        self.assertEqual(result['n_origins'],12)
        self.assertEqual(result['block_length_origins'],3)

    def test_defaults_preserve_original_stationary_bootstrap_protocol(self):
        import inspect
        module=importlib.import_module('src.bayesian_finite_analysis')
        signature=inspect.signature(module.paired_summary)
        self.assertEqual(signature.parameters['reps'].default,9999)
        self.assertEqual(signature.parameters['block_length'].default,8)
        self.assertEqual(signature.parameters['seed'].default,271026)
        arrays={'dss':np.arange(12.),'energy_score':np.arange(12.)/10,
                'squared_errors':np.arange(1.,97.).reshape(12,8),'coverage_90':np.ones((12,8))}
        result=module.paired_summary(arrays,arrays,reps=40)
        self.assertIn('stationary',result['method'])
        for row in result['metrics'].values():
            self.assertEqual(row['M5_minus_M4'],0.)
            self.assertEqual(row['ci95'],[0.,0.])


class CompleteArchiveTests(unittest.TestCase):
    def test_incomplete_experiment_is_rejected_before_analysis_output(self):
        import json,tempfile
        from pathlib import Path
        module=importlib.import_module('src.bayesian_finite_analysis')
        self.assertTrue(hasattr(module,'load_complete_experiment'),'Missing complete-experiment gate')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'summary.json').write_text(json.dumps({'accepted':False,'planned_origins':130,
                'completed_origins':3,'failed_origins':[]}))
            with self.assertRaisesRegex(ValueError,'130-origin'):
                module.load_complete_experiment(root)
            self.assertFalse((root/'analysis').exists())

    def test_incomplete_analysis_writes_no_results(self):
        import json,tempfile
        from pathlib import Path
        module=importlib.import_module('src.bayesian_finite_analysis')
        self.assertTrue(hasattr(module,'analyze_experiment'),'Missing gated analysis entry point')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'summary.json').write_text(json.dumps({'accepted':False,'planned_origins':130,'completed_origins':3,'failed_origins':[]}))
            with self.assertRaisesRegex(ValueError,'130-origin'):
                module.analyze_experiment(root)
            self.assertFalse((root/'analysis').exists())


class OriginReconstructionTests(unittest.TestCase):
    def test_existing_origin_loader_is_strictly_read_only(self):
        import json
        from pathlib import Path
        module=importlib.import_module('src.bayesian_finite_analysis')
        self.assertTrue(hasattr(module,'read_origin'),'Missing read-only origin loader')
        from src.bayesian_finite_experiment import make_plan
        root=Path(__file__).resolve().parents[1]/'results/bayesian_finite_20261027T194239Z'
        folder=root/'rolling_main/origins/2016-01-08'
        if not (folder/'forecast.json').exists(): self.skipTest('Optional real accepted archive absent')
        plan=make_plan(root/'phase2_gate.json'); row=plan['origins'][0]
        before={str(p):p.stat().st_mtime_ns for p in folder.iterdir() if p.is_file()}
        result=module.read_origin(folder,row,plan)
        self.assertTrue(result['accepted'])
        self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in folder.iterdir() if p.is_file()})
        with self.assertRaisesRegex(ValueError,'specification'):
            module.read_origin(folder,dict(row,seed=row['seed']+1),plan)

    def test_saved_origin_reconstructs_and_changed_mean_is_rejected(self):
        import copy,json
        from pathlib import Path
        module=importlib.import_module('src.bayesian_finite_analysis')
        self.assertTrue(hasattr(module,'audit_origin'),'Missing independent archive reconstruction')
        from src.bayesian_stage_a import make_design
        folder=Path(__file__).resolve().parents[1]/'results/bayesian_finite_20261027T194239Z/rolling_main/origins/2016-01-08'
        if not (folder/'forecast.json').exists(): self.skipTest('Optional real accepted archive absent')
        report=json.loads((folder/'forecast.json').read_text()); design=make_design()
        result=module.audit_origin(folder,report,design,energy_check=True)
        self.assertTrue(result['passes']); self.assertEqual(result['record_count'],4)
        altered=copy.deepcopy(report); altered['records'][0]['mean'][0]+=.1
        with self.assertRaisesRegex(ValueError,'reconstruction'):
            module.audit_origin(folder,altered,design)


class MatchedAggregationTests(unittest.TestCase):
    def test_full_fixture_and_missing_record_gate(self):
        module=importlib.import_module('src.bayesian_finite_analysis')
        self.assertTrue(hasattr(module,'aggregate_records'),'Missing matched aggregation')
        rows=[]
        for i in range(130):
            for h in [1,4]:
                for model,score,error,coverage in [('M4',1.,4.,True),('M5',2.,9.,False)]:
                    rows.append({'origin':f'fixture-{i:03d}','model':model,'horizon':h,'dss':score,
                                 'energy_score':score/10,'squared_errors':[error]*8,'coverage_90':[coverage]*8,
                                 'interval_lower_90':[0.]*8,'interval_upper_90':[score]*8})
        summary=module.aggregate_records(rows,reps=40)
        self.assertEqual(summary['n_origins'],130)
        for horizon in summary['horizons'].values():
            for key,want in [('dss',1.),('energy_score',.1),('rmse',1.),('coverage_90',-1.)]:
                self.assertAlmostEqual(horizon['main']['metrics'][key]['M5_minus_M4'],want)
            self.assertEqual(len(horizon['per_maturity']),8)
        with self.assertRaisesRegex(ValueError,'130'):
            module.aggregate_records(rows[:-1],reps=40)


if __name__=='__main__':
    unittest.main()
