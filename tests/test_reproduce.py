"""Fail-closed integrity and isolated reproduction acceptance tests."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from tests.support import scratch_root


class InventoryTests(unittest.TestCase):
    def test_archived_python_text_is_hashed_and_syntax_checked(self):
        from src.reproduce import inventory
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder);p=root/'docs/archive_pre_resumption/scrape.py.txt'
            p.parent.mkdir(parents=True);p.write_text('def broken(:\n',encoding='utf-8')
            report=inventory(root)
            self.assertEqual(report['code_count'],1)
            self.assertEqual(report['python_count'],1)
            self.assertFalse(report['passed'])

    def test_inventory_includes_nested_acquisition_and_archived_sources(self):
        try:
            from src.reproduce import inventory
        except ImportError:
            self.fail('Missing end-to-end reproducibility runner')
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder)
            for name in ['src/acquisition.py','docs/sources/retrieve.py','results/archive_before_validation/models_executed.py','tests/test_example.py']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x = 1\n')
            p=root/'.venv/ignore.py';p.parent.mkdir();p.write_text('not python',encoding='utf-8')
            report=inventory(root)
            self.assertEqual(report['code_count'],4)
            self.assertTrue(report['passed'])
            entries={r['path']:r for r in report['files']}
            self.assertEqual(entries['src/acquisition.py']['sha256'],hashlib.sha256(b'x = 1\n').hexdigest())
            self.assertEqual(entries['docs/sources/retrieve.py']['role'],'source_acquisition')
            self.assertEqual(entries['results/archive_before_validation/models_executed.py']['role'],'historical_snapshot')
            self.assertTrue(all(r['syntax_ok'] for r in report['files']))
            (root/'src/acquisition.py').write_text('def invalid(:\n',encoding='utf-8')
            self.assertFalse(inventory(root)['passed'])


class WorkspaceTests(unittest.TestCase):
    def test_stage_copies_code_sources_and_raw_but_no_prior_outputs_or_environment(self):
        from src import reproduce
        self.assertTrue(hasattr(reproduce,'stage_workspace'),'Missing isolated staging implementation')
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder)/'project';root.mkdir()
            keep=['src/model.py','tests/test_fixture.py','config/settings.json','references.bib',
                  'data/raw/input.csv','docs/sources/paper.pdf',
                  'results/archive_before_validation/full_130_origin/mcmc_diagnostics.csv',
                  'results/state_space_revised/verify_outputs.py','paper/MRP_Draft.qmd','paper/apa.csl']
            for name in keep:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(b'x = 1\n' if p.suffix=='.py' else b'fixture')
            for name in ['results/validated_baselines/scores.csv','paper/MRP_Draft.pdf','.venv/ignore.py','logs/old.log']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'old-output')
            dest=root/'reproduction/run1'
            report=reproduce.stage_workspace(root,dest)
            for name in keep:self.assertEqual((root/name).read_bytes(),(dest/name).read_bytes())
            self.assertFalse((dest/'results/validated_baselines/scores.csv').exists())
            self.assertFalse((dest/'paper/MRP_Draft.pdf').exists())
            self.assertFalse((dest/'.venv').exists())
            self.assertEqual(report['copied_files'],len(keep))
            with self.assertRaises(FileExistsError):reproduce.stage_workspace(root,dest)
            with self.assertRaises(ValueError):reproduce.stage_workspace(root,root/'src/nested')


class ComparisonTests(unittest.TestCase):
    def test_empty_archives_and_empty_arrays_fail_closed(self):
        from src.reproduce import compare_numeric_archive
        import numpy as np
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            a=Path(folder)/'a.npz';b=Path(folder)/'b.npz'
            np.savez(a);np.savez(b)
            self.assertFalse(compare_numeric_archive(a,b)['passed'])
            np.savez(a,means=np.zeros(0));np.savez(b,means=np.zeros(0))
            self.assertFalse(compare_numeric_archive(a,b)['passed'])

    def test_numerical_comparison_rejects_missing_or_tampered_forecasts(self):
        from src import reproduce
        self.assertTrue(hasattr(reproduce,'compare_numeric_archive'),'Missing numerical verification')
        import numpy as np
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            a=Path(folder)/'a.npz';b=Path(folder)/'b.npz'
            values={'means':np.array([[1.,2.]]),'keys':np.array([['2016-01-08','1','M0']])}
            np.savez_compressed(a,**values);np.savez_compressed(b,**values)
            self.assertTrue(reproduce.compare_numeric_archive(a,b)['passed'])
            np.savez_compressed(b,means=values['means']+1,keys=values['keys'])
            self.assertFalse(reproduce.compare_numeric_archive(a,b)['passed'])
            np.savez_compressed(b,means=values['means'])
            self.assertFalse(reproduce.compare_numeric_archive(a,b)['passed'])


class RunnerTests(unittest.TestCase):
    def test_saved_commands_cover_acquisition_every_model_artifacts_pdf_and_tests(self):
        from src import reproduce
        self.assertTrue(callable(getattr(reproduce,'build_plan',None)),'Missing complete reproduction plan')
        plan=reproduce.build_plan(Path('workspace'),Path('python'))
        names=[row['name'] for row in plan]
        self.assertEqual(names,['acquisition','prepare','baselines','availability','mcs','bayesian_default',
                                'bayesian_extended','m3','m3_verify','artifacts','supplements','render','pdf_verify','tests'])
        for row in plan:
            self.assertIsInstance(row['command'],list)
            self.assertGreater(row['timeout_s'],0)
        self.assertIn('--no-previews',plan[-2]['command'])


    def test_step_records_actual_exit_codes_and_never_hides_failure(self):
        import sys
        from src import reproduce
        self.assertTrue(callable(getattr(reproduce,'run_step',None)),'Missing real subprocess execution')
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder)
            good=reproduce.run_step(root,{'name':'good','command':[sys.executable,'-c','print("actual execution")'],'timeout_s':10})
            self.assertTrue(good['passed']);self.assertEqual(good['returncode'],0)
            self.assertIn('actual execution',(root/good['log']).read_text(encoding='utf-8'))
            bad=reproduce.run_step(root,{'name':'bad','command':[sys.executable,'-c','raise SystemExit(7)'],'timeout_s':10})
            self.assertFalse(bad['passed']);self.assertEqual(bad['returncode'],7)


    def test_pipeline_stops_on_real_failure_and_records_original_preservation(self):
        import sys,json
        from src import reproduce
        self.assertTrue(callable(getattr(reproduce,'run_pipeline',None)),'Missing full isolated replay')
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder);(root/'src').mkdir()
            (root/'src/__init__.py').write_text('',encoding='utf-8')
            (root/'src/acquisition.py').write_text('raise SystemExit(7)\n',encoding='utf-8')
            report=reproduce.run_pipeline(root,root/'reproduction/run',Path(sys.executable))
            saved=json.loads((root/'reproduction/run/reproduction_report.json').read_text(encoding='utf-8'))
            self.assertEqual(saved,report)
            self.assertFalse(report['passed']);self.assertFalse(report['execution_passed'])
            self.assertTrue(report['original_artifacts_preserved'])
            self.assertEqual(len(report['steps']),1)
            self.assertEqual(report['steps'][0]['returncode'],7)
            self.assertEqual(report['planned_steps'],14)


class CliTests(unittest.TestCase):
    def test_workspace_cli_propagates_replay_failure(self):
        import sys,json
        from src import reproduce
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder);(root/'src').mkdir()
            (root/'src/__init__.py').write_text('',encoding='utf-8')
            (root/'src/acquisition.py').write_text('raise SystemExit(7)\n',encoding='utf-8')
            out=root/'reproduction/cli'
            self.assertEqual(reproduce.main(['--root',str(root),'--workspace',str(out),'--python',sys.executable]),1)
            report=json.loads((out/'reproduction_report.json').read_text(encoding='utf-8'))
            self.assertFalse(report['passed'])

    def test_cli_inventory_writes_counted_report_and_returns_failure_for_bad_code(self):
        import json
        from src import reproduce
        self.assertTrue(callable(getattr(reproduce,'main',None)), 'Missing executable reproduction CLI')
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder);(root/'src').mkdir()
            (root/'src/a.py').write_text('x=1\n',encoding='utf-8')
            report=root/'logs/integrity.json'
            self.assertEqual(reproduce.main(['--root',str(root),'--inventory-only','--report',str(report)]),0)
            saved=json.loads(report.read_text(encoding='utf-8'))
            self.assertTrue(saved['passed']);self.assertEqual(saved['code_count'],1)
            (root/'src/a.py').write_text('def broken(:\n',encoding='utf-8')
            self.assertEqual(reproduce.main(['--root',str(root),'--inventory-only','--report',str(report)]),1)
            self.assertFalse(json.loads(report.read_text(encoding='utf-8'))['passed'])


if __name__=='__main__':unittest.main()
