"""Revision artifacts must be generated from saved, explicitly separated results."""
import unittest
from pathlib import Path
import tempfile


class RevisionArtifacts(unittest.TestCase):
    def test_generated_tables_preserve_result_status_and_no_bayesian_forecast_scores(self):
        try:
            from src.revision_artifacts import build
        except ImportError:
            self.fail('Missing revision artifact builder')
        scratch=Path.home()/'AppData/Local/hermes/cache/scratch'
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            out=Path(folder)
            build(out,out)
            lag=(out/'availability_h4.md').read_text()
            self.assertIn('{#tbl-availability}',lag)
            self.assertIn('M0',lag)
            self.assertNotIn('M5',lag)
            diagnostics=(out/'revised_pilot.md').read_text()
            self.assertIn('{#tbl-revised-pilot}',diagnostics)
            self.assertIn('M5',diagnostics)
            self.assertNotIn('DSS',diagnostics)
            self.assertTrue((out/'revised_pilot_traces.png').exists())


if __name__=='__main__':
    unittest.main()
