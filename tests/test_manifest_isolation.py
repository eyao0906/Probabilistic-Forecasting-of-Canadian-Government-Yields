import csv
from tests.support import scratch_root
import tempfile
import unittest
from pathlib import Path
from src.prepare_data import prepare

class ManifestIsolation(unittest.TestCase):
    def test_fixture_processing_does_not_overwrite_project_manifest(self):
        # Tests write only next to their own output, never the production manifest.
        import src.prepare_data as p
        before=p.MANIFEST.read_bytes() if p.MANIFEST.exists() else None
        with tempfile.TemporaryDirectory(dir=scratch_root()) as temp:
            root=Path(temp); raw=root/'raw.csv'; out=root/'out'/'weekly.csv'
            with raw.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['Date',*p.COLUMNS.values()]);w.writerow(['2020-01-10',*([.02]*8)])
            prepare(raw,out)
            self.assertEqual(p.MANIFEST.read_bytes() if p.MANIFEST.exists() else None,before)
            self.assertTrue((out.parent/'source_manifest.json').exists())

if __name__=='__main__':unittest.main()
