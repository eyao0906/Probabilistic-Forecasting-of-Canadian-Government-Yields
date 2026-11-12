"""All historical public retrieval inputs must be represented in the replay catalog."""
from pathlib import Path
import unittest,json


class CatalogTests(unittest.TestCase):
    def test_generator_cannot_repin_modified_historical_body(self):
        import tempfile,hashlib
        from tests.support import scratch_root
        from src.source_catalog import build_catalog
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder)
            (root/'config').mkdir();(root/'data/raw').mkdir(parents=True)
            (root/'docs/sources/literature_verified').mkdir(parents=True)
            (root/'docs/sources/retrieval_log.json').write_text('[]')
            (root/'docs/sources/literature_verified/retrieval_manifest.json').write_text('[]')
            (root/'references.bib').write_text('')
            path='data/raw/boc_daily_2000_2026.csv'
            old={'id':'boc_original','path':path,'sha256':hashlib.sha256(b'original').hexdigest()}
            (root/'config/sources.json').write_text(json.dumps({'sources':[old]}))
            (root/path).write_bytes(b'replaced vintage')
            with self.assertRaisesRegex(ValueError,'pin|digest'):
                build_catalog(root)
            self.assertEqual(json.loads((root/'config/sources.json').read_text())['sources'][0],old)

    def test_catalog_covers_original_bodies_and_current_metadata(self):
        try:
            from src.source_catalog import build_catalog
        except ImportError:
            self.fail('Missing replay-catalog generator')
        root=Path(__file__).resolve().parents[1]
        catalog=build_catalog(root)
        entries=catalog['sources'];ids=[r['id'] for r in entries]
        self.assertEqual(len(ids),len(set(ids)))
        paths=[r['path'] for r in entries]
        self.assertEqual(len(paths),len(set(paths)))
        self.assertIn('boc_original',ids);self.assertIn('fed_original',ids)
        known=json.loads((root/'docs/sources/literature_verified/retrieval_manifest.json').read_text())
        self.assertTrue({r['requested_url'] for r in known}.issubset({r['url'] for r in entries}))
        for row in entries:
            if (root/row['path']).exists(): self.assertIn('sha256',row)


if __name__=='__main__': unittest.main()
