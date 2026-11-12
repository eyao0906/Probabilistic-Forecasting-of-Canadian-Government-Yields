"""Acquisition must fail closed and preserve pinned source evidence."""
import io
from pathlib import Path
import tempfile
import unittest


class AcquisitionTests(unittest.TestCase):
    def test_payload_validation_checks_type_not_just_http_status(self):
        try:
            from src.acquisition import validate_payload
        except ImportError:
            self.fail('Missing canonical acquisition implementation')
        with self.assertRaises(ValueError):
            validate_payload('pdf', b'<html>challenge</html>')
        with self.assertRaises(ValueError):
            validate_payload('boc_csv', b'<html>access denied</html>')
        with self.assertRaises(ValueError):
            validate_payload('crossref_json', b'{"message":{"DOI":"wrong"}}', doi='10.1/expected')
        valid=b'Date,ZC025YR,ZC050YR,ZC100YR,ZC200YR,ZC500YR,ZC700YR,ZC1000YR,ZC3000YR\n2020-01-03,.01,.01,.01,.01,.01,.01,.01,.01\n'
        self.assertEqual(validate_payload('boc_csv', valid)['rows'],1)


    def test_yield_csv_rejects_malformed_rows_and_html_token_decoys(self):
        from src.acquisition import validate_payload
        header=b'Date,ZC025YR,ZC050YR,ZC100YR,ZC200YR,ZC500YR,ZC700YR,ZC1000YR,ZC3000YR\n'
        row=b'2020-01-03,.01,.01,.01,.01,.01,.01,.01,.01\n'
        for payload in [header+row+row, header+row.replace(b'2020-01-03',b'invalid'),
                        header+row.replace(b'.01',b'inf',1), header+row.replace(b'.01',b'challenge',1),
                        header+row.rstrip()+b',extra\n']:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_payload('boc_csv',payload)
        with self.assertRaises(ValueError):
            validate_payload('fed_csv',b'<html>SVENY01 access denied</html>')
        info=validate_payload('boc_csv',header+row+row.replace(b'2020-01-03',b'2020-01-06').replace(b'.01',b'na'))
        self.assertEqual(info['complete_rows'],1)
        self.assertEqual(info['rows'],2)


    def test_safe_cached_and_live_acquisition_preserves_evidence(self):
        import hashlib
        import urllib.error
        from src import acquisition as module
        self.assertTrue(hasattr(module,'acquire_one'),'Missing cache/download implementation')
        scratch=Path.home()/'AppData/Local/hermes/cache/scratch';scratch.mkdir(parents=True,exist_ok=True)
        body=b'original public evidence'
        spec={'id':'page','url':'https://example.org/page','path':'sources/page.txt','kind':'text','sha256':hashlib.sha256(body).hexdigest()}
        class Response(io.BytesIO):
            status=200
            headers={'Content-Type':'text/plain'}
            def geturl(self): return spec['url']
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root=Path(folder)
            no_network=lambda *a,**k: self.fail('Offline or cache hit made a network request')
            with self.assertRaises(FileNotFoundError): module.acquire_one(spec,root,offline=True,opener=no_network)
            for key in ['../escape.txt','C:/escape.txt','/escape.txt']:
                with self.assertRaises(ValueError): module.acquire_one({**spec,'path':key},root,offline=True)
            report=module.acquire_one(spec,root,opener=lambda *a,**k:Response(body))
            self.assertEqual(report['status'],'downloaded')
            self.assertEqual(module.acquire_one(spec,root,offline=True,opener=no_network)['status'],'cached_verified')
            (root/spec['path']).write_bytes(b'tampered')
            with self.assertRaises(ValueError): module.acquire_one(spec,root,offline=True,opener=no_network)
            bad={**spec,'path':'sources/expected.pdf','kind':'pdf'}
            with self.assertRaises(ValueError): module.acquire_one(bad,root,opener=lambda *a,**k:Response(b'<html>blocked</html>'))
            self.assertFalse((root/bad['path']).exists())
            partial={**spec,'path':'sources/interrupted.txt'}
            def broken(*args,**kwargs): raise OSError('interrupted transfer')
            with self.assertRaises(OSError): module.acquire_one(partial,root,opener=broken)
            self.assertFalse((root/partial['path']).exists())


    def test_direct_download_enforces_robots_and_preserves_policy_body(self):
        from src import acquisition as module
        from unittest.mock import patch
        from tests.support import scratch_root
        body=b'User-agent: *\nDisallow: /private\n'
        calls=[]
        class Response(io.BytesIO):
            status=200
            headers={'Content-Type':'text/plain'}
            def geturl(self): return 'https://example.org/robots.txt'
        def fake(request,**kwargs):
            calls.append(request.full_url)
            return Response(body)
        spec={'id':'private','url':'https://example.org/private','path':'source.txt','kind':'text'}
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder)
            with patch('urllib.request.urlopen',side_effect=fake), patch('urllib.request.build_opener') as factory:
                factory.return_value.open.side_effect=fake
                with self.assertRaises(PermissionError): module.acquire_one(spec,root)
            self.assertEqual(calls,['https://example.org/robots.txt'])
            self.assertFalse((root/'source.txt').exists())
            self.assertTrue(any(p.read_bytes()==body for p in (root/'_robots').glob('*.body')))


    def test_incomplete_http_body_is_preserved_with_failed_transaction(self):
        from src import acquisition as module
        from tests.support import scratch_root
        import json
        body=b'incomplete source'
        spec={'id':'short','url':'https://example.org/file','path':'file.txt','kind':'text'}
        class Response(io.BytesIO):
            status=200
            headers={'content-length':str(len(body)+1),'Content-Type':'text/plain'}
            def geturl(self): return spec['url']
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder)
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                module.acquire_one(spec,root,opener=lambda *a,**k:Response(body))
            self.assertFalse((root/'file.txt').exists())
            self.assertTrue(any(p.read_bytes()==body for p in (root/'_failed').glob('*.body')))
            transactions=[json.loads(p.read_text()) for p in (root/'_transactions').glob('*.json')]
            self.assertEqual(len(transactions),1)
            self.assertEqual(transactions[0]['status'],'error')
            self.assertEqual(transactions[0]['http_status'],200)
            self.assertEqual(transactions[0]['bytes'],len(body))


    def test_catalog_and_pdf_replay_are_portable_and_cache_only(self):
        from src import acquisition as module
        self.assertTrue(hasattr(module,'run_catalog'),'Missing explicit replay runner')
        self.assertTrue(hasattr(module,'extract_pdf'),'Missing deterministic PDF extraction')
        from pypdf import PdfWriter
        import hashlib,json
        writer=PdfWriter();writer.add_blank_page(width=100,height=100)
        buf=io.BytesIO();writer.write(buf);pdf=buf.getvalue()
        scratch=Path.home()/'AppData/Local/hermes/cache/scratch'
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root=Path(folder);(root/'doc.pdf').write_bytes(pdf)
            spec={'id':'doc','url':'https://example.org/doc','path':'doc.pdf','kind':'pdf','sha256':hashlib.sha256(pdf).hexdigest()}
            report=module.run_catalog([spec],root,offline=True)
            self.assertTrue(report['passed'])
            info=module.extract_pdf(root/'doc.pdf',root/'derived/doc.txt')
            self.assertEqual(info['pages'],1)
            self.assertEqual((root/'derived/doc.txt').read_text(encoding='utf-8'),'===== PDF PAGE 1 =====\n')
            self.assertIn('pypdf_version',info)
            with self.assertRaises(ValueError): module.run_catalog([spec,spec],root,offline=True)
            bad={**spec,'id':'missing','path':'missing.pdf'}
            self.assertFalse(module.run_catalog([bad],root,offline=True)['passed'])


if __name__=='__main__':
    unittest.main()
