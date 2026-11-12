"""Regression fixtures for non-destructive, finite canonical data preparation."""
import csv, io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from src import prepare_data


class PrepareSafetyTests(unittest.TestCase):
    def test_raw_input_cannot_be_used_as_output(self):
        scratch=Path.home()/'AppData/Local/hermes/cache/scratch'
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            path=Path(folder)/'raw.csv';path.write_text('original bytes',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'same|raw|identical'):
                prepare_data.prepare(path,path)
            self.assertEqual(path.read_text(),'original bytes')

    def test_nonfinite_rows_are_missing_not_complete_curves(self):
        scratch=Path.home()/'AppData/Local/hermes/cache/scratch'
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root=Path(folder);raw=root/'raw.csv'
            columns=['Date',*prepare_data.COLUMNS.values()]
            with raw.open('w',newline='') as handle:
                writer=csv.writer(handle);writer.writerow(columns)
                writer.writerow(['2020-01-02',*([.01]*8)])
                writer.writerow(['2020-01-03',float('nan'),*([.01]*7)])
                writer.writerow(['2020-01-06',float('inf'),*([.01]*7)])
            rows,manifest=prepare_data.prepare(raw,root/'processed/weekly.csv')
            self.assertEqual(len(rows),1)
            self.assertEqual(manifest['rows_missing_selected_maturities'],2)

    def test_download_respects_custom_raw_path_and_real_provenance(self):
        scratch=Path.home()/'AppData/Local/hermes/cache/scratch'
        body=b'Date,ZC025YR,ZC050YR,ZC100YR,ZC200YR,ZC500YR,ZC700YR,ZC1000YR,ZC3000YR\n2020-01-03,.01,.01,.01,.01,.01,.01,.01,.01\n'
        class Response(io.BytesIO):
            status=200;headers={'Content-Type':'text/plain'}
            def __init__(self,payload,url): super().__init__(payload);self.url=url
            def geturl(self): return self.url
        calls=[]
        def serve(request,**kwargs):
            url=request.full_url;calls.append(url)
            payload=b'User-agent: *\nAllow: /\n' if url.endswith('/robots.txt') else body
            return Response(payload,url)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            raw=Path(folder)/'vintage/raw.csv'
            with patch('urllib.request.build_opener') as factory:
                factory.return_value.open.side_effect=serve
                target=prepare_data.download_raw('2020-01-01','2020-01-31',destination=raw)
            self.assertEqual(calls,['https://www.bankofcanada.ca/robots.txt',
                                    'https://www.bankofcanada.ca/stats/results/csv?lookupPage=lookup_yield_curve.php&dFrom=2020-01-01&dTo=2020-01-31'])
            self.assertEqual(target,raw)
            self.assertEqual(raw.read_bytes(),body)
            rows,m=prepare_data.prepare(raw,Path(folder)/'processed/weekly.csv')
            self.assertEqual(m['retrieval_evidence'],'Authenticated download metadata with matching SHA-256')
            self.assertIn('retrieved_utc',m['download_transaction'])
            with self.assertRaises(FileExistsError):
                prepare_data.download_raw('2020-01-01','2020-01-31',destination=raw)


if __name__=='__main__': unittest.main()
