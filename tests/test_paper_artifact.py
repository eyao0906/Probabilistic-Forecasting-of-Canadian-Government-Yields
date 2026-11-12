"""Artifact acceptance: inspect actual PDF text, not render success alone."""
import unittest
from pathlib import Path
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]

class PdfAcceptance(unittest.TestCase):
    def test_front_matter_lists_have_entries(self):
        texts=[p.extract_text() for p in PdfReader(ROOT/'paper/MRP_Draft.pdf').pages]
        self.assertIn('Probabilistic Forecasting',texts[0])
        tables=[t for t in texts if t.startswith('List of Tables')]
        figures=[t for t in texts if t.startswith('List of Figures')]
        self.assertEqual(len(tables),1)
        self.assertEqual(len(figures),1)
        self.assertIn('Weekly Canadian yields',tables[0])
        self.assertIn('Selected Canadian yield histories',figures[0])

    def test_physical_page_count_is_in_guideline_range(self):
        count=len(PdfReader(ROOT/'paper/MRP_Draft.pdf').pages)
        self.assertTrue(25<=count<=35)

if __name__=='__main__':unittest.main()
