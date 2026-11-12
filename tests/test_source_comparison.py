"""The fresh public vintage must be compared by dated maturity cells, not bytes."""
from pathlib import Path
import unittest


class SourceComparisonTests(unittest.TestCase):
    def test_missing_dates_and_incomplete_rows_are_explicit_not_silent(self):
        import tempfile
        from tests.support import scratch_root
        from src.verify_data_sources import compare,COLS
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder);header='Date,'+','.join(COLS)+'\n'
            def row(day,value): return day+','+','.join([value]*len(COLS))+'\n'
            original=root/'a.csv';fresh=root/'b.csv'
            original.write_text(header+row('2020-01-02','.01')+row('2020-01-03','.02')+row('2020-01-06','.03'))
            fresh.write_text(header+row('2020-01-02','.01')+row('2020-01-03','na'))
            report=compare(original,fresh,start='2020-01-01',end='2020-12-31')
            self.assertEqual(report['compared_complete_dates'],1)
            self.assertEqual(report['original_only_raw_dates_in_range'],['2020-01-06'])
            self.assertEqual(report['fresh_incomplete_dates_in_range'],1)
            self.assertFalse(report['research_panel_equal'])

    def test_fed_comparison_reads_preamble_and_normalizes_percentage_points(self):
        import tempfile
        from tests.support import scratch_root
        from src.verify_data_sources import compare
        fields=['Date']+[f'SVENY{i:02d}' for i in range(1,31)]+['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2']
        header='Staff product\n'+','.join(fields)+'\n'
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            root=Path(folder);a=root/'a.csv';b=root/'b.csv'
            a.write_text(header+'2020-01-02,'+','.join(['1']*36)+'\n')
            b.write_text(header+'2020-01-02,'+','.join(['2']*36)+'\n')
            report=compare(a,b,kind='fed_csv')
            self.assertEqual(report['dates_with_difference'],1)
            self.assertAlmostEqual(report['maximum_abs_decimal_yield_difference'],.01)
            self.assertEqual(report['differing_cells'],30)

    def test_daily_comparison_is_recomputable_from_two_files(self):
        try:
            from src.verify_data_sources import compare
        except ImportError:
            self.fail('Missing saved-code source comparison')
        root=Path(__file__).resolve().parents[1]
        report=compare(root/'data/raw/boc_daily_2000_2026.csv',root/'docs/sources/boc_source_check_2015.txt')
        self.assertEqual(report['compared_complete_dates'],21)
        self.assertEqual(report['dates_with_difference'],0)
        self.assertEqual(report['maximum_abs_decimal_yield_difference'],0.)


if __name__=='__main__':unittest.main()
