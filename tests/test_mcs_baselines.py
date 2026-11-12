"""Formal MCS wrapper must reproduce the independently maintained library."""
import unittest
import numpy as np


class MCSTests(unittest.TestCase):
    def test_range_method_matches_arch_with_stationary_bootstrap(self):
        try:
            from src.mcs_baselines import compute
        except ImportError:
            self.fail('Missing validated MCS wrapper')
        from arch.bootstrap import MCS
        rng=np.random.default_rng(9)
        losses=rng.normal(size=(100,3))+np.array([0.,.2,.5])
        expected=MCS(losses,size=.1,reps=199,block_size=4,method='R',bootstrap='stationary',seed=13)
        expected.compute()
        actual=compute(losses,['A','B','C'],reps=199,seed=13)
        for index,row in expected.pvalues.iterrows():
            self.assertAlmostEqual(actual['pvalues'][['A','B','C'][int(index)]],float(row.iloc[0]))
        self.assertEqual(actual['included'],[['A','B','C'][int(i)] for i in expected.included])
        self.assertEqual(actual['model_universe'],['A','B','C'])


if __name__=='__main__':
    unittest.main()
