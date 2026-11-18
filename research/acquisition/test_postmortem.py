import unittest
import numpy as np
from planner_postmortem import decomposition, fit_predict

class PostmortemTests(unittest.TestCase):
 def test_decomposition_conserves_regret(self):
  d=decomposition([[1,3],[2,1],[4,2]],[1,0,1])
  self.assertAlmostEqual(d['total_regret'],d['mixture_penalty']+d['assignment_effect'])
 def test_bayes_only_has_zero_regret(self):
  d=decomposition([[1,3],[2,1],[4,2]],[0,0,0])
  self.assertEqual(d['total_regret'],0);self.assertEqual(d['assignment_effect'],0)
 def test_molecule_independent_cost_has_no_assignment_effect(self):
  d=decomposition([[1,3],[1,3],[1,3]],[0,1,1])
  self.assertAlmostEqual(d['assignment_effect'],0)
 def test_fit_is_energy_offset_invariant(self):
  a=np.linspace(-3,3,8)[:,None]; y=np.cos(a[:,0]); at=np.array([[.2],[.8]])
  np.testing.assert_allclose(fit_predict(a,y+50,at,3)-50,fit_predict(a,y,at,3),atol=1e-10)

if __name__=='__main__':unittest.main()
