import copy,unittest
import numpy as np
from finetune_core import soft_targets,messages,split_checks
from qm_benchmark import STRATEGIES

class FineTuneTests(unittest.TestCase):
 def context(self):return {'graph':'[CH3:1][C:2](=O)[NH:4][CH3:5]','driven_dihedral':[1,2,4,5],'bond':{'amide_like':True},'pool_angles_degrees':[0,30,60]}
 def test_soft_labels_prefer_lower_cost(self):
  p=soft_targets([.3,.2,.5,.4,.9]);self.assertEqual(int(np.argmax(p)),1);self.assertAlmostEqual(p.sum(),1.)
  np.testing.assert_allclose(p,soft_targets(np.array([.3,.2,.5,.4,.9])+100))
 def test_soft_labels_ties_and_invalid(self):
  np.testing.assert_allclose(soft_targets([2]*5),np.ones(5)/5)
  with self.assertRaises(ValueError):soft_targets([1,2,np.nan,4,5])
 def test_latin_option_balance(self):
  orders=[messages(self.context(),i)[1] for i in range(5)]
  for s in STRATEGIES:self.assertEqual(sorted(order.index(s) for order in orders),list(range(5)))
 def test_prompt_hides_extra_fields(self):
  a=self.context();b=copy.deepcopy(a);b.update(qm_energy='SECRET_LABEL',teacher_costs='SECRET_COST',id='SECRET_ID')
  self.assertEqual(messages(a),messages(b));self.assertNotIn('SECRET',str(messages(b)))
 def test_fresh_scaffold_embargo(self):
  datasets={str(i):{'scaffold':str(i),'parent_smiles':str(i)} for i in range(3)}
  split={'splits':{'train':['0'],'validation':['1'],'test':['2']},'previous_test_scaffolds':['9'],'all_previous_scaffolds':['0','9']}
  split_checks(datasets,split)
  datasets['2']['scaffold']='0'
  with self.assertRaises(ValueError):split_checks(datasets,split)
 def test_old_test_scaffolds_cannot_train(self):
  datasets={str(i):{'scaffold':str(i),'parent_smiles':str(i)} for i in range(3)}
  split={'splits':{'train':['0'],'validation':['1'],'test':['2']},'previous_test_scaffolds':['0'],'all_previous_scaffolds':['0']}
  with self.assertRaises(ValueError):split_checks(datasets,split)

if __name__=='__main__':unittest.main()
