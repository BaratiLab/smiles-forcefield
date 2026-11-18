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
 def test_regret_selector_ignores_validation_targets(self):
  from finetune_replay import fit_regret_selector
  inputs={n:{'tabular_features':[float(i)]} for i,n in enumerate(['a','b','v'])}
  teacher={n:{'costs':{s:float(i+j)/10 for j,s in enumerate(STRATEGIES)}} for i,n in enumerate(['a','b','v'])}
  params={'n_estimators':8,'min_samples_leaf':1,'random_state':1,'n_jobs':1}
  first=fit_regret_selector(inputs,teacher,['a','b'],params)
  teacher['v']['costs']={s:1e6 for s in STRATEGIES}
  second=fit_regret_selector(inputs,teacher,['a','b'],params)
  np.testing.assert_array_equal(first.predict([[2.]]),second.predict([[2.]]))
 def test_primary_bootstrap_uses_scaffold_clusters(self):
  from finetune_analyze import paired
  stats=[{'molecule':str(i),'scaffold':str(i//2),'policy':p,'normalized_auc':float(2 if p=='finetuned_llm' else 1)} for i in range(4) for p in ['finetuned_llm','bayes']]
  result=paired(stats,'bayes');self.assertEqual(result['scaffold_cluster_95_percent_ci'],[1.,1.]);self.assertEqual(result['scaffold_clusters'],2);self.assertEqual(result['losses'],4)
 def test_compact_context_cannot_read_energy_targets(self):
  from rdkit import Chem
  from qm_data import chemistry
  from finetune_prepare import compact_input
  m=Chem.AddHs(Chem.MolFromSmiles('CC(=O)NC'))
  for i,a in enumerate(m.GetAtoms()):a.SetAtomMapNum(i+1)
  d={'molecule':{'smiles':Chem.MolToSmiles(m),'torsions':[[1,2,4,5]],'chemistry':chemistry(m,[0,1,3,4])},'parent_smiles':'CC(=O)NC','pool':[{'angles_rad':[a],'qm_energy_kcal_mol':a*100} for a in [0.,1.,2.]]}
  before=compact_input(d)
  for r in d['pool']:r['qm_energy_kcal_mol']='SECRET'
  self.assertEqual(before,compact_input(d));self.assertNotIn('SECRET',str(before))
 def test_old_test_scaffolds_cannot_train(self):
  datasets={str(i):{'scaffold':str(i),'parent_smiles':str(i)} for i in range(3)}
  split={'splits':{'train':['0'],'validation':['1'],'test':['2']},'previous_test_scaffolds':['0'],'all_previous_scaffolds':['0']}
  with self.assertRaises(ValueError):split_checks(datasets,split)

if __name__=='__main__':unittest.main()
