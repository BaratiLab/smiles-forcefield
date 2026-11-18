import copy,json,unittest
import numpy as np
from rdkit import Chem
from qm_data import scaffold,identity,chemistry
from qm_benchmark import planner_input,rule,select,evaluate,build_messages,STRATEGIES
from poc import Ridge,features,bayes_batch

class QMTests(unittest.TestCase):
 def fixture(self):
  m=Chem.AddHs(Chem.MolFromSmiles('CC(=O)NC'));a=np.linspace(-np.pi,np.pi,24,endpoint=False)
  records=[{'id':str(i),'angles_rad':[float(v)],'qm_energy_kcal_mol':float(np.cos(v)+.2*np.sin(2*v)),'baseline_energy_kcal_mol':0.} for i,v in enumerate(a)]
  return {'molecule':{'id':'example','smiles':'CC(=O)NC','torsions':[[1,2,4,5]],'chemistry':chemistry(m,[0,1,3,4])},'scaffold':'acyclic:CCCC','pool':records[:16],'test':records[16:]}
 def test_no_labels_or_ids_in_prompt(self):
  d=self.fixture();before=planner_input(d);changed=copy.deepcopy(d)
  for split in ['pool','test']:
   for r in changed[split]:r.update(qm_energy_kcal_mol=123456.,baseline_energy_kcal_mol=-555.,id='SECRET_ID')
  changed['molecule']['id']='SECRET_ID';changed['molecule']['secret']='SECRET_LABEL'
  self.assertEqual(before,planner_input(changed));self.assertNotIn('SECRET',json.dumps(planner_input(changed)))
 def test_scaffold_and_parent(self):
  self.assertEqual(scaffold(Chem.MolFromSmiles('Cc1ccncc1')),scaffold(Chem.MolFromSmiles('Oc1ccncc1')))
  self.assertNotEqual(scaffold(Chem.MolFromSmiles('CCC')),scaffold(Chem.MolFromSmiles('CCCC')))
  self.assertEqual(identity(Chem.MolFromSmiles('C[C@H](O)C(=O)O')),identity(Chem.MolFromSmiles('C[C@@H](O)C(=O)O')))
 def test_rule_is_driven_bond_specific(self):
  d=self.fixture();self.assertTrue(d['molecule']['chemistry']['amide_like']);self.assertEqual(rule(d['molecule']),'orthogonal_focus')
  m=Chem.AddHs(Chem.MolFromSmiles('CC(=O)NCC'))
  self.assertFalse(chemistry(m,[1,3,4,5])['amide_like'])
 def test_bayes_equivalence_and_budget(self):
  d=self.fixture();a=np.array([r['angles_rad'] for r in d['pool']]);x=features(a);init=[0,2,4,6];available=sorted(set(range(16))-set(init));model=Ridge(x[init],np.zeros(4))
  self.assertEqual(select(model,x,a,init,available,2,'uniform'),bayes_batch(model,x,available,2))
  rows,h=evaluate(d,{'a':'uniform','b':'uniform','c':'orthogonal_focus'})
  for s in range(5):
   self.assertEqual(h[f'a_{s}'],h[f'b_{s}']);self.assertEqual(h[f'a_{s}'][:4],h[f'c_{s}'][:4]);self.assertEqual(len(set(h[f'c_{s}'])),12)
 def test_test_labels_do_not_change_actions(self):
  d=self.fixture();_,before=evaluate(d,{'a':'orthogonal_focus'});changed=copy.deepcopy(d)
  for r in changed['test']:r['qm_energy_kcal_mol']+=float(r['id'])**2
  _,after=evaluate(changed,{'a':'orthogonal_focus'});self.assertEqual(before,after)
 def test_energy_zero_invariance(self):
  d=self.fixture();before,_=evaluate(d,{'a':'uniform'});changed=copy.deepcopy(d)
  for s in ['pool','test']:
   for r in changed[s]:r['qm_energy_kcal_mol']+=100000.
  after,_=evaluate(changed,{'a':'uniform'});np.testing.assert_allclose([x['rmse'] for x in before],[x['rmse'] for x in after],atol=1e-9)
 def test_prompt_options_unique_and_reproducible(self):
  d=self.fixture();f={'examples':[]};a,order=build_messages(planner_input(d),f);b,order2=build_messages(planner_input(d),f)
  self.assertEqual(a,b);self.assertEqual(set(order),set(STRATEGIES));self.assertEqual(order,order2)

if __name__=='__main__':unittest.main()
