import unittest,json,tempfile
from pathlib import Path
import numpy as np
from poc import Ridge,features,bayes_batch
from plan_benchmark import select,evaluate,rule_plan,STRATEGIES,request_plan
from unittest.mock import patch

class Plans(unittest.TestCase):
 def test_exactly_one_label_free_planner_call(self):
  calls=[]
  class Reply:
   def __init__(self,d):self.d=d
   def __enter__(self):return self
   def __exit__(self,*args):pass
   def read(self):return json.dumps(self.d).encode()
  def respond(req,timeout):
   if isinstance(req,str):return Reply({'models':[{'name':'test','digest':'digest'}]})
   calls.append(json.loads(req.data));return Reply({'response':'{"strategy":"uniform","rationale":"Prior coverage"}','prompt_eval_count':10,'eval_count':4})
  molecule={'id':'a','smiles':'CCCC','torsions':[[1,2,3,4]],'secret_test_energy':999,'reason':'Not passed'}
  with tempfile.TemporaryDirectory() as t,patch('urllib.request.urlopen',respond):
   result=request_plan(molecule,Path(t)/'plan','test','http://unused')
  self.assertEqual(len(calls),1);self.assertNotIn('secret_test_energy',calls[0]['prompt']);self.assertNotIn('Not passed',calls[0]['prompt']);self.assertEqual(result['calls'],1)
 def test_uniform_exactly_matches_bayesian(self):
  a=np.random.default_rng(4).uniform(-np.pi,np.pi,(30,2));x=features(a);m=Ridge(x[:4],np.zeros(4))
  self.assertEqual(select(m,x,a,list(range(4)),list(range(4,30)),4,'uniform'),bayes_batch(m,x,list(range(4,30)),4))
 def test_rule_uses_chemistry_and_dimension(self):
  self.assertEqual(rule_plan({'smiles':'CC(=O)NC','torsions':[[1,2,3,4]]}),'sector_balance')
  self.assertEqual(rule_plan({'smiles':'CCCC','torsions':[[1,2,3,4]]}),'uniform')
  self.assertEqual(rule_plan({'smiles':'OCCc1ccccc1','torsions':[[1,2,3,4],[2,3,4,5]]}),'angular_diversity')
 def test_same_plan_same_trajectory_and_initial_budget(self):
  rng=np.random.default_rng(17);a=rng.uniform(-np.pi,np.pi,(40,1));y=np.cos(a[:,0])
  d={'molecule':{'id':'test','smiles':'CCCC','torsions':[[1,2,3,4]]}}
  for split,ids in [('pool',range(24)),('test',range(24,40))]:d[split]=[{'id':str(i),'angles_rad':a[i].tolist(),'qm_energy_kcal_mol':float(y[i]),'baseline_energy_kcal_mol':0.} for i in ids]
  rows,h=evaluate(d,{'strategy':'angular_diversity'},[0,1],budget=12)
  for seed in [0,1]:
   self.assertEqual(h[f'llm_plan_{seed}'],h[f'fixed_diversity_{seed}'])
   initial=[v[:4] for k,v in h.items() if k.endswith('_'+str(seed))];self.assertTrue(all(v==initial[0] for v in initial))
  self.assertTrue(all(len(v)==12 and len(set(v))==12 for v in h.values()))
 def test_test_labels_do_not_change_selections(self):
  rng=np.random.default_rng(8);a=rng.uniform(-np.pi,np.pi,(24,1));d={'molecule':{'id':'x','smiles':'CCCC','torsions':[[1,2,3,4]]},'pool':[{'id':str(i),'angles_rad':v.tolist(),'qm_energy_kcal_mol':float(np.cos(v[0])),'baseline_energy_kcal_mol':0.} for i,v in enumerate(a)],'test':[{'id':'t','angles_rad':[0.2],'qm_energy_kcal_mol':1.,'baseline_energy_kcal_mol':0.}]}
  _,first=evaluate(d,{'strategy':'sector_balance'},[0]);d['test'][0]['qm_energy_kcal_mol']=1e6
  _,second=evaluate(d,{'strategy':'sector_balance'},[0]);self.assertEqual(first,second)
if __name__=='__main__':unittest.main()
