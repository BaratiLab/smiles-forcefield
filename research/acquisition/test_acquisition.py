import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import ollama_adapter
from poc import load_dataset
class Reply:
 def __init__(self,value):self.value=value
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def read(self):return json.dumps(self.value).encode()
class AdapterTests(unittest.TestCase):
 def test_schema_and_real_model_metadata(self):
  calls=[]
  def urlopen(req,timeout):
   calls.append(req)
   if isinstance(req,str):return Reply({'models':[{'name':'qwen2.5:0.5b','digest':'sha256:test'}]})
   payload=json.loads(req.data);self.assertEqual(payload['format']['properties']['candidate_ids']['items']['enum'],['a','b'])
   return Reply({'response':json.dumps({'candidate_ids':['b'],'hypothesis':'Explore uncertainty'}),'model':'qwen2.5:0.5b','eval_count':10})
  with patch('urllib.request.urlopen',urlopen):r=ollama_adapter.acquire({'candidates':[{'id':'a'},{'id':'b'}],'batch_size':1})
  self.assertEqual(r['candidate_ids'],['b']);self.assertEqual(r['controller']['model_digest'],'sha256:test')
 def test_invalid_action_never_falls_back(self):
  def urlopen(req,timeout):
   return Reply({'models':[{'name':'qwen2.5:0.5b','digest':'x'}]}) if isinstance(req,str) else Reply({'response':'{"candidate_ids":["secret"]}'})
  with patch('urllib.request.urlopen',urlopen),self.assertRaises(ValueError):ollama_adapter.acquire({'candidates':[{'id':'a'}],'batch_size':1})
class DatasetTests(unittest.TestCase):
 def test_duplicate_geometry_rejected_across_splits(self):
  d={'molecule':{'torsions':[[1,2,3,4]]},'units':{'angles':'radian','energy':'kcal/mol'},'metadata':{'kind':'mm_proxy'},'pool':[{'id':'a','angles_rad':[0.0],'qm_energy_kcal_mol':1.,'baseline_energy_kcal_mol':0.,'geometry_sha256':'same'}],'test':[{'id':'b','angles_rad':[0.0],'qm_energy_kcal_mol':1.,'baseline_energy_kcal_mol':0.,'geometry_sha256':'same'}]}
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'data.json';p.write_text(json.dumps(d))
   with self.assertRaisesRegex(ValueError,'Duplicate geometry'):load_dataset(p)
if __name__=='__main__':unittest.main()
