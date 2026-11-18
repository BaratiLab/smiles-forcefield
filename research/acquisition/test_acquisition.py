import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import ollama_adapter
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
   if isinstance(req,str):return Reply({'models':[{'name':'qwen2.5:1.5b','digest':'sha256:test'}]})
   payload=json.loads(req.data);self.assertEqual(payload['format']['properties']['candidate_ids']['items']['enum'],['a','b'])
   return Reply({'response':json.dumps({'candidate_ids':['b'],'hypothesis':'Explore uncertainty'}),'model':'qwen2.5:1.5b','eval_count':10})
  with patch('urllib.request.urlopen',urlopen):r=ollama_adapter.acquire({'candidates':[{'id':'a'},{'id':'b'}],'batch_size':1})
  self.assertEqual(r['candidate_ids'],['b']);self.assertEqual(r['controller']['model_digest'],'sha256:test')
 def test_invalid_action_never_falls_back(self):
  def urlopen(req,timeout):
   return Reply({'models':[{'name':'qwen2.5:1.5b','digest':'x'}]}) if isinstance(req,str) else Reply({'response':'{"candidate_ids":["secret"]}'})
  with patch('urllib.request.urlopen',urlopen),self.assertRaises(ValueError):ollama_adapter.acquire({'candidates':[{'id':'a'}],'batch_size':1})
if __name__=='__main__':unittest.main()
