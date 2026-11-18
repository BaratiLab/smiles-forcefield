"""Private Ollama acquisition controller; stdout is action JSON only."""
import hashlib,json,os,sys,time,urllib.request

def acquire(state):
 model=os.getenv('OLLAMA_MODEL','qwen2.5:1.5b')
 url=os.getenv('OLLAMA_URL','http://smiles-ollama:11434')
 ids=[r['id'] for r in state['candidates']];batch=state['batch_size']
 schema={'type':'object','properties':{'candidate_ids':{'type':'array','items':{'type':'string','enum':ids},'minItems':batch,'maxItems':batch},'hypothesis':{'type':'string'}},'required':['candidate_ids','hypothesis'],'additionalProperties':False}
 system='You select experiments for fitting molecular torsion energy corrections. Choose only available IDs. Use uncertainty, angular diversity, observations, and chemical context. Avoid redundant points in a batch. You cannot see unqueried labels or test results. Return only the requested JSON. Keep hypothesis under 40 words.'
 prompt=json.dumps(state,separators=(',',':'))
 payload={'model':model,'system':system,'prompt':prompt,'stream':False,'format':schema,'keep_alive':0,'options':{'temperature':0,'seed':2026,'num_ctx':8192,'num_predict':256,'num_thread':2}}
 with urllib.request.urlopen(url+'/api/tags',timeout=20) as r:tags=json.load(r)
 digest=next((m['digest'] for m in tags['models'] if m['name']==model),None)
 if not digest:raise ValueError('Model digest unavailable')
 records=[]
 for attempt in range(2):
  start=time.monotonic()
  req=urllib.request.Request(url+'/api/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=600) as r:response=json.load(r)
  records.append({k:response.get(k) for k in ['model','created_at','total_duration','load_duration','prompt_eval_count','eval_count','done_reason','response']})
  try:
   action=json.loads(response['response']);chosen=action['candidate_ids']
   if len(chosen)!=batch or len(set(chosen))!=batch or not set(chosen)<=set(ids):raise ValueError('Invalid selection')
   action['controller']={'kind':'ollama_llm','model':model,'model_digest':digest,'prompt_version':'torsion-v1','prompt_sha256':hashlib.sha256((system+prompt).encode()).hexdigest(),'decoding':payload['options'],'attempts':records,'last_call_seconds':time.monotonic()-start}
   return action
  except (ValueError,KeyError,TypeError):
   payload['prompt']=prompt+'\nYour previous response was invalid. Select exactly '+str(batch)+' DISTINCT available IDs.'
 print(json.dumps({'failed_attempts':records}),file=sys.stderr)
 raise ValueError('Model returned invalid actions twice; no heuristic fallback')
if __name__=='__main__':print(json.dumps(acquire(json.load(sys.stdin))))
