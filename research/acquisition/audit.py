"""Audit matched budgets and detect a first-available presentation-order policy."""
import argparse,json
from pathlib import Path

def audit(root):
 root=Path(root);status=json.loads((root/'status.json').read_text())
 result={'completed_pairs':sum(r['status']=='completed' for r in status),'failed_pairs':sum(r['status']!='completed' for r in status),'rounds':0,'first_available_rounds':0,'attempts':0,'prompt_tokens':0,'output_tokens':0,'inference_seconds':0.,'models':[],'matched_budgets':True}
 models=set()
 for m in sorted({r['molecule'] for r in status}):
  paths=[root/m/p/'manifest.json' for p in ['random','bayes','llm']]
  if not all(p.exists() for p in paths):raise ValueError('Missing method manifest')
  manifests=[json.loads(p.read_text()) for p in paths]
  if len({(d['initial'],d['budget'],d['batch'],tuple(d['seeds']),d['dataset_sha256']) for d in manifests})!=1:raise ValueError('Unmatched experiment configuration')
  for seed in manifests[0]['seeds']:
   ids=[json.loads((root/m/p/f'{p}_{seed}'/'selected_ids.json').read_text()) for p in ['random','bayes','llm']]
   n=manifests[0]['initial'];budget=manifests[0]['budget']
   if any(len(s)!=budget or len(set(s))!=budget or s[:n]!=ids[0][:n] for s in ids):raise ValueError('Unmatched budget or initial observations')
  for path in sorted((root/m/'llm').glob('llm_*/*_response.json')):
   response=json.loads(path.read_text());request=json.loads(path.with_name(path.name.replace('_response','_request')).read_text())
   result['rounds']+=1
   result['first_available_rounds']+=response['candidate_ids']==[r['id'] for r in request['candidates'][:request['batch_size']]]
   c=response['controller'];models.add((c['model'],c['model_digest'],c['prompt_version']))
   for a in c['attempts']:
    result['attempts']+=1;result['prompt_tokens']+=a['prompt_eval_count'];result['output_tokens']+=a['eval_count'];result['inference_seconds']+=a['total_duration']/1e9
 result['models']=[dict(model=m,digest=d,prompt_version=p) for m,d,p in sorted(models)]
 result['warning']='All choices match first-available control; no demonstrated data-sensitive acquisition.' if result['rounds']==result['first_available_rounds'] else 'Order sensitivity needs counterfactual candidate-order tests.'
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=audit(a.runs);Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
