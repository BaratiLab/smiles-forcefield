"""One chemical plan per molecule, then deterministic Bayesian acquisition.
Proxy replay only. Test labels never enter the planner or acquisition selector.
"""
import argparse,csv,hashlib,json,os,time,urllib.request
from pathlib import Path
import numpy as np
from poc import Ridge,features,load_dataset

STRATEGIES={
 'uniform':'Greedy conditional Bayesian information gain over all candidates, without extra weights.',
 'angular_diversity':'Multiply information gain by 0.5 + minimum periodic angular distance / (2*pi); update distance after each selected point.',
 'sector_balance':'Divide information gain by 1 + observations in the candidate angular sector. Four equal bins per torsion, joint bins for multiple torsions; update within batch.'}
POLICIES=['bayes','rule','llm_plan','fixed_diversity','fixed_sectors']

def rule_plan(molecule):
 from rdkit import Chem
 m=Chem.MolFromSmiles(molecule['smiles'])
 if len(molecule['torsions'])>1:return 'angular_diversity'
 # Freeze this SMARTS rule before model evaluation; no molecule-name lookup.
 if m.HasSubstructMatch(Chem.MolFromSmarts('C(=O)N')):return 'sector_balance'
 return 'uniform'

def select(model,x,angles,selected,available,batch,strategy):
 if strategy not in STRATEGIES:raise ValueError('Unknown strategy')
 cov=model.cov.copy();remaining=list(available);picked=[];observed=list(selected)
 sectors=np.floor(((angles+np.pi)%(2*np.pi))/(np.pi/2)).astype(int)
 for _ in range(batch):
  variance=np.maximum(np.einsum('ij,jk,ik->i',x[remaining],cov,x[remaining]),0.)
  scores=.5*np.log1p(variance/model.noise**2)
  if strategy=='angular_diversity':
   delta=angles[remaining,None,:]-angles[None,observed,:]
   distance=np.abs((delta+np.pi)%(2*np.pi)-np.pi).mean(axis=2).min(axis=1)
   scores*=.5+distance/(2*np.pi)
  elif strategy=='sector_balance':
   counts=(sectors[remaining,None,:]==sectors[None,observed,:]).all(axis=2).sum(axis=1)
   scores/=1+counts
  idx=remaining.pop(int(np.argmax(scores)));picked.append(idx);observed.append(idx)
  v=cov@x[idx];cov-=np.outer(v,v)/(model.noise**2+x[idx]@v)
 return picked

def request_plan(molecule,folder,model,url):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
 seed=int(hashlib.sha256(molecule['smiles'].encode()).hexdigest()[:8],16)
 options=list(STRATEGIES);np.random.default_rng(seed).shuffle(options)
 state={'molecule':{k:molecule[k] for k in ['id','smiles','torsions']},'objective':'Choose one acquisition prior for an additive Fourier torsion correction, harmonics 1-3, with fixed observation noise. No label data are available.','strategies':[{'name':k,'algorithm':STRATEGIES[k]} for k in options],'instruction':'Return strategy and a brief chemical rationale. Uniform information gain is a valid choice; extra diversity or sector weighting need not help.'}
 schema={'type':'object','properties':{'strategy':{'type':'string','enum':options},'rationale':{'type':'string'}},'required':['strategy','rationale'],'additionalProperties':False}
 with urllib.request.urlopen(url+'/api/tags',timeout=20) as r:tags=json.load(r)
 digest=next((m['digest'] for m in tags['models'] if m['name']==model),None)
 if not digest:raise ValueError('Requested model digest not found')
 payload={'model':model,'prompt':json.dumps(state,separators=(',',':')),'system':'You are a molecular simulation experiment planner. Choose one implemented strategy. Do not invent energies. Keep rationale under 25 words. Return JSON only.','format':schema,'stream':False,'keep_alive':'60s','options':{'temperature':0,'seed':2026,'num_ctx':2048,'num_predict':192,'num_thread':2}}
 (folder/'request.json').write_text(json.dumps(payload,indent=2))
 start=time.monotonic();req=urllib.request.Request(url+'/api/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=600) as r:response=json.load(r)
 elapsed=time.monotonic()-start
 (folder/'response.json').write_text(json.dumps(response,indent=2))
 plan=json.loads(response['response'])
 if plan.get('strategy') not in STRATEGIES or not isinstance(plan.get('rationale'),str):raise ValueError('Invalid model plan; no fallback')
 record={'strategy':plan['strategy'],'rationale':plan['rationale'],'model':model,'model_digest':digest,'seconds':elapsed,'prompt_tokens':response['prompt_eval_count'],'output_tokens':response['eval_count'],'strategy_presentation':options,'first_option_selected':plan['strategy']==options[0],'prompt_version':'one-plan-v1','calls':1}
 (folder/'plan.json').write_text(json.dumps(record,indent=2));return record

def evaluate(d,plan,seeds,budget=12,initial=4,batch=4):
 if not 1<=initial<budget<=len(d['pool']) or batch<1:raise ValueError('Invalid budget')
 pool=d['pool'];a=np.array([r['angles_rad'] for r in pool]);x=features(a);xt=features([r['angles_rad'] for r in d['test']])
 y=np.array([r['qm_energy_kcal_mol']-r['baseline_energy_kcal_mol'] for r in pool]);yt=np.array([r['qm_energy_kcal_mol']-r['baseline_energy_kcal_mol'] for r in d['test']])
 rules=rule_plan(d['molecule']);rows=[];histories={}
 mapping={'bayes':'uniform','rule':rules,'llm_plan':plan['strategy'],'fixed_diversity':'angular_diversity','fixed_sectors':'sector_balance'}
 for seed in seeds:
  start=np.random.default_rng(seed).choice(len(pool),initial,replace=False).tolist()
  for policy in POLICIES:
   selected=start.copy()
   while True:
    model=Ridge(x[selected],y[selected]);err=model.predict(xt)-yt;rmse=float(np.sqrt(np.mean((err-err.mean())**2)))
    rows.append({'molecule':d['molecule']['id'],'policy':policy,'strategy':mapping[policy],'seed':seed,'queries':len(selected),'rmse':rmse})
    if len(selected)==budget:break
    available=sorted(set(range(len(pool)))-set(selected))
    selected+=select(model,x,a,selected,available,min(batch,budget-len(selected)),mapping[policy])
   histories[f'{policy}_{seed}']=[pool[i]['id'] for i in selected]
 return rows,histories

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4]);p.add_argument('--model',default='qwen2.5:1.5b');p.add_argument('--url',default=os.getenv('OLLAMA_URL','http://smiles-ollama:11434'));p.add_argument('--budget',type=int,default=12);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rows=[];plans={};manifest={'model':a.model,'seeds':a.seeds,'budget':a.budget,'initial':4,'batch':4,'data_hashes':{},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dataset_kinds':{}}
 molecules=json.loads((Path(__file__).parent/'molecules.json').read_text())
 for m in molecules:
  path=Path(a.data)/(m['id']+'.json');d=load_dataset(path);manifest['data_hashes'][m['id']]=hashlib.sha256(path.read_bytes()).hexdigest();manifest['dataset_kinds'][m['id']]=d['metadata']['kind']
  plan=request_plan(d['molecule'],out/'plans'/m['id'],a.model,a.url);plans[m['id']]=plan
  r,h=evaluate(d,plan,a.seeds,a.budget);rows+=r;(out/(m['id']+'-selections.json')).write_text(json.dumps(h,indent=2))
  (out/'plans.json').write_text(json.dumps(plans,indent=2));print(m['id'],plan['strategy'],round(plan['seconds'],2),'seconds',flush=True)
  with (out/'metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 summary=[]
 for m in molecules:
  for policy in POLICIES:
   curves=[[r for r in rows if r['molecule']==m['id'] and r['policy']==policy and r['seed']==s] for s in a.seeds]
   finals=[c[-1]['rmse'] for c in curves];aucs=[sum((b['queries']-a['queries'])*(a['rmse']+b['rmse'])/2 for a,b in zip(c,c[1:]))/(c[-1]['queries']-c[0]['queries']) for c in curves]
   summary.append({'molecule':m['id'],'policy':policy,'strategy':curves[0][0]['strategy'],'mean_final_rmse':float(np.mean(finals)),'sd_final_rmse':float(np.std(finals,ddof=1)),'mean_error_auc':float(np.mean(aucs))})
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'COMPLETE').write_text('5 molecules; 5 policies; '+str(len(a.seeds))+' matched seeds; one LLM plan per molecule\n')
if __name__=='__main__':main()
