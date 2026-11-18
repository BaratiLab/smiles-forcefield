"""Frozen, resumable development/held-out replay. Planner inputs contain no target labels."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from poc import Ridge,features,load_dataset
from plan_benchmark import select as original_select

STRATEGIES=['uniform','angular_diversity','sector_balance','orthogonal_focus','cis_focus']
DESCRIPTIONS={
 'uniform':'Greedy conditional Bayesian information gain (IG), no chemical weighting.',
 'angular_diversity':'IG times (0.5 + nearest periodic distance to observations/(2*pi)).',
 'sector_balance':'IG divided by (1 + observation count in same one of four angular sectors).',
 'orthogonal_focus':'IG times (1 + sin(angle)^2), emphasizing +/-90 degree rotations.',
 'cis_focus':'IG times (1 + cos(angle)), emphasizing near-zero outer-atom dihedral.'}
SEEDS=list(range(5));QUERIES=[4,6,8,10,12]

def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rule(m):
 c=m['chemistry']
 if c['amide_like'] or c['conjugated']:return 'orthogonal_focus'
 if c['both_ends_aromatic']:return 'cis_focus'
 return 'uniform'

def select(model,x,angles,selected,available,batch,strategy):
 if strategy in STRATEGIES[:3]:return original_select(model,x,angles,selected,available,batch,strategy)
 if strategy not in STRATEGIES:raise ValueError('Unknown acquisition strategy')
 cov=model.cov.copy();remaining=list(available);picked=[]
 weights=1+np.sin(angles[:,0])**2 if strategy=='orthogonal_focus' else 1+np.cos(angles[:,0])
 for _ in range(batch):
  var=np.maximum(np.einsum('ij,jk,ik->i',x[remaining],cov,x[remaining]),0.)
  score=.5*np.log1p(var/model.noise**2)*weights[remaining]
  i=remaining.pop(int(np.argmax(score)));picked.append(i)
  v=cov@x[i];cov-=np.outer(v,v)/(model.noise**2+x[i]@v)
 return picked

def planner_input(d):
 # Explicit allowlist: neither record IDs nor any energy values are serialized.
 a=np.array([r['angles_rad'][0] for r in d['pool']]);deg=np.rad2deg(a)
 return {'smiles':d['molecule']['smiles'],'driven_dihedral_atom_maps':d['molecule']['torsions'][0],
  'chemistry':dict(d['molecule']['chemistry']),
  'pool_angles_degrees':np.round(deg,1).tolist()}

def evaluate(d,mapping):
 pool=d['pool'];angles=np.array([r['angles_rad'] for r in pool]);x=features(angles);xt=features([r['angles_rad'] for r in d['test']])
 y=np.array([r['qm_energy_kcal_mol']-r['baseline_energy_kcal_mol'] for r in pool]);yt=np.array([r['qm_energy_kcal_mol']-r['baseline_energy_kcal_mol'] for r in d['test']]);yt-=yt.mean()
 baseline=float(np.sqrt(np.mean(yt**2)))
 if baseline<1e-8:raise ValueError('Degenerate normalization; do not silently remove heldout molecule')
 rows=[];histories={}
 for seed in SEEDS:
  initial=np.random.default_rng(seed).choice(len(pool),4,replace=False).tolist()
  for policy,strategy in mapping.items():
   rng=np.random.default_rng(seed+100000);selected=initial.copy()
   for q in QUERIES:
    assert len(selected)==q
    model=Ridge(x[selected],y[selected]-y[selected].mean());e=model.predict(xt)-yt;rmse=float(np.sqrt(np.mean((e-e.mean())**2)))
    rows.append({'molecule':d['molecule']['id'],'scaffold':d['scaffold'],'policy':policy,'strategy':strategy,'seed':seed,'queries':q,'rmse':rmse,'normalized_rmse':rmse/baseline,'baseline_rmse':baseline})
    if q<12:
     available=sorted(set(range(len(pool)))-set(selected))
     picked=rng.choice(available,2,replace=False).tolist() if strategy=='random' else select(model,x,angles,selected,available,2,strategy)
     selected+=picked
   histories[f'{policy}_{seed}']=[pool[i]['id'] for i in selected]
 return rows,histories

def molecule_metrics(rows):
 result=[]
 for policy in sorted({r['policy'] for r in rows}):
  curves=[[r for r in rows if r['policy']==policy and r['seed']==s] for s in SEEDS]
  norm_aucs=[float(np.trapezoid([r['normalized_rmse'] for r in c],x=QUERIES)/8) for c in curves]
  aucs=[float(np.trapezoid([r['rmse'] for r in c],x=QUERIES)/8) for c in curves]
  result.append({'molecule':rows[0]['molecule'],'scaffold':rows[0]['scaffold'],'policy':policy,'strategy':curves[0][0]['strategy'],
   'normalized_auc':float(np.mean(norm_aucs)),'auc':float(np.mean(aucs)),
   'final_rmse':float(np.mean([c[-1]['rmse'] for c in curves])),'normalized_final_rmse':float(np.mean([c[-1]['normalized_rmse'] for c in curves]))})
 return result

def develop(data,out):
 data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 split=json.loads((data/'split.json').read_text());scores=[];examples=[]
 for name in split['splits']['development']:
  d=load_dataset(data/'datasets'/f'{name}.json');rows,_=evaluate(d,{s:s for s in STRATEGIES});stats=molecule_metrics(rows);scores+=stats
  ranked=sorted(stats,key=lambda r:(r['normalized_auc'],STRATEGIES.index(r['strategy'])))
  examples.append({'input':planner_input(d),'strategy':ranked[0]['strategy'],'margin':ranked[1]['normalized_auc']-ranked[0]['normalized_auc'],'development_molecule':name})
 chosen=[]
 for s in STRATEGIES:
  candidates=[e for e in examples if e['strategy']==s and e['margin']>0]
  if candidates:chosen.append(max(candidates,key=lambda e:e['margin']))
 means={s:float(np.mean([r['normalized_auc'] for r in scores if r['strategy']==s])) for s in STRATEGIES}
 freq={s:sum(e['strategy']==s for e in examples)/len(examples) for s in STRATEGIES}
 frozen={'version':'qm-planner-v2','protocol_sha256':hashfile(Path(__file__).with_name('qm_protocol.json')),'split_sha256':hashfile(data/'split.json'),
  'development_ids':split['splits']['development'],'examples':chosen,'development_mean_normalized_auc':means,'development_strategy_frequency':freq,'global_strategy':min(means,key=means.get),
  'rule':'amide_like or conjugated -> orthogonal_focus; both_ends_aromatic -> cis_focus; otherwise uniform'}
 (out/'frozen_planner.json').write_text(json.dumps(frozen,indent=2));(out/'development_metrics.json').write_text(json.dumps(scores,indent=2))
 print(json.dumps({'development_molecules':len(examples),'means':means,'examples':[(e['development_molecule'],e['strategy']) for e in chosen]},indent=2),flush=True)

def build_messages(item,frozen):
 system='You choose a chemical prior for Bayesian acquisition of QM-minus-UFF torsion corrections. The fit is fixed additive Fourier harmonics 1-3. Select the strategy expected to minimize held-out centered energy error over a 12-label budget. All choices use identical models and budgets. Chemical weights are hypotheses, not guaranteed improvements. Answer exactly one option letter; no explanation.'
 examples=[{'chemistry':e['input']['chemistry'],'strategy':e['strategy']} for e in frozen['examples']]
 # Deterministic random option ordering prevents a fixed strategy-position association.
 order=STRATEGIES.copy();seed=int(hashlib.sha256(item['smiles'].encode()).hexdigest()[:8],16);np.random.default_rng(seed).shuffle(order)
 state={'development_examples':examples,'target':item,'options':{chr(65+i):{'strategy':s,'algorithm':DESCRIPTIONS[s]} for i,s in enumerate(order)}}
 return [{'role':'system','content':system},{'role':'user','content':json.dumps(state,separators=(',',':'))+'\nBest strategy letter:'}],order

def run(data,frozen_path,plans_path,out):
 data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 split=json.loads((data/'split.json').read_text());frozen=json.loads(Path(frozen_path).read_text());plans=json.loads(Path(plans_path).read_text())
 assert frozen['split_sha256']==hashfile(data/'split.json')
 assert set(plans)==set(split['splits']['heldout'])
 allrows=[];allstats=[]
 for name in split['splits']['heldout']:
  path=data/'datasets'/f'{name}.json';assert hashfile(path)==split['dataset_hashes'][name]
  d=load_dataset(path);plan=plans[name]
  assert plan['input_sha256']==hashlib.sha256(json.dumps(planner_input(d),sort_keys=True).encode()).hexdigest()
  # Label-free strategy assignment with development frequencies; no model calls.
  rng=np.random.default_rng(int(hashlib.sha256(('blind:'+name).encode()).hexdigest()[:8],16))
  blind=str(rng.choice(STRATEGIES,p=[frozen['development_strategy_frequency'][s] for s in STRATEGIES]))
  mapping={'bayes':'uniform','rule':rule(d['molecule']),'llm_plan':plan['strategy'],'random':'random','development_global':frozen['global_strategy'],'chemistry_blind':blind}
  mapping.update({f'fixed_{s}':s for s in STRATEGIES if s!='uniform'})
  rows,h=evaluate(d,mapping);allrows+=rows;allstats+=molecule_metrics(rows)
  (out/f'{name}-selections.json').write_text(json.dumps(h))
 with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
 (out/'molecule_metrics.json').write_text(json.dumps(allstats,indent=2))
 (out/'manifest.json').write_text(json.dumps({'split_sha256':hashfile(data/'split.json'),'frozen_planner_sha256':hashfile(frozen_path),'plans_sha256':hashfile(plans_path),
  'benchmark_sha256':hashfile(__file__),'protocol_sha256':hashfile(Path(__file__).with_name('qm_protocol.json')),'heldout_molecules':len(plans),'policies':list(mapping),'seeds':SEEDS,'queries':QUERIES},indent=2))
 (out/'COMPLETE').write_text('Heldout replay complete\n');print(f'{len(plans)} heldout molecules; {len(allrows)} metric rows',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='task',required=True)
 d=sub.add_parser('develop');d.add_argument('--data',required=True);d.add_argument('--out',required=True)
 r=sub.add_parser('run');r.add_argument('--data',required=True);r.add_argument('--frozen',required=True);r.add_argument('--plans',required=True);r.add_argument('--out',required=True)
 a=p.parse_args()
 if a.task=='develop':develop(a.data,a.out)
 else:run(a.data,a.frozen,a.plans,a.out)
