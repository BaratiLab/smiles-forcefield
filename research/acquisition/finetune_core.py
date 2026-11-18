"""Cost-sensitive acquisition supervision and label-free prompt helpers."""
import hashlib,json
import numpy as np
from qm_benchmark import STRATEGIES

FORMULAS={
 'uniform':'Bayesian information gain IG only',
 'angular_diversity':'IG*(0.5+nearest periodic angular distance/(2*pi))',
 'sector_balance':'IG/(1+observations in same 90-degree sector)',
 'orthogonal_focus':'IG*(1+sin(angle)^2)',
 'cis_focus':'IG*(1+cos(angle))'}

def soft_targets(costs,temperature=.05):
 values=np.asarray(costs,dtype=float)
 if values.shape!=(5,) or not np.isfinite(values).all() or temperature<=0:raise ValueError('Five finite costs and positive temperature required')
 p=np.exp(-(values-values.min())/temperature);return p/p.sum()

def messages(item,rotation=0):
 # Allowlist independent of dataset IDs, energy targets and teacher costs.
 context={k:item[k] for k in ['graph','driven_dihedral','bond','pool_angles_degrees']}
 order=STRATEGIES.copy();rng=np.random.default_rng(int(hashlib.sha256(('order:'+context['graph']).encode()).hexdigest()[:8],16));rng.shuffle(order)
 order=order[rotation:]+order[:rotation]
 state={'molecule':context,'actions':{chr(65+i):{'strategy':s,'score':FORMULAS[s]} for i,s in enumerate(order)}}
 system='Choose an acquisition prior to minimize centered QM-minus-UFF test energy error. All actions fit the same Fourier harmonics 1-3 model, with 4 initial and 12 total QM labels, batches of 2. IG is conditional Bayesian information gain; angular distance and sector counts update within batch. Return exactly one action letter A-E.'
 return [{'role':'system','content':system},{'role':'user','content':json.dumps(state,separators=(',',':'))+'\nAction:'}],order

def split_checks(datasets,split):
 groups={k:{datasets[n]['scaffold'] for n in names} for k,names in split['splits'].items()}
 for a,b in [('train','validation'),('train','test'),('validation','test')]:
  if groups[a]&groups[b]:raise ValueError('Scaffold leakage')
 if set.union(*groups.values())&set(split['previous_test_scaffolds']):raise ValueError('Previous test scaffold leakage')
 if (groups['validation']|groups['test'])&set(split['all_previous_scaffolds']):raise ValueError('Fresh set overlaps previous scaffolds')
 parents=[datasets[n]['parent_smiles'] for names in split['splits'].values() for n in names]
 if len(set(parents))!=len(parents):raise ValueError('Parent leakage')
 return groups
