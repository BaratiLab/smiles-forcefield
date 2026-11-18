"""Exploratory retired-test diagnosis. No confirmatory claims or model tuning."""
import argparse,csv,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from poc import Ridge,features
from qm_benchmark import STRATEGIES

def decomposition(costs,choices,reference=0):
 costs=np.asarray(costs,float);choices=np.asarray(choices,int);n=len(choices)
 probabilities=np.bincount(choices,minlength=costs.shape[1])/n
 actual=float(costs[np.arange(n),choices].mean());base=float(costs[:,reference].mean());mixture=float(costs.mean(axis=0)@probabilities)
 return {'actual_mean':actual,'bayes_mean':base,'independent_action_mixture_mean':mixture,'mixture_penalty':mixture-base,'assignment_effect':actual-mixture,'total_regret':actual-base}

def centered_rmse(y,pred):
 e=np.asarray(pred)-np.asarray(y);return float(np.sqrt(np.mean((e-e.mean())**2)))

def fit_predict(a,y,at,h):
 model=Ridge(features(a,h),y-y.mean());return model.predict(features(at,h))+y.mean()

def analyze(data,inputs_path,plans_path,runs,training,out):
 data=Path(data);runs=Path(runs);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 plans=json.loads(Path(plans_path).read_text());inputs=json.loads(Path(inputs_path).read_text());stats=json.loads((runs/'molecule_metrics.json').read_text());lookup={(r['molecule'],r['policy']):r for r in stats};names=sorted(plans)
 fixed=['bayes']+['fixed_'+s for s in STRATEGIES[1:]];costs=np.array([[lookup[n,p]['normalized_auc'] for p in fixed] for n in names]);choices=np.array([STRATEGIES.index(plans[n]['finetuned']['strategy']) for n in names]);base=costs[:,0];actual=costs[np.arange(len(names)),choices];regret=actual-base
 dec=decomposition(costs,choices);groups=[]
 for i,s in enumerate(STRATEGIES):
  mask=choices==i
  groups.append({'strategy':s,'selected_count':int(mask.sum()),'mean_regret_when_selected':float(regret[mask].mean()) if mask.any() else None,'contribution_to_overall_mean_regret':float(regret[mask].sum()/len(names)),'fixed_strategy_mean':float(costs[:,i].mean()),'oracle_best_count':int((costs.argmin(axis=1)==i).sum())})
 rng=np.random.default_rng(20261119);permuted=np.array([costs[np.arange(len(names)),rng.permutation(choices)].mean() for _ in range(10000)])
 order_costs={}
 for j in range(5):order_costs[chr(65+j)]=float(np.mean([costs[k,STRATEGIES.index(plans[n]['finetuned']['order'][j])] for k,n in enumerate(names)]))
 per_molecule=[{'molecule':n,'strategy':STRATEGIES[choices[i]],'regret':float(regret[i]),'bayes_auc':float(base[i]),'llm_auc':float(actual[i]),'best_strategy':STRATEGIES[int(costs[i].argmin())]} for i,n in enumerate(names)]
 per_molecule.sort(key=lambda r:r['regret'],reverse=True)
 # Pool-only leave-one-out model selection on exactly the retired Bayes acquisition IDs.
 harmonics=[1,2,3,4,6];capacity=[];curve_rows=[]
 for n in names:
  d=json.loads((data/'datasets'/f'{n}.json').read_text());pool=d['pool'];index={r['id']:i for i,r in enumerate(pool)}
  a=np.array([r['angles_rad'] for r in pool]);at=np.array([r['angles_rad'] for r in d['test']]);y=np.array([r['qm_energy_kcal_mol']-r['baseline_energy_kcal_mol'] for r in pool]);yt=np.array([r['qm_energy_kcal_mol']-r['baseline_energy_kcal_mol'] for r in d['test']]);scale=float(np.std(yt))
  full={h:centered_rmse(yt,fit_predict(a,y,at,h))/scale for h in harmonics}
  h3train=centered_rmse(y,fit_predict(a,y,a,3))/scale
  hfiles=json.loads((runs/f'{n}-selections.json').read_text());seed_aucs=[];cv_aucs=[];fixed_h_aucs=[]
  for seed in range(5):
   selected=[index[i] for i in hfiles[f'bayes_{seed}']];errs={h:[] for h in harmonics};cv_curve=[]
   for q in [4,6,8,10,12]:
    ids=np.array(selected[:q]);aa=a[ids];yy=y[ids];cv_scores=[]
    for h in harmonics:
     pred=fit_predict(aa,yy,at,h);errs[h].append(centered_rmse(yt,pred)/scale)
     oof=[]
     for j in range(q):
      keep=np.arange(q)!=j;oof.append(float(fit_predict(aa[keep],yy[keep],aa[j:j+1],h)[0]))
     cv_scores.append(centered_rmse(yy,oof))
    chosen=harmonics[int(np.argmin(cv_scores))];cv_curve.append(errs[chosen][-1]);curve_rows.append({'molecule':n,'seed':seed,'queries':q,'cv_harmonics':chosen,'cv_normalized_rmse':cv_curve[-1],'fixed_h3_normalized_rmse':errs[3][-1]})
   fixed_h_aucs.append([float(np.trapezoid(errs[h],dx=2)/8) for h in harmonics]);cv_aucs.append(float(np.trapezoid(cv_curve,dx=2)/8))
  mean_h=np.mean(fixed_h_aucs,axis=0);capacity.append({'molecule':n,'full_pool_test_rmse_h3':full[3],'full_pool_train_rmse_h3':h3train,'full_pool_oracle_over_h_test_rmse':min(full.values()),'h3_acquisition_auc':float(mean_h[harmonics.index(3)]),'same_ids_pool_only_cv_h_auc':float(np.mean(cv_aucs)),'same_ids_hindsight_h_oracle_auc':float(mean_h.min()),'hindsight_best_harmonics':harmonics[int(mean_h.argmin())]})
 result={'kind':'posthoc exploratory analysis of retired test set; not a fresh comparison','molecules':len(names),'action_order':STRATEGIES,'regret_decomposition':dec,'action_groups':groups,
  'same_action_frequency_permutation':{'draws':10000,'expected_mean':dec['independent_action_mixture_mean'],'permutation_2_5_97_5_percentiles':np.quantile(permuted,[.025,.975]).tolist(),'actual_percentile':float(np.mean(permuted<=actual.mean())),'interpretation':'Descriptive label-assignment diagnostic; permutations are not scaffold-adjusted causal evidence of chemical reasoning.'},
  'always_letter_counterfactual_means':order_costs,'always_angular_diversity_mean':float(costs[:,1].mean()),'menu_oracle_mean':float(costs.min(axis=1).mean()),'menu_headroom_relative_to_bayes':float(1-costs.min(axis=1).mean()/base.mean()),
  'top_five_losses':per_molecule[:5],'top_five_share_of_total_regret':float(sum(r['regret'] for r in per_molecule[:5])/regret.sum()),
  'representation_diagnostic':{k:float(np.mean([r[k] for r in capacity])) for k in ['full_pool_test_rmse_h3','full_pool_train_rmse_h3','full_pool_oracle_over_h_test_rmse','h3_acquisition_auc','same_ids_pool_only_cv_h_auc','same_ids_hindsight_h_oracle_auc']},
  'hindsight_harmonic_counts':dict(Counter(r['hindsight_best_harmonics'] for r in capacity)),
  'representation_scope':'Same retired Bayes acquisition IDs, unchanged alpha=1/noise=.1, H in [1,2,3,4,6]; CV uses observed labels only. Hindsight H uses evaluator labels and is not deployable. Full-pool references are finite-data errors, not irreducible error floors.',
  'training_validation':json.loads((Path(training)/'validation.json').read_text()),'training_unique_parents':95,'training_scaffolds':37,'test_median_nearest_train_similarity':.20204081632653062}
 (out/'diagnosis.json').write_text(json.dumps(result,indent=2));(out/'molecule-regrets.json').write_text(json.dumps(per_molecule,indent=2));(out/'representation-per-molecule.json').write_text(json.dumps(capacity,indent=2));(out/'representation-curves.json').write_text(json.dumps(curve_rows))
 print(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--inputs',required=True);p.add_argument('--plans',required=True);p.add_argument('--runs',required=True);p.add_argument('--training',required=True);p.add_argument('--out',required=True);a=p.parse_args();analyze(a.data,a.inputs,a.plans,a.runs,a.training,a.out)
