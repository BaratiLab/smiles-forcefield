"""Audit and paired scaffold-cluster inference for a completed frozen held-out run."""
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from qm_benchmark import STRATEGIES,planner_input,hashfile

def paired_bootstrap(stats,control,rng):
 lookup={(r['molecule'],r['policy']):r for r in stats}
 names=sorted({r['molecule'] for r in stats});groups={}
 for m in names:
  row=lookup[m,'llm_plan'];delta=row['normalized_auc']-lookup[m,control]['normalized_auc']
  groups.setdefault(row['scaffold'],[]).append(delta)
 sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
 draws=rng.integers(len(groups),size=(10000,len(groups)))
 effects=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
 delta=np.array([lookup[m,'llm_plan']['normalized_auc']-lookup[m,control]['normalized_auc'] for m in names])
 return {'comparison':'llm_plan minus '+control,'mean_paired_normalized_auc_difference':float(delta.mean()),
  'scaffold_cluster_97.5_percent_ci':np.quantile(effects,[.0125,.9875]).tolist(),
  'molecule_wins':int((delta < -1e-12).sum()),'ties':int((abs(delta)<=1e-12).sum()),'losses':int((delta>1e-12).sum()),
  'bootstrap_draws':10000,'scaffold_clusters':len(groups)}

def analyze(data,frozen,plans_dir,runs,out):
 data=Path(data);plans_dir=Path(plans_dir);runs=Path(runs);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 split=json.loads((data/'split.json').read_text());frozen_json=json.loads(Path(frozen).read_text());plans=json.loads((plans_dir/'plans.json').read_text())
 stats=json.loads((runs/'molecule_metrics.json').read_text());manifest=json.loads((runs/'manifest.json').read_text())
 assert (runs/'COMPLETE').exists() and (plans_dir/'COMPLETE').exists()
 dev=[json.loads((data/'datasets'/f'{m}.json').read_text()) for m in split['splits']['development']]
 test=[json.loads((data/'datasets'/f'{m}.json').read_text()) for m in split['splits']['heldout']]
 assert not {d['scaffold'] for d in dev}&{d['scaffold'] for d in test}
 assert len({d['parent_smiles'] for d in dev+test})==360
 assert len(test)==300 and len(plans)==300 and len(stats)==300*len(manifest['policies'])
 assert sum(p['calls'] for p in plans.values())==300
 assert manifest['plans_sha256']==hashfile(plans_dir/'plans.json') and manifest['frozen_planner_sha256']==hashfile(frozen)
 for d in dev+test:
  m=d['molecule']['id'];assert hashfile(data/'datasets'/f'{m}.json')==split['dataset_hashes'][m]
  records=d['pool']+d['test'];assert len({r['geometry_sha256'] for r in records})==24
 for d in test:
  m=d['molecule']['id'];item=planner_input(d);plan=plans[m]
  assert plan['input_sha256']==hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
  request=json.loads((plans_dir/m/'request.json').read_text());assert hashlib.sha256(request['prompt'].encode()).hexdigest()==plan['prompt_sha256']
  state=json.loads(request['messages'][1]['content'].split('\nBest strategy letter:')[0]);assert state['target']==item
  assert state['development_examples']==[{'chemistry':e['input']['chemistry'],'strategy':e['strategy']} for e in frozen_json['examples']]
  h=json.loads((runs/f'{m}-selections.json').read_text());fixed='bayes' if plan['strategy']=='uniform' else 'fixed_'+plan['strategy']
  pool_ids={r['id'] for r in d['pool']};test_ids={r['id'] for r in d['test']}
  for s in range(5):
   assert h[f'llm_plan_{s}']==h[f'{fixed}_{s}']
   for p in manifest['policies']:
    ids=h[f'{p}_{s}'];assert len(ids)==len(set(ids))==12 and set(ids)<=pool_ids and not set(ids)&test_ids
    assert ids[:4]==h[f'bayes_{s}'][:4]
 comparisons=[paired_bootstrap(stats,c,np.random.default_rng(20261117+i)) for i,c in enumerate(['bayes','rule','development_global','chemistry_blind'])]
 means={p:{k:float(np.mean([r[k] for r in stats if r['policy']==p])) for k in ['normalized_auc','auc','final_rmse','normalized_final_rmse']} for p in manifest['policies']}
 primary_pass=all(c['scaffold_cluster_97.5_percent_ci'][1]<0 for c in comparisons[:2])
 audit={'heldout_molecules':300,'development_molecules':60,'heldout_scaffolds':len({d['scaffold'] for d in test}),'development_scaffolds':len({d['scaffold'] for d in dev}),
  'scaffold_overlap':0,'generation_calls':300,'trajectories':300*len(manifest['policies'])*5,'test_qm_geometries':2400,
  'all_input_hashes_match':True,'all_prompt_inputs_label_free_allowlist':True,'all_budgets_initial_sets_and_pool_membership_match':True,'all_llm_actions_reproduce_fixed_strategy_control':True,
  'strategy_counts':dict(Counter(p['strategy'] for p in plans.values())),
  'letter_counts':dict(Counter(p['chosen_letter'] for p in plans.values())),
  'planning_seconds':sum(p['seconds'] for p in plans.values())+json.loads((plans_dir/'prefix-cache-runtime.json').read_text())['prefill_seconds'],
  'shared_prefix_cache':json.loads((plans_dir/'prefix-cache-runtime.json').read_text()),'median_planning_seconds':float(np.median([p['seconds'] for p in plans.values()])),
  'prompt_tokens':sum(p['prompt_tokens'] for p in plans.values()),'output_tokens':300,
  'primary_improvement_criterion_passed':primary_pass,'comparisons':comparisons,'means':means}
 (out/'analysis.json').write_text(json.dumps(audit,indent=2))
 lines=['# Held-out QM acquisition benchmark','', '**Outcome: '+('the prespecified primary criterion passed.' if primary_pass else 'the prespecified LLM-improvement criterion did not pass.')+'** The criterion requires lower mean normalized error-query AUC than both pure Bayes and the frozen chemistry rule, with each paired scaffold-cluster 97.5% confidence interval entirely below zero.','', '## Experiment','', '300 unique held-out molecular parents across '+str(audit['heldout_scaffolds'])+' scaffold groups; 60 development molecules across '+str(audit['development_scaffolds'])+' disjoint groups. Five matched initialization seeds per molecule; 4 initial labels, batches of 2, 12 labels total, 16-point acquisition pools and 8-point unseen conformer tests. The source consists of complete 24-point one-dimensional published QM torsion scans. Ten policies give 15,000 trajectories.','', 'The target is a Fourier correction to UFF energies at the identical QM geometries. QM labels are from OpenFF Sage 2.2.1 ForceBalance targets, with energies converted from hartree to kcal/mol. UFF is evaluated without minimizing geometry. These are real QM labels, but this is a finite-pool acquisition replay, not an MD-validation study or an export-accuracy study.','', '## Planner changes','', 'Qwen2.5-7B-Instruct in 4-bit MLX format replaces the 1.5B model. The prompt contains the mapped molecular graph, driven atom maps, bond-specific chemistry flags and unlabeled pool angles. Five feature-only development examples are selected by a frozen, per-strategy best-margin rule on the 60 development molecules. No held-out energies, record IDs, or evaluation statistics enter a planner request. One constrained A–E next-token decision chooses a Bayesian acquisition weighting, and that plan is reused for all five seeds. This is in-context learning, not weight fine-tuning. All option log probabilities, prompts, model-file hashes and decisions are saved.','', 'The same Fourier harmonics 1–3, Gaussian prior and fixed observation noise are used in every policy. All strategies choose by information gain; extra actions multiply it by diversity, sector balance, orthogonal-angle emphasis, or cis-angle emphasis. The rule uses the same bond descriptors: amide-like or conjugated -> orthogonal focus; otherwise both ends aromatic -> cis focus; otherwise uniform. The best global development strategy and a chemistry-blind development-frequency draw are additional non-LLM controls.','', '## Held-out metrics','', 'Molecule means after averaging seeds. Lower is better. Normalized AUC is the trapezoidal mean error across 4–12 queries divided by the centered baseline QM-minus-UFF test RMSE. This normalizer is used by the evaluator only.','', '| Policy | Normalized AUC | AUC (kcal/mol) | Final RMSE (kcal/mol) |','|---|---:|---:|---:|']
 for p,m in means.items():lines.append(f"| {p} | {m['normalized_auc']:.5f} | {m['auc']:.5f} | {m['final_rmse']:.5f} |")
 lines+=['','## Paired inference','', 'Effects below are LLM minus comparator. Negative favors the LLM. Seeds are averaged before inference; scaffolds, not trajectories, are the bootstrap clusters. The two primary comparisons use 97.5% intervals to account for two comparisons. Additional-control intervals are descriptive.','', '| Comparator | Paired normalized AUC difference | 97.5% cluster interval | Wins / ties / losses |','|---|---:|---|---|']
 for c in comparisons:
  lo,hi=c['scaffold_cluster_97.5_percent_ci'];lines.append(f"| {c['comparison'].split(' minus ')[1]} | {c['mean_paired_normalized_auc_difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {c['molecule_wins']} / {c['ties']} / {c['losses']} |")
 lines+=['','## Behavior and latency','',f"Strategies: `{json.dumps(audit['strategy_counts'],sort_keys=True)}`. Option letters: `{json.dumps(audit['letter_counts'],sort_keys=True)}`. The latter is an order-bias diagnostic, not a counterfactual prompt-order experiment.", '',f"Exactly 300 decisions took {audit['planning_seconds']:.1f} seconds of measured inference, median {audit['median_planning_seconds']:.2f} seconds per molecule. There were {audit['prompt_tokens']:,} prompt tokens and 300 output tokens. One exact shared prompt-prefix prefill is included; model loading, model download and data preparation are excluded. Individual decision timings exclude that shared prefill. Logical prompt tokens include cached prefixes. Inference ran locally on Apple M2 via MLX, with no new AWS resources or paid inference API. These timings are not a controlled speed comparison with the earlier AWS run.", '', '## Audits and limits','', 'The audit checks all 360 input dataset hashes, all prompt hashes and allowlisted targets, parent uniqueness, scaffold disjointness, identical initial sets and query budgets, pool-only selections, and exact replay against the fixed control for every LLM plan. Distances are used to reject exact labeled-geometry duplicates at 1e-6 angstrom resolution after rigid-motion invariance; symmetry-equivalent conformers are not exhaustively deduplicated. Only one eligible scan per nonstereo molecular parent is retained, deterministically by record ID. No energy-window filtering is applied.','', '“Held out” means excluded from this project’s development examples and decision tuning. Absence from LLM pretraining cannot be guaranteed. The QM archive includes Sage fitting data; this benchmark does not claim held-out Sage force-field accuracy, and deliberately uses UFF as the baseline. Scaffold groups use ring Murcko scaffolds, with generic full-graph topology for acyclic molecules; they do not guarantee low pairwise molecular similarity. Pilot identities and pilot ring scaffolds were excluded.','', 'The eight conformer tests come from the same constrained scan as the pool, so this tests interpolation across angles within a molecule, not generalization to new conformer families. The candidate geometries themselves were produced by QM constrained optimization; replay treats these geometries as available and therefore does not establish prospective QM wall-time savings. A prospective study must generate cheap candidate geometries and run QM only on acquired configurations. Complete one-dimensional scans bias chemical coverage. The finite menu, fixed Gaussian model and noise limit expressivity. In this linear-Gaussian design, acquisition depends on geometry and covariance rather than observed energy values. The LLM selects a chemical prior; it does not invent force-field coefficients or QM labels. A stronger prompt/model can still fail. Do not tune on these 300 test molecules and present a rerun as an independent held-out result.','', '## Sources and reproduction','', '- [Official OpenFF Sage 2.2.1 data and fitting repository](https://github.com/openforcefield/sage-2.2.1), pinned source commit `b70c449e3a98e40f0b6e4c6b2f27932fce47be42`.','- [Quantized model card and Apache 2.0 license](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit).','- [MLX-LM inference library](https://github.com/ml-explore/mlx-lm).','', 'The frozen protocol, importer, development selector, planner, evaluator, tests and analysis are in `research/acquisition`. The reproduction README lists commands. Dataset and source hashes, full selected inputs, raw requests, model decisions, per-seed curves and selection IDs are retained in the companion archive.','']
 (out/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps({k:v for k,v in audit.items() if k not in ['means']},indent=2))
 return audit

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--frozen',required=True);p.add_argument('--plans-dir',required=True);p.add_argument('--runs',required=True);p.add_argument('--out',required=True);a=p.parse_args();analyze(a.data,a.frozen,a.plans_dir,a.runs,a.out)
