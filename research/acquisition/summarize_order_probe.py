"""Summarize retired-cohort semantic-order sensitivity and repeatability checks."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np

def summarize(probe,original_path,runs,out):
 probe=Path(probe);assert (probe/'COMPLETE').exists();rows=json.loads((probe/'decisions.json').read_text());original=json.loads(Path(original_path).read_text());stats=json.loads((Path(runs)/'molecule_metrics.json').read_text());lookup={(r['molecule'],r['policy']):r['normalized_auc'] for r in stats}
 names=sorted({r['molecule'] for r in rows});per=[];maximum=0.
 for name in names:
  r=sorted([v for v in rows if v['molecule']==name],key=lambda v:v['rotation']);assert [v['rotation'] for v in r]==list(range(5))
  req=json.loads((probe/name/'0-request.json').read_text());assert hashlib.sha256(req['prompt'].encode()).hexdigest()==original[name]['finetuned']['prompt_sha256']
  delta=max(abs(r[0]['logits'][s]-v) for s,v in original[name]['finetuned']['option_logits'].items());maximum=max(maximum,delta)
  errors=[lookup[name,'bayes' if v['strategy']=='uniform' else 'fixed_'+v['strategy']] for v in r]
  per.append({'molecule':name,'unique_semantic_actions':len({v['strategy'] for v in r}),'rotation0_repeats_original':r[0]['strategy']==original[name]['finetuned']['strategy'],'evaluation_auc_range':max(errors)-min(errors),'actions':[v['strategy'] for v in r]})
 summary={'kind':'posthoc causal prompt-order intervention on 20 retired molecules; not an independent performance test','molecules':len(names),'calls':len(rows),'changed_semantic_action_under_rotation':sum(r['unique_semantic_actions']>1 for r in per),'rotation0_exact_action_repeats':sum(r['rotation0_repeats_original'] for r in per),'rotation0_max_logit_absolute_difference':maximum,'unique_action_counts':dict(Counter(r['unique_semantic_actions'] for r in per)),'mean_within_molecule_normalized_auc_range':float(np.mean([r['evaluation_auc_range'] for r in per])),'letters':dict(Counter(r['letter'] for r in rows)),'seconds':sum(r['seconds'] for r in rows),'per_molecule':per}
 Path(out).write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='per_molecule'},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--probe',required=True);p.add_argument('--original',required=True);p.add_argument('--runs',required=True);p.add_argument('--out',required=True);a=p.parse_args();summarize(a.probe,a.original,a.runs,a.out)
