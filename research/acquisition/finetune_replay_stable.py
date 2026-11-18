"""Open fresh QM labels only after adapter selection and all fresh decisions are saved."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from qm_benchmark import STRATEGIES,evaluate,molecule_metrics,rule,hashfile
from finetune_core import split_checks

def fit_regret_selector(inputs,teacher,train,parameters):
 x=np.array([inputs[n]['tabular_features'] for n in train])
 y=np.array([[teacher[n]['costs'][s]-teacher[n]['costs']['uniform'] for s in STRATEGIES] for n in train])
 estimator=ExtraTreesRegressor(**parameters);estimator.fit(x,y);return estimator

def replay(data,supervision,training,plans_dir,out):
 data=Path(data);supervision=Path(supervision);training=Path(training);plans_dir=Path(plans_dir);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 assert (plans_dir/'COMPLETE').exists()
 split=json.loads((data/'split.json').read_text());inputs=json.loads((supervision/'inputs.json').read_text());teacher=json.loads((supervision/'teacher.json').read_text());plans=json.loads((plans_dir/'plans.json').read_text())
 datasets={n:json.loads((data/'datasets'/f'{n}.json').read_text()) for names in split['splits'].values() for n in names};split_checks(datasets,split)
 assert set(plans)==set(split['splits']['test']) and set(teacher)==set(split['splits']['train']+split['splits']['validation'])
 selected=json.loads((training/'training_complete.json').read_text())
 if selected.get('status')=='technical_early_stop':raise RuntimeError('Early-stop diagnostic is sealed: complete BF16 repair before scoring fresh QM outcomes')
 assert json.loads((plans_dir/'manifest.json').read_text())['adapter_sha256']==selected['adapter_sha256']==hashfile(training/'adapters.safetensors')
 protocol=json.loads(Path(__file__).with_name('finetune_protocol_stable.json').read_text());train=split['splits']['train']
 estimator=fit_regret_selector(inputs,teacher,train,protocol['ExtraTrees'])
 global_strategy=json.loads((supervision/'manifest.json').read_text())['global_strategy'];allrows=[];allstats=[];predictions={}
 for name in split['splits']['test']:
  d=datasets[name];assert hashfile(data/'datasets'/f'{name}.json')==split['dataset_hashes'][name]
  predicted=estimator.predict(np.array(inputs[name]['tabular_features'])[None])[0];choice=STRATEGIES[int(np.argmin(predicted))]
  predictions[name]={'predicted_regret':dict(zip(STRATEGIES,predicted.tolist())),'strategy':choice}
  mapping={'bayes':'uniform','rule':rule(d['molecule']),'base_llm':plans[name]['base']['strategy'],'finetuned_llm':plans[name]['finetuned']['strategy'],
   'training_global':global_strategy,'extra_trees':choice,'random':'random'}
  mapping.update({f'fixed_{s}':s for s in STRATEGIES if s!='uniform'})
  rows,histories=evaluate(d,mapping);allrows+=rows;allstats+=molecule_metrics(rows);(out/f'{name}-selections.json').write_text(json.dumps(histories))
 with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
 (out/'molecule_metrics.json').write_text(json.dumps(allstats,indent=2));(out/'extra_trees.json').write_text(json.dumps({'parameters':protocol['ExtraTrees'],'feature_importances':estimator.feature_importances_.tolist(),'predictions':predictions},indent=2))
 manifest={'split_sha256':hashfile(data/'split.json'),'teacher_sha256':hashfile(supervision/'teacher.json'),'plans_sha256':hashfile(plans_dir/'plans.json'),
  'adapter_sha256':selected['adapter_sha256'],'selected_step':selected['selected_checkpoint']['step'],'source_sha256':hashfile(__file__),
  'policies':list(mapping),'test_molecules':len(plans),'seeds':list(range(5)),'queries':[4,6,8,10,12],'test_label_scoring_after_all_plans':True}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'COMPLETE').write_text('Fresh QM replay complete\n');print(json.dumps(manifest,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--supervision',required=True);p.add_argument('--training',required=True);p.add_argument('--plans-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args();replay(a.data,a.supervision,a.training,a.plans_dir,a.out)
