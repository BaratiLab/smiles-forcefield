"""One matched base/adapted decision per fresh molecule, after checkpoint selection."""
import argparse,json
from pathlib import Path
from qm_benchmark import hashfile
from finetune_mlx import setup,decide

def infer(data,supervision,training,model_path,out):
 import mlx.core as mx
 from mlx.utils import tree_flatten
 data=Path(data);supervision=Path(supervision);training=Path(training);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 split=json.loads((data/'split.json').read_text());inputs=json.loads((supervision/'inputs.json').read_text());finished=json.loads((training/'training_complete.json').read_text());provenance=json.loads((training/'provenance.json').read_text())
 assert not provenance['smoke_only']
 if finished.get('status')=='technical_early_stop':
  amendment=json.loads((training/'training-amendment.json').read_text());assert finished['amendment_sha256']==hashfile(training/'training-amendment.json')
  assert amendment['completed_finite_updates']==finished['trained_updates']==83 and finished['selected_checkpoint']['step']==24
 else:assert finished['trained_updates']==96
 assert provenance['split_sha256']==hashfile(data/'split.json') and provenance['teacher_sha256']==hashfile(supervision/'teacher.json')
 assert provenance['protocol_sha256']==hashfile(Path(__file__).with_name('finetune_protocol.json'))
 checkpoints=[v for v in json.loads((training/'validation.json').read_text()) if v['step']>0]
 assert min(checkpoints,key=lambda v:(v['mean_validation_normalized_auc'],v['step']))==finished['selected_checkpoint']
 assert finished['adapter_sha256']==hashfile(training/'adapters.safetensors')
 model,tok,_=setup(model_path,20261118);model.eval();initial=dict(tree_flatten(model.trainable_parameters()));adapted=mx.load(str(training/'adapters.safetensors'))
 assert set(initial)==set(adapted) and all(initial[k].shape==adapted[k].shape for k in initial)
 assert any(bool(mx.any(adapted[k]!=0).item()) for k in adapted if k.endswith('lora_b'))
 assert provenance['model_files_sha256']['model.safetensors']==hashfile(Path(model_path)/'model.safetensors')
 tokens=[tok.encode(chr(65+i),add_special_tokens=False) for i in range(5)];assert all(len(v)==1 for v in tokens);tokens=[v[0] for v in tokens]
 manifest={'split_sha256':hashfile(data/'split.json'),'inputs_sha256':hashfile(supervision/'inputs.json'),'training_complete_sha256':hashfile(training/'training_complete.json'),
  'adapter_sha256':finished['adapter_sha256'],'selected_step':finished['selected_checkpoint']['step'],'source_sha256':hashfile(__file__),'prompt_source_sha256':hashfile(Path(__file__).with_name('finetune_core.py')),'fresh_test_labels_used_for_planning':False}
 mp=out/'manifest.json'
 if mp.exists():assert json.loads(mp.read_text())==manifest
 else:mp.write_text(json.dumps(manifest,indent=2))
 plans={}
 for i,name in enumerate(split['splits']['test'],1):
  folder=out/name;folder.mkdir(exist_ok=True);path=folder/'plans.json'
  if path.exists():record=json.loads(path.read_text())
  else:
   record={}
   for key,weights in [('base',initial),('finetuned',adapted)]:
    model.load_weights(list(weights.items()),strict=False);mx.eval(model.trainable_parameters())
    if i==1:
     loaded=dict(tree_flatten(model.trainable_parameters()));assert all(bool(mx.array_equal(loaded[k],weights[k]).item()) for k in weights)
    plan,request=decide(model,tok,inputs[name],tokens);record[key]=plan;(folder/f'{key}-request.json').write_text(json.dumps(request,indent=2));mx.clear_cache()
   assert record['base']['prompt_sha256']==record['finetuned']['prompt_sha256']
   path.write_text(json.dumps(record,indent=2))
  plans[name]=record;(out/'plans.json').write_text(json.dumps(plans,indent=2));print(f'{i}/{len(split["splits"]["test"])} {name} base={record["base"]["strategy"]} fine={record["finetuned"]["strategy"]}',flush=True)
 (out/'COMPLETE').write_text('Fresh plans complete; no fresh QM outcomes scored yet\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--supervision',required=True);p.add_argument('--training',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args();infer(a.data,a.supervision,a.training,a.model,a.out)
