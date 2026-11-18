"""Cost-sensitive QLoRA attention adapters; fresh test inputs are never trained on."""
import argparse,hashlib,importlib.metadata,json,time
from pathlib import Path
import numpy as np
from finetune_core import messages
from qm_benchmark import STRATEGIES,hashfile

def setup(model_path,seed):
 import mlx.core as mx
 from mlx_lm import load
 from mlx_lm.tuner.utils import linear_to_lora_layers
 mx.random.seed(seed);model,tok=load(model_path,tokenizer_config={'trust_remote_code':False})
 model.freeze();config=json.loads(Path(__file__).with_name('finetune_protocol.json').read_text())['adapter']
 linear_to_lora_layers(model,config['num_layers'],{k:v for k,v in config.items() if k!='num_layers'})
 return model,tok,config

def encode(tok,item,rotation=0):
 msg,order=messages(item,rotation);prompt=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
 return tok.encode(prompt,add_special_tokens=False),order,prompt,msg

def option_logits(model,ids,letter_tokens):
 import mlx.core as mx
 # Evaluate the vocabulary projection only at the last position; same next-token logits.
 hidden=model.model(mx.array(ids)[None])[:,-1:,:]
 return model.lm_head(hidden)[0,0,mx.array(letter_tokens)].astype(mx.float32)

def decide(model,tok,item,letter_tokens):
 import mlx.core as mx
 ids,order,prompt,msg=encode(tok,item);start=time.monotonic();logits=option_logits(model,ids,letter_tokens);mx.eval(logits)
 raw=np.array(logits.tolist());prob=np.exp(raw-raw.max());prob/=prob.sum();index=int(np.argmax(raw))
 if not np.isfinite(raw).all():raise ValueError('Nonfinite planner logits')
 return {'strategy':order[index],'chosen_letter':chr(65+index),'option_logits':dict(zip(order,raw.tolist())),'option_probabilities':dict(zip(order,prob.tolist())),
  'order':order,'seconds':time.monotonic()-start,'prompt_tokens':len(ids),'calls':1,'input_sha256':hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest(),
  'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}, {'messages':msg,'prompt':prompt}

def train(data,supervision,model_path,out,smoke=False):
 import mlx.core as mx
 import mlx.nn as nn
 import mlx.optimizers as optim
 from mlx.utils import tree_flatten,tree_map
 data=Path(data);supervision=Path(supervision);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 protocol=json.loads(Path(__file__).with_name('finetune_protocol.json').read_text());split=json.loads((data/'split.json').read_text());teacher=json.loads((supervision/'teacher.json').read_text());inputs=json.loads((supervision/'inputs.json').read_text())
 assert set(teacher)==set(split['splits']['train']+split['splits']['validation']) and not set(teacher)&set(split['splits']['test'])
 assert json.loads((supervision/'manifest.json').read_text())['split_sha256']==hashfile(data/'split.json')
 model,tok,config=setup(model_path,protocol['seed']);flat=tree_flatten(model.trainable_parameters())
 assert flat and all(n.rsplit('.',1)[-1] in ['lora_a','lora_b'] for n,v in flat)
 nparams=sum(v.size for n,v in flat)
 tokens=[tok.encode(chr(65+i),add_special_tokens=False) for i in range(5)];assert all(len(v)==1 for v in tokens);tokens=[v[0] for v in tokens]
 weights={p.name:hashfile(p) for p in Path(model_path).glob('*') if p.is_file() and p.suffix in ['.safetensors','.json']}
 provenance={'protocol_sha256':hashfile(Path(__file__).with_name('finetune_protocol.json')),'split_sha256':hashfile(data/'split.json'),
  'supervision_manifest_sha256':hashfile(supervision/'manifest.json'),'teacher_sha256':hashfile(supervision/'teacher.json'),'model_files_sha256':weights,
  'source_sha256':hashfile(__file__),'prompt_source_sha256':hashfile(Path(__file__).with_name('finetune_core.py')),'trainable_parameters':nparams,'trainable_names':[n for n,v in flat],
  'mlx':importlib.metadata.version('mlx'),'mlx_lm':importlib.metadata.version('mlx-lm'),'device':mx.device_info(),'smoke_only':smoke}
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2));(out/'adapter_config.json').write_text(json.dumps({'fine_tune_type':'lora','num_layers':config['num_layers'],'lora_parameters':{k:v for k,v in config.items() if k!='num_layers'}},indent=2))
 optimizer=optim.AdamW(learning_rate=protocol['optimizer']['learning_rate'],weight_decay=protocol['optimizer']['weight_decay'])
 def loss_fn(model,ids,target):
  logits=option_logits(model,ids,tokens);return -(target*(logits-mx.logsumexp(logits))).sum()
 value_grad=nn.value_and_grad(model,loss_fn)
 history=[];validation=[];started=time.monotonic()
 def validate(step):
  model.eval();plans={};costs=[];losses=[];folder=out/'validation'/f'step-{step:03d}';folder.mkdir(parents=True)
  names=split['splits']['validation'][:1] if smoke else split['splits']['validation']
  for name in names:
   plan,request=decide(model,tok,inputs[name],tokens);plans[name]=plan;costs.append(teacher[name]['costs'][plan['strategy']]);(folder/f'{name}-request.json').write_text(json.dumps(request,indent=2))
   target=np.array([teacher[name]['soft_targets'][s] for s in plan['order']]);p=np.array([plan['option_probabilities'][s] for s in plan['order']]);losses.append(float(-(target*np.log(np.maximum(p,1e-30))).sum()))
   mx.clear_cache()
  (folder/'plans.json').write_text(json.dumps(plans,indent=2));record={'step':step,'mean_validation_normalized_auc':float(np.mean(costs)),'mean_validation_soft_ce':float(np.mean(losses))};validation.append(record);(out/'validation.json').write_text(json.dumps(validation,indent=2));print('VALIDATION '+json.dumps(record),flush=True)
 validate(0)
 rng=np.random.default_rng(protocol['seed']);names=split['splits']['train'];schedule=[]
 while len(schedule)<protocol['updates']:schedule.extend(rng.permutation(names).tolist())
 updates=1 if smoke else protocol['updates']
 for step,name in enumerate(schedule[:updates],1):
  model.train();accum=None;losses=[];t=time.monotonic()
  for rotation in range(5):
   ids,order,prompt,msg=encode(tok,inputs[name],rotation)
   target=mx.array([teacher[name]['soft_targets'][s] for s in order],dtype=mx.float32)
   loss,grads=value_grad(model,ids,target);mx.eval(loss,grads)
   accum=grads if accum is None else tree_map(lambda a,b:a+b,accum,grads)
   mx.eval(accum);losses.append(float(loss.item()));mx.clear_cache()
  grads=tree_map(lambda v:v/5,accum);grads,norm=optim.clip_grad_norm(grads,protocol['optimizer']['gradient_clip_norm']);optimizer.update(model,grads);mx.eval(model.trainable_parameters(),optimizer.state,norm)
  record={'update':step,'training_molecule':name,'soft_ce':float(np.mean(losses)),'gradient_norm':float(norm.item()),'seconds':time.monotonic()-t,'rotations':5}
  if not np.isfinite([record['soft_ce'],record['gradient_norm']]).all():raise ValueError('Nonfinite training')
  history.append(record)
  with (out/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print('TRAIN '+json.dumps(record),flush=True)
  if step in protocol['validation_checkpoints'] or smoke:
   path=out/f'adapters-{step:03d}.safetensors';mx.save_safetensors(str(path),dict(tree_flatten(model.trainable_parameters())));validate(step)
 candidates=[r for r in validation if r['step']>0];best=min(candidates,key=lambda r:(r['mean_validation_normalized_auc'],r['step']))
 selected=out/f"adapters-{best['step']:03d}.safetensors";import shutil;shutil.copy2(selected,out/'adapters.safetensors')
 norm=float(sum(float((v*v).sum().item()) for n,v in tree_flatten(model.trainable_parameters()) if n.endswith('lora_b'))**.5)
 assert norm>0,'No actual adapter update'
 manifest={'selected_checkpoint':best,'adapter_sha256':hashfile(out/'adapters.safetensors'),'validation_file_sha256':hashfile(out/'validation.json'),'training_history_sha256':hashfile(out/'training.jsonl'),'wall_seconds':time.monotonic()-started,'trained_updates':updates,'microsteps':updates*5,'nonzero_lora_b_norm':norm,'fresh_test_evaluated':False}
 (out/'training_complete.json').write_text(json.dumps(manifest,indent=2));print('COMPLETE '+json.dumps(manifest),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--supervision',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();train(a.data,a.supervision,a.model,a.out,a.smoke)
