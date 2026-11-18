"""Local one-token constrained LLM planner. No remote code or model server needed."""
import argparse,hashlib,importlib.metadata,json,time
from pathlib import Path
import numpy as np
from qm_benchmark import planner_input,build_messages,hashfile

def infer(data,frozen_path,model_path,out,limit=None):
 import mlx.core as mx
 from mlx_lm import load
 from mlx_lm.generate import generate_step
 data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 split=json.loads((data/'split.json').read_text());frozen=json.loads(Path(frozen_path).read_text())
 assert frozen['split_sha256']==hashfile(data/'split.json')
 model,tokenizer=load(model_path,tokenizer_config={'trust_remote_code':False})
 modelhash={p.name:hashfile(p) for p in Path(model_path).glob('*') if p.is_file() and p.suffix in ['.json','.safetensors']}
 provenance={'model_path':str(model_path),'model_files_sha256':modelhash,'mlx_lm':importlib.metadata.version('mlx-lm'),'mlx':importlib.metadata.version('mlx'),
  'device':mx.device_info(),'split_sha256':hashfile(data/'split.json'),'frozen_planner_sha256':hashfile(frozen_path),
  'planner_source_sha256':hashfile(__file__),'prompt_source_sha256':hashfile(Path(__file__).with_name('qm_benchmark.py')),
  'decision':'argmax next-token log probability restricted to A B C D E; one forward-generation call per molecule, max_tokens=1; no rationale or retries'}
 provpath=out/'provenance.json'
 if provpath.exists():assert json.loads(provpath.read_text())==provenance,'Cannot resume changed model or experiment'
 else:provpath.write_text(json.dumps(provenance,indent=2))
 names=split['splits']['heldout'];names=names[:limit] if limit else names
 plans={}
 letters=['A','B','C','D','E'];tokens=[tokenizer.encode(s,add_special_tokens=False) for s in letters]
 assert all(len(t)==1 for t in tokens)
 for idx,name in enumerate(names):
  path=data/'datasets'/f'{name}.json';assert hashfile(path)==split['dataset_hashes'][name]
  d=json.loads(path.read_text());item=planner_input(d);itemhash=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
  folder=out/name;folder.mkdir(exist_ok=True);planpath=folder/'plan.json'
  messages,order=build_messages(item,frozen)
  prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  phash=hashlib.sha256(prompt.encode()).hexdigest()
  if planpath.exists():
   plan=json.loads(planpath.read_text());assert plan['input_sha256']==itemhash and plan['prompt_sha256']==phash
  else:
   (folder/'request.json').write_text(json.dumps({'messages':messages,'prompt':prompt,'strategy_order':order,'allowed_tokens':tokens},indent=2))
   ids=tokenizer.encode(prompt,add_special_tokens=False)
   started=time.monotonic()
   # Constrain only decoding; log all five unrestricted token log probabilities too.
   def constrained(logits):
    allowed=mx.array([t[0] for t in tokens]);scores=logits[:,allowed];return allowed[mx.argmax(scores,axis=-1)]
   gen=generate_step(mx.array(ids),model,max_tokens=1,sampler=constrained,prefill_step_size=512)
   token,logprobs=next(gen);mx.eval(logprobs);elapsed=time.monotonic()-started
   raw=logprobs.tolist();chosen=letters.index(tokenizer.decode([int(token)]));scores=[float(raw[t[0]]) for t in tokens]
   probabilities=np.exp(np.array(scores)-max(scores));probabilities/=probabilities.sum()
   plan={'strategy':order[chosen],'chosen_letter':letters[chosen],'input_sha256':itemhash,'prompt_sha256':phash,
    'seconds':elapsed,'prompt_tokens':len(ids),'output_tokens':1,'calls':1,'strategy_order':order,
    'option_log_probabilities':dict(zip(order,scores)),'conditional_option_probabilities':dict(zip(order,probabilities.tolist()))}
   planpath.write_text(json.dumps(plan,indent=2));del gen;mx.clear_cache()
  plans[name]=plan;(out/'plans.json').write_text(json.dumps(plans,indent=2))
  print(f'{idx+1}/{len(names)} {name} {plan["strategy"]} {plan["seconds"]:.2f}s',flush=True)
 if len(plans)==len(split['splits']['heldout']):(out/'COMPLETE').write_text('One LLM decision per heldout molecule\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--frozen',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int);a=p.parse_args();infer(a.data,a.frozen,a.model,a.out,a.limit)
