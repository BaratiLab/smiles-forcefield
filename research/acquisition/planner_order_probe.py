"""Post hoc semantic option-order stress test on a retired test cohort. No training."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from finetune_mlx_stable import setup,encode,option_logits
from qm_benchmark import hashfile

def probe(data,inputs_path,training,original_plans,model,out,n=20):
 import mlx.core as mx
 data=Path(data);training=Path(training);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 split=json.loads((data/'split.json').read_text());inputs=json.loads(Path(inputs_path).read_text());original=json.loads(Path(original_plans).read_text())
 names=sorted(split['splits']['test'],key=lambda s:hashlib.sha256(('posthoc-order-v1:'+s).encode()).hexdigest())[:n]
 model_obj,tok,_=setup(model,20261118);model_obj.load_weights(str(training/'adapters.safetensors'),strict=False);model_obj.eval()
 tokens=[tok.encode(chr(65+i),add_special_tokens=False) for i in range(5)];assert all(len(x)==1 for x in tokens);tokens=[x[0] for x in tokens]
 manifest={'kind':'posthoc diagnostic, retired test set; not independent confirmation','subset_rule':'first n by SHA256(posthoc-order-v1:record_id), outcome-independent','n':n,'names':names,'adapter_sha256':hashfile(training/'adapters.safetensors'),'original_plans_sha256':hashfile(original_plans),'source_sha256':hashfile(__file__),'new_training':False,'rotations_per_molecule':5}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));rows=[]
 for i,name in enumerate(names,1):
  folder=out/name;folder.mkdir()
  for rotation in range(5):
   ids,order,prompt,messages=encode(tok,inputs[name],rotation);start=time.monotonic();scores=option_logits(model_obj,ids,tokens);mx.eval(scores);values=np.array(scores.tolist());j=int(np.argmax(values))
   if not np.isfinite(values).all():raise ValueError('Nonfinite probe scores')
   row={'molecule':name,'rotation':rotation,'strategy':order[j],'letter':chr(65+j),'order':order,'logits':dict(zip(order,values.tolist())),'seconds':time.monotonic()-start,'original_strategy':original[name]['finetuned']['strategy']}
   rows.append(row);(folder/f'{rotation}-request.json').write_text(json.dumps({'messages':messages,'prompt':prompt},indent=2));(out/'decisions.json').write_text(json.dumps(rows,indent=2));mx.clear_cache()
  print(f'{i}/{len(names)} {name} '+str([r['strategy'] for r in rows[-5:]]),flush=True)
 (out/'COMPLETE').write_text('Posthoc order probe complete\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--inputs',required=True);p.add_argument('--training',required=True);p.add_argument('--original-plans',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=20);a=p.parse_args();probe(a.data,a.inputs,a.training,a.original_plans,a.model,a.out,a.n)
