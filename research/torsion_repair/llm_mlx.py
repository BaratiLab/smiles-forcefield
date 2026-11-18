"""Local, untuned structured proposer. No external API or hidden outcome access."""
import argparse, hashlib, json, time
from pathlib import Path

SYSTEM='''You propose bounded torsion-repair hypotheses from verified chemistry and acquired QM residuals. Do not predict coefficients. Return ONLY a JSON object {"programs":[...]} with at most TWO programs. Each program has EXACT keys action, scope, harmonics. action is revise, specialize, keep, or abstain. scope is broad unless action=specialize, then local. harmonics is a sorted unique integer list in 1..6. keep/abstain use [1,2,3]. broad/local mean the verified environment scopes in state. Broad is the shared default prior; local specializes the chemical environment prior. Unsupported local priors are zero. Harmonics are hypotheses, not exact symmetry claims. Tools fit parameters and support-only validation can reject your proposals. Never change molecular state. No A-E labels. No rationale text.'''

def infer(out,model_path,max_tokens):
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    out=Path(out);states=json.loads((out/'states.json').read_text());path=out/'llm-plans.json'
    plans=json.loads(path.read_text()) if path.exists() else {}
    model,tok=load(model_path,tokenizer_config={'trust_remote_code':False});model.set_dtype(mx.bfloat16);model.eval()
    manifest={'model_path':str(Path(model_path).resolve()),'fine_tuned_for_repair':False,'adapter':'none; old angle-classification adapter is not a repair model','dtype':'bfloat16','temperature':0,'max_tokens':max_tokens,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'states_sha256':hashlib.sha256((out/'states.json').read_bytes()).hexdigest(),'model_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(model_path).glob('*.safetensors')}}
    mp=out/'llm-manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Resume provenance mismatch')
    mp.write_text(json.dumps(manifest,indent=2));(out/'requests').mkdir(exist_ok=True)
    for i,(n,state) in enumerate(states.items(),1):
        if n in plans:continue
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(state,sort_keys=True)}]
        prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);started=time.monotonic()
        raw=generate(model,tok,prompt=prompt,max_tokens=max_tokens,sampler=make_sampler(temp=0),verbose=False)
        error=None;programs=[]
        try:
            text=raw.strip()
            if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
            parsed=json.loads(text)
            if set(parsed)!={'programs'} or not isinstance(parsed['programs'],list) or len(parsed['programs'])>2:raise ValueError('Invalid proposal envelope')
            programs=parsed['programs']
        except (ValueError,TypeError) as e:error=str(e)
        record={'programs':programs,'raw_response':raw,'parse_error':error,'seconds':time.monotonic()-started,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
        plans[n]=record;(out/'requests'/f'{n}.json').write_text(json.dumps({'messages':messages,'prompt':prompt},indent=2));path.write_text(json.dumps(plans,indent=2));mx.clear_cache()
        print(f'{i}/{len(states)} {n} {record["seconds"]:.1f}s {programs} error={error}',flush=True)
    (out/'LLM_COMPLETE').write_text('Development proposals complete. No evaluator labels were passed to the proposer.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--model',required=True);p.add_argument('--max-tokens',type=int,default=180);a=p.parse_args();infer(a.out,a.model,a.max_tokens)
