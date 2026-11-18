"""Reproduce the explicitly amended checkpoint recovery; never hide a failed run."""
from pathlib import Path
import argparse
import json,hashlib,shutil,datetime
import numpy as np
from safetensors.numpy import load_file
p=argparse.ArgumentParser(description='Recover only the recorded pre-test numerical-stop amendment, without retraining')
p.add_argument('--training',required=True);p.add_argument('--failure-log',required=True);p.add_argument('--amendment',required=True);args=p.parse_args()
training=Path(args.training);failure=Path(args.failure_log)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (training/'training_complete.json').exists()
provenance=json.loads((training/'provenance.json').read_text());assert not provenance['smoke_only']
history=[json.loads(l) for l in (training/'training.jsonl').read_text().splitlines()];assert len(history)==83 and history[-1]['update']==83
assert 'ValueError: Nonfinite training' in failure.read_text()
validation=json.loads((training/'validation.json').read_text());candidates=[r for r in validation if r['step']>0];assert {r['step'] for r in candidates}=={24,48,72}
best=min(candidates,key=lambda r:(r['mean_validation_normalized_auc'],r['step']));assert best['step']==24
source=training/f"adapters-{best['step']:03d}.safetensors";weights=load_file(source);norm=float(np.sqrt(sum(np.square(v.astype(np.float64)).sum() for k,v in weights.items() if k.endswith('lora_b'))));assert np.isfinite(norm) and norm>0
shutil.copy2(source,training/'adapters.safetensors');shutil.copy2(Path(args.amendment),training/'training-amendment.json');shutil.copy2(failure,training/'training-failure.log')
start=(training/'provenance.json').stat().st_mtime;end=failure.stat().st_mtime
manifest={'status':'technical_early_stop','scheduled_updates':96,'trained_updates':83,'failed_update':84,'microsteps':415,'attempted_microsteps':420,
 'selected_checkpoint':best,'adapter_sha256':sha(training/'adapters.safetensors'),'validation_file_sha256':sha(training/'validation.json'),'training_history_sha256':sha(training/'training.jsonl'),
 'amendment_sha256':sha(training/'training-amendment.json'),'failure_log_sha256':sha(training/'training-failure.log'),'wall_seconds':end-start,'wall_time_estimated_from_file_timestamps':True,
 'start_timestamp_utc':datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),'end_timestamp_utc':datetime.datetime.fromtimestamp(end,datetime.timezone.utc).isoformat(),
 'nonzero_lora_b_norm':norm,'norm_scope':'selected checkpoint 24','unique_training_parents_used':len({r['training_molecule'] for r in history}),'selected_checkpoint_training_parents':24,'fresh_test_evaluated':False}
(training/'training_complete.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
