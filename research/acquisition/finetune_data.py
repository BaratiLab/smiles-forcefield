"""Freeze fine-tuning train/validation/fresh-test groups before reading acquisition outcomes."""
import argparse,json,shutil,urllib.request
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from qm_data import digest
from qm_benchmark import hashfile

def prepare(inventory,previous,out):
 inventory=Path(inventory);previous=Path(previous);out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'datasets').mkdir()
 old=json.loads((previous/'split.json').read_text())
 groups={k:{json.loads((previous/'datasets'/f'{n}.json').read_text())['scaffold'] for n in v} for k,v in old['splits'].items()}
 datasets=[json.loads(p.read_text()) for p in inventory.glob('datasets/*.json')]
 planned={'train':[],'validation':[],'test':[]};quarantine=[]
 for d in datasets:
  if d['scaffold'] in groups['heldout']:quarantine.append(d['molecule']['id']);continue
  if d['scaffold'] in groups['development']:key='train'
  else:key='validation' if int(digest('qlora-split-v1'+d['scaffold'])[:8],16)%5==0 else 'test'
  planned[key].append(d)
 for v in planned.values():v.sort(key=lambda d:digest('qlora-order-v1:'+d['parent_smiles']))
 names=[d['molecule']['id'] for v in planned.values() for d in v];ids=[int(n.split('-')[1]) for n in names];records=[]
 endpoint='https://api.qcarchive.molssi.org/api/v1/records/torsiondrive/bulkGet'
 for start in range(0,len(ids),60):
  batch=ids[start:start+60];req=urllib.request.Request(endpoint,data=json.dumps({'ids':batch,'missing_ok':False,'include':['*']}).encode(),headers={'Content-Type':'application/json','Accept':'application/json'})
  with urllib.request.urlopen(req,timeout=30) as r:chunk=json.load(r)
  if [r['id'] for r in chunk]!=batch:raise ValueError('Record order mismatch')
  records += [{'id':r['id'],'status':r['status'],'specification':r['specification']} for r in chunk]
 status={str(r['id']):r['status'] for r in records}
 for r in records:
  s=r['specification']['optimization_specification']['qc_specification']
  if s['program']!='psi4' or s['basis']!='dzvp' or s['method'] not in ['b3lyp-d3bj','b3lyp-d3(bj)']:raise ValueError('Unexpected QM method')
 splits={k:[d for d in v if status[d['metadata']['source_record_id']]=='complete'] for k,v in planned.items()}
 for a,b in [('train','validation'),('train','test'),('validation','test')]:
  assert not {d['scaffold'] for d in splits[a]}&{d['scaffold'] for d in splits[b]}
 assert not ({d['scaffold'] for d in splits['test']}|{d['scaffold'] for d in splits['validation']})&set.union(*groups.values())
 hashes={}
 for v in splits.values():
  for d in v:
   name=d['molecule']['id'];src=inventory/'datasets'/f'{name}.json';dst=out/'datasets'/src.name;shutil.copy2(src,dst);hashes[name]=hashfile(dst)
 manifest={'version':'qm-acquisition-qlora-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':hashfile(Path(__file__).with_name('finetune_protocol.json')),
  'inventory_manifest_sha256':hashfile(inventory/'split.json'),'previous_split_sha256':hashfile(previous/'split.json'),
  'splits':{k:[d['molecule']['id'] for d in v] for k,v in splits.items()},'planned_counts':{k:len(v) for k,v in planned.items()},
  'valid_counts':{k:len(v) for k,v in splits.items()},'scaffold_counts':{k:len({d['scaffold'] for d in v}) for k,v in splits.items()},
  'qc_excluded':{f'torsion-{i}':s for i,s in status.items() if s!='complete'},'qc_status_counts':dict(Counter(status.values())),
  'quarantined_previous_test_scaffold_molecules':len(quarantine),'previous_test_scaffolds':sorted(groups['heldout']),'all_previous_scaffolds':sorted(set.union(*groups.values())),
  'dataset_hashes':hashes}
 (out/'qcarchive-specifications.json').write_text(json.dumps(records,indent=2));(out/'split.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({k:v for k,v in manifest.items() if k not in ['splits','dataset_hashes','previous_test_scaffolds','all_previous_scaffolds']},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--previous',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(a.inventory,a.previous,a.out)
