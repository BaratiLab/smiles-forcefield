"""Independently verify published QM specifications, geometry order and split similarity."""
import argparse,json,tarfile,urllib.request
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator

def verify(data,archive,out):
 data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 split=json.loads((data/'split.json').read_text());names=sum(split['splits'].values(),[]);ids=[int(n.split('-')[1]) for n in names]
 endpoint='https://api.qcarchive.molssi.org/api/v1/records/torsiondrive/bulkGet';records=[]
 for start in range(0,len(ids),60):
  batch=ids[start:start+60]
  req=urllib.request.Request(endpoint,data=json.dumps({'ids':batch,'missing_ok':False,'include':['*']}).encode(),headers={'Content-Type':'application/json','Accept':'application/json'})
  with urllib.request.urlopen(req,timeout=30) as response:chunk=json.load(response)
  if [r['id'] for r in chunk]!=batch:raise ValueError('QCArchive record ordering mismatch')
  records += [{'id':r['id'],'status':r['status'],'specification':r['specification']} for r in chunk]
 (out/'qcarchive-specifications.json').write_text(json.dumps(records,indent=2))
 methods=Counter()
 for record in records:
  spec=record['specification']['optimization_specification']['qc_specification']
  if spec['program']!='psi4' or spec['basis']!='dzvp' or spec['method'] not in ['b3lyp-d3bj','b3lyp-d3(bj)']:raise ValueError('QM method differs from protocol')
  methods[spec['method']]+=1
 status={'endpoint':endpoint,'fetched_at_utc':datetime.now(timezone.utc).isoformat(),'records':len(records),'method_alias_counts':dict(methods),'status_counts':dict(Counter(r['status'] for r in records)),'noncomplete_ids':[r['id'] for r in records if r['status']!='complete']}
 (out/'qc-status.json').write_text(json.dumps(status,indent=2))
 # Do not execute archive contents or use unrestricted extraction.
 wanted=set(names);files={}
 with tarfile.open(archive,'r|gz') as tf:
  for member in tf:
   parts=Path(member.name).parts
   if member.isfile() and len(parts)==3 and parts[0]=='targets' and parts[1] in wanted and parts[2] in ['qdata.txt','scan.xyz']:files[member.name]=tf.extractfile(member).read().decode()
 maximum=0.;frames=0
 for name in names:
  q=files[f'targets/{name}/qdata.txt'];coords=[np.array([float(v) for v in line.split()[1:]]).reshape(-1,3) for line in q.splitlines() if line.startswith('COORDS ')]
  lines=files[f'targets/{name}/scan.xyz'].splitlines();xyz=[];i=0
  while i<len(lines):
   if not lines[i].strip():i+=1;continue
   n=int(lines[i]);xyz.append(np.array([[float(x) for x in line.split()[1:4]] for line in lines[i+2:i+2+n]]));i+=n+2
  if len(coords)!=len(xyz) or len(coords)!=24:raise ValueError('Unexpected geometry count')
  for c,f in zip(coords,xyz):
   err=float(abs(c-f).max())
   if err>=1e-6:raise ValueError('QM coordinate/unit/order mismatch')
   maximum=max(maximum,err);frames+=1
 (out/'geometry-audit.json').write_text(json.dumps({'molecules':len(names),'frames':frames,'qdata_xyz_order_and_coordinates_match':True,'max_coordinate_difference_angstrom':maximum},indent=2))
 gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);fps={}
 for key in ['development','heldout']:
  fps[key]=[gen.GetFingerprint(Chem.MolFromSmiles(json.loads((data/'datasets'/f'{m}.json').read_text())['parent_smiles'])) for m in split['splits'][key]]
 nearest=[max(DataStructs.BulkTanimotoSimilarity(f,fps['development'])) for f in fps['heldout']]
 (out/'chemical-similarity-audit.json').write_text(json.dumps({'Morgan_radius':2,'fingerprint_bits':2048,'nearest_development_Tanimoto_median':float(np.median(nearest)),'nearest_development_Tanimoto_p95':float(np.quantile(nearest,.95)),'nearest_development_Tanimoto_max':max(nearest),'scaffold_split_does_not_mean_similarity_free':True},indent=2))
 print(json.dumps(status,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--archive',required=True);p.add_argument('--out',required=True);a=p.parse_args();verify(a.data,a.archive,a.out)
