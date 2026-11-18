"""Generate teacher costs using train/validation QM only; materialize label-free inputs."""
import argparse,json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator,Descriptors
from qm_benchmark import STRATEGIES,evaluate,molecule_metrics,hashfile
from finetune_core import soft_targets,split_checks

def compact_input(d):
 mol=Chem.MolFromSmiles(d['molecule']['smiles']);tor=d['molecule']['torsions'][0]
 for a in mol.GetAtoms():
  if a.GetAtomMapNum() not in tor:a.SetAtomMapNum(0)
 params=Chem.RemoveHsParameters();params.removeMapped=False
 compact=Chem.RemoveHs(mol,params)
 c=d['molecule']['chemistry'];bond={k:c[k] for k in ['elements','bond_order','conjugated','amide_like','central_bond_in_ring','both_ends_aromatic','adjacent_aromatic','heavy_neighbor_counts','formal_charge']}
 a=np.array([r['angles_rad'][0] for r in d['pool']]);parent=Chem.MolFromSmiles(d['parent_smiles'])
 fp=np.asarray(rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=256).GetFingerprint(parent),dtype=float)
 numeric=[c['bond_order'],float(c['conjugated']),float(c['amide_like']),float(c['central_bond_in_ring']),float(c['both_ends_aromatic']),float(c['adjacent_aromatic']),*c['heavy_neighbor_counts'],c['formal_charge'],parent.GetNumHeavyAtoms(),parent.GetRingInfo().NumRings(),Descriptors.MolWt(parent)]
 numeric += [float(f(n*a).mean()) for n in [1,2,3] for f in [np.cos,np.sin]]
 numeric += [float(np.diff(np.r_[np.sort((a+np.pi)%(2*np.pi)),np.sort((a+np.pi)%(2*np.pi))[0]+2*np.pi]).max())]
 return {'graph':Chem.MolToSmiles(compact),'driven_dihedral':tor,'bond':bond,'pool_angles_degrees':np.round(np.degrees(a),1).tolist(),'tabular_features':np.r_[fp,numeric].tolist()}

def prepare(data,out):
 data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 split=json.loads((data/'split.json').read_text());datasets={n:json.loads((data/'datasets'/f'{n}.json').read_text()) for names in split['splits'].values() for n in names};split_checks(datasets,split)
 inputs={n:compact_input(d) for n,d in datasets.items()};(out/'inputs.json').write_text(json.dumps(inputs,indent=2));scores={};curves=[]
 for key in ['train','validation']:
  for name in split['splits'][key]:
   d=datasets[name];assert hashfile(data/'datasets'/f'{name}.json')==split['dataset_hashes'][name]
   rows,_=evaluate(d,{s:s for s in STRATEGIES});curves+=rows;stats=molecule_metrics(rows)
   costs=[next(r['normalized_auc'] for r in stats if r['strategy']==s) for s in STRATEGIES]
   scores[name]={'split':key,'costs':dict(zip(STRATEGIES,costs)),'soft_targets':dict(zip(STRATEGIES,soft_targets(costs).tolist())),'best_strategy':STRATEGIES[int(np.argmin(costs))]}
 assert set(scores).isdisjoint(split['splits']['test'])
 train_means={s:float(np.mean([scores[n]['costs'][s] for n in split['splits']['train']])) for s in STRATEGIES}
 (out/'teacher.json').write_text(json.dumps(scores,indent=2));(out/'teacher-curves.json').write_text(json.dumps(curves))
 manifest={'split_sha256':hashfile(data/'split.json'),'inputs_sha256':hashfile(out/'inputs.json'),'teacher_sha256':hashfile(out/'teacher.json'),'protocol_sha256':hashfile(Path(__file__).with_name('finetune_protocol.json')),'train_mean_costs':train_means,'global_strategy':min(train_means,key=train_means.get),'fresh_test_teacher_costs_generated':False}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(a.data,a.out)
