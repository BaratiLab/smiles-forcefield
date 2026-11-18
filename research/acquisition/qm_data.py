"""Import the published OpenFF ForceBalance QM scans, without executing archive content."""
import argparse, hashlib, io, json, tarfile
from collections import Counter
from pathlib import Path
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem.Scaffolds import MurckoScaffold

SOURCE_COMMIT='b70c449e3a98e40f0b6e4c6b2f27932fce47be42'
SOURCE_URL=f'https://raw.githubusercontent.com/openforcefield/sage-2.2.1/{SOURCE_COMMIT}/04_fit-forcefield/nor4/fb-fit/targets.tar.gz'
HARTREE_TO_KCAL=627.5094740631

def digest(value):return hashlib.sha256(value.encode()).hexdigest()
def identity(m):
 m=Chem.RemoveHs(Chem.Mol(m))
 for atom in m.GetAtoms():atom.SetAtomMapNum(0)
 Chem.RemoveStereochemistry(m)
 return Chem.MolToSmiles(m,canonical=True)
def scaffold(m):
 m=Chem.RemoveHs(Chem.Mol(m));Chem.RemoveStereochemistry(m)
 for atom in m.GetAtoms():atom.SetAtomMapNum(0)
 sc=MurckoScaffold.GetScaffoldForMol(m)
 # Empty Murcko scaffolds otherwise group every acyclic molecule together.
 if sc.GetNumAtoms()==0:return 'acyclic:'+Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(m))
 return 'murcko:'+Chem.MolToSmiles(sc)

def chemistry(m,torsion):
 a,b,c,d=torsion;bond=m.GetBondBetweenAtoms(b,c)
 amides=m.GetSubstructMatches(Chem.MolFromSmarts('[C,S](=[O,S])-[N]'))
 amide=any({b,c}=={v[0],v[2]} for v in amides)
 ends=[m.GetAtomWithIdx(i) for i in (b,c)]
 return {'central_bond_atoms':[b+1,c+1], 'elements':[v.GetSymbol() for v in ends],
  'bond_order':bond.GetBondTypeAsDouble(), 'conjugated':bond.GetIsConjugated(),
  'amide_like':amide,'central_bond_in_ring':bond.IsInRing(),
  'both_ends_aromatic':all(v.GetIsAromatic() for v in ends),
  'adjacent_aromatic':any(v.GetIsAromatic() for v in ends),
  'heavy_neighbor_counts':[sum(n.GetAtomicNum()>1 for n in v.GetNeighbors()) for v in ends],
  'formal_charge':Chem.GetFormalCharge(m)}

def prepare(archive,out,dev_n=60,test_n=300):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'datasets').mkdir()
 prior=json.loads((Path(__file__).parent/'molecules.json').read_text())
 excluded_ids={identity(Chem.MolFromSmiles(m['smiles'])) for m in prior}
 # Keep pilot ring scaffolds out of both development and test.
 pilot_mols=[Chem.MolFromSmiles(m['smiles']) for m in prior]
 excluded_scaffolds={scaffold(m) for m in pilot_mols if m.GetRingInfo().NumRings()}
 rejects=Counter();datasets=[];seen=set()
 with tarfile.open(archive,mode='r|gz') as tf:
  payloads={x.name:tf.extractfile(x).read() for x in tf if x.isfile() and '/torsion-' in x.name and x.name.rsplit('/',1)[-1] in {'metadata.json','input.sdf','scan.xyz','qdata.txt'}}
  names=sorted(n.rsplit('/',1)[0] for n in payloads if n.endswith('/metadata.json'))
  for prefix in names:
   try:
    read=lambda name:payloads[prefix+'/'+name]
    meta=json.loads(read('metadata.json'))
    if len(meta['dihedrals'])!=1:rejects['not_1d']+=1;continue
    mol=next(iter(Chem.ForwardSDMolSupplier(io.BytesIO(read('input.sdf')),removeHs=False)))
    if mol is None:raise ValueError('Invalid SDF')
    parent=identity(mol);sc=scaffold(mol)
    if parent in excluded_ids or sc in excluded_scaffolds:rejects['pilot_identity_or_ring_scaffold']+=1;continue
    if parent in seen:rejects['repeated_parent']+=1;continue
    if not AllChem.UFFHasAllMoleculeParams(mol):rejects['uff_unsupported']+=1;continue
    energies=[float(line.split()[1]) for line in read('qdata.txt').decode().splitlines() if line.startswith('ENERGY ')]
    lines=read('scan.xyz').decode().splitlines();coords=[];symbols=[];pos=0
    while pos<len(lines):
     if not lines[pos].strip():pos+=1;continue
     n=int(lines[pos]);atomlines=[line.split() for line in lines[pos+2:pos+2+n]]
     assert n==mol.GetNumAtoms()
     assert [a[0] for a in atomlines]==[a.GetSymbol() for a in mol.GetAtoms()]
     coords.append(np.array([[float(x) for x in a[1:4]] for a in atomlines]));pos+=n+2
    if len(coords)!=24 or len(energies)!=24 or len(meta['torsion_grid_ids'])!=24:rejects['incomplete_24_point_grid']+=1;continue
    tor=meta['dihedrals'][0];records=[];conf=mol.GetConformer()
    for i,(xyz,e,grid) in enumerate(zip(coords,energies,meta['torsion_grid_ids'])):
     for a,p in enumerate(xyz):conf.SetAtomPosition(a,p.tolist())
     angle=rdMolTransforms.GetDihedralRad(conf,*tor)
     delta=(np.degrees(angle)-grid[0]+180)%360-180
     if abs(delta)>2:raise ValueError('Geometry/grid mismatch')
     ff=AllChem.UFFGetMoleculeForceField(mol,vdwThresh=100.,ignoreInterfragInteractions=False)
     baseline=ff.CalcEnergy()
     if not np.isfinite([e,baseline,angle]).all():raise ValueError('Nonfinite')
     # Rigid-motion invariant labeled distance fingerprint; does not identify symmetry permutations.
     dist=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1)
     records.append({'id':f'{prefix.rsplit("/",1)[1]}-grid-{i:02d}','angles_rad':[angle],
      'qm_energy_kcal_mol':e*HARTREE_TO_KCAL,'baseline_energy_kcal_mol':baseline,
      'geometry_sha256':hashlib.sha256(np.round(dist,6).tobytes()).hexdigest(),'source_grid_degrees':grid})
    # 8 random geometries held out, fixed before policy evaluation; not selected by energies.
    order=np.random.default_rng(int(digest(parent)[:8],16)).permutation(24)
    test=set(order[:8]);pool=[v for i,v in enumerate(records) if i not in test];held=[v for i,v in enumerate(records) if i in test]
    if len({v['geometry_sha256'] for v in records})!=24:rejects['duplicate_geometry']+=1;continue
    mapped=Chem.Mol(mol)
    for i,a in enumerate(mapped.GetAtoms()):a.SetAtomMapNum(i+1)
    d={'molecule':{'id':prefix.rsplit('/',1)[1],'smiles':Chem.MolToSmiles(mapped), 'torsions':[[i+1 for i in tor]],'chemistry':chemistry(mol,tor)},
     'parent_smiles':parent,'scaffold':sc,'units':{'angles':'radian','energy':'kcal/mol'},
     'metadata':{'kind':'qm','qm_protocol':'B3LYP-D3BJ/DZVP; published OpenFF Sage 2.2.1 ForceBalance targets; qdata ENERGY in hartree',
      'baseline_protocol':f'RDKit {rdBase.rdkitVersion} UFF single-point on identical QM Cartesian coordinates; vdwThresh=100; no minimization',
      'source_url':SOURCE_URL,'source_record_id':prefix.rsplit('-',1)[1],'source_member_sha256':{n:hashlib.sha256(read(n)).hexdigest() for n in ['qdata.txt','scan.xyz','input.sdf','metadata.json']}},'pool':pool,'test':held}
    datasets.append(d);seen.add(parent)
   except (ValueError,AssertionError,RuntimeError,KeyError,StopIteration) as e:rejects[type(e).__name__+':'+str(e)[:80]]+=1
 # Scaffold assignment by salted hash. Freeze identities before scoring any policy.
 dev=[d for d in datasets if int(digest('qm-holdout-v1:'+d['scaffold'])[:8],16)%5==0]
 test=[d for d in datasets if int(digest('qm-holdout-v1:'+d['scaffold'])[:8],16)%5!=0]
 key=lambda d:digest('select-v1:'+d['parent_smiles'])
 dev=sorted(dev,key=key)[:dev_n];test=sorted(test,key=key)[:test_n]
 if len(dev)<dev_n or len(test)<test_n:raise ValueError(f'Insufficient eligible molecules: dev={len(dev)}, test={len(test)}, rejects={dict(rejects)}')
 assert not {d['scaffold'] for d in dev}&{d['scaffold'] for d in test}
 splits={'development':[d['molecule']['id'] for d in dev],'heldout':[d['molecule']['id'] for d in test]}
 hashes={}
 for d in dev+test:
  p=out/'datasets'/(d['molecule']['id']+'.json');p.write_text(json.dumps(d,indent=2));hashes[d['molecule']['id']]=hashlib.sha256(p.read_bytes()).hexdigest()
 manifest={'source_url':SOURCE_URL,'source_commit':SOURCE_COMMIT,'archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
  'eligible_unique_parents':len(datasets),'rejected':dict(rejects),'splits':splits,'dataset_hashes':hashes,
  'split_method':'SHA256(qm-holdout-v1:scaffold) mod 5, development=0, heldout=1..4; ring Murcko or acyclic generic full graph; nonstereo parent dedup; exclude pilot identities and ring scaffolds',
  'conformer_split':'parent-hash-seeded random 8/24 test geometries; 16/24 acquisition pool; no QM-energy filtering',
  'development_scaffolds':len({d['scaffold'] for d in dev}),'heldout_scaffolds':len({d['scaffold'] for d in test})}
 (out/'split.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k not in ['splits','dataset_hashes']},indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--out',required=True);p.add_argument('--development',type=int,default=60);p.add_argument('--heldout',type=int,default=300);a=p.parse_args();prepare(a.archive,a.out,a.development,a.heldout)
