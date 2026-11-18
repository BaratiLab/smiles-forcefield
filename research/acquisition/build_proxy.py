"""Real geometries, MMFF94 minus UFF labels. This is NOT a QM benchmark."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolTransforms

def build(out, size=48):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 for k,m in enumerate(json.loads((Path(__file__).parent/'molecules.json').read_text())):
  mol=Chem.AddHs(Chem.MolFromSmiles(m['smiles']))
  assert AllChem.EmbedMolecule(mol,randomSeed=101+k)==0
  assert AllChem.MMFFOptimizeMolecule(mol,maxIters=2000)==0
  maps={a.GetAtomMapNum():a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
  torsions=[[maps[i] for i in t] for t in m['torsions']]
  props=AllChem.MMFFGetMoleculeProperties(mol)
  data={'molecule':m,'units':{'angles':'radian','energy':'kcal/mol'},'metadata':{'kind':'mm_proxy','reference_protocol':'RDKit MMFF94, unconstrained single point','baseline_protocol':'RDKit UFF, same coordinates','qm_protocol':'NONE: qm_energy_kcal_mol is a legacy field containing MMFF94, NOT QM','rdkit_version':rdBase.rdkitVersion,'warning':'Engineering proxy only. No quantum energies or Sage refinement.'}}
  hashes=set()
  for split,seed in [('pool',900+k),('test',1900+k)]:
   rng=np.random.default_rng(seed);rows=[];writer=Chem.SDWriter(str(out/f'{m["id"]}-{split}.sdf'))
   attempts=0
   while len(rows)<size:
    attempts+=1
    if attempts>size*20:raise RuntimeError('Too many nonconverged geometries')
    geom=Chem.Mol(mol);conf=geom.GetConformer()
    targets=rng.uniform(-170,170,len(torsions))
    for t,a in zip(torsions,targets):rdMolTransforms.SetDihedralDeg(conf,*t,float(a))
    if m['id']=='phenylethanol':
     h=next(a.GetIdx() for a in geom.GetAtomWithIdx(maps[1]).GetNeighbors() if a.GetAtomicNum()==1)
     rdMolTransforms.SetDihedralDeg(conf,h,maps[1],maps[2],maps[3],float(rng.uniform(-180,180)))
    ff=AllChem.MMFFGetMoleculeForceField(geom,props)
    for t,a in zip(torsions,targets):ff.MMFFAddTorsionConstraint(*t,False,float(a),float(a),100.)
    ff.Initialize();converged=ff.Minimize(maxIts=2000)
    if converged:continue
    xyz=np.asarray(conf.GetPositions());digest=hashlib.sha256(xyz.round(6).tobytes()).hexdigest()
    if digest in hashes:continue
    hashes.add(digest)
    ref=AllChem.MMFFGetMoleculeForceField(geom,props).CalcEnergy()
    baseline=AllChem.UFFGetMoleculeForceField(geom).CalcEnergy()
    if not np.isfinite([ref,baseline]).all():continue
    identity=f'{split}_{len(rows):04d}'
    rows.append({'id':identity,'angles_rad':[float(rdMolTransforms.GetDihedralRad(conf,*t)) for t in torsions],'qm_energy_kcal_mol':ref,'baseline_energy_kcal_mol':baseline,'geometry_sha256':digest,'optimization_converged':True})
    geom.SetProp('_Name',identity);geom.SetProp('geometry_sha256',digest);writer.write(geom)
   writer.close();data[split]=rows
  (out/f'{m["id"]}.json').write_text(json.dumps(data,indent=2));print(m['id'],'ready',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=48);a=p.parse_args();build(a.out,a.size)
