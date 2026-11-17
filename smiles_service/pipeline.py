"""One isolated chemistry job, executed in a bounded worker subprocess."""
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from .chemistry import generate,validate

SCOPE='One isolated geometry; unswitched vacuum interactions. Export consistency is not force-field accuracy. Native NAMD was not run.'

def progress(folder,stage):
    p=folder/'progress.tmp';p.write_text(json.dumps({'stage':stage}));p.replace(folder/'progress.json')

def package(folder,artifacts,report):
    molecule=artifacts/'molecule'
    (molecule/'validation.json').write_text(json.dumps(report,indent=2))
    (molecule/'README.txt').write_text('Sage 2.2.1 / AM1-BCC\n'+SCOPE+'\nGROMACS: use validate.mdp. LAMMPS: molecule_pointenergy.in. NAMD: singlepoint.namd (untested).\nNo solvent, pressure coupling, or production MD settings included.\n')
    common=['README.txt','validation.json','provenance.json','atom_mapping.json','molecule.sdf','molecule.pdb']
    for kind in ['all','gromacs','lammps','namd']:
        paths=list(molecule.rglob('*')) if kind=='all' else [molecule/n for n in common]+list((molecule/kind).rglob('*'))
        with ZipFile(folder/(kind+'.zip'),'w',ZIP_DEFLATED) as z:
            for p in sorted(paths):
                if p.is_file() and not p.name.startswith('#'):
                    z.write(p,str(p.relative_to(molecule)))
    return {kind:hashlib.sha256((folder/(kind+'.zip')).read_bytes()).hexdigest() for kind in ['all','gromacs','lammps','namd']}

def main():
    folder=Path(sys.argv[1]).resolve()
    request=json.loads((folder/'request.json').read_text())
    artifacts=folder/'artifacts';artifacts.mkdir()
    (artifacts/'molecules.json').write_text(json.dumps([{'id':'molecule','name':'Submitted molecule','smiles':request['smiles']}]))
    progress(folder,'Assigning Sage parameters and AM1-BCC charges')
    generate.ROOT=artifacts;generate.MOLECULES=artifacts/'molecules.json';generate.main()
    progress(folder,'Comparing native GROMACS/LAMMPS energies and forces')
    validate.ROOT=artifacts;validate.main()
    report=json.loads((artifacts/'validation.json').read_text())
    if len(report['results'])!=3 or not all(r['passed'] for r in report['results']):raise RuntimeError('Incomplete or failed validation')
    for p in (artifacts/'molecule').glob('gromacs/molecule_pointenergy.mdp'):p.unlink()
    for p in (artifacts/'molecule').glob('namd/molecule_pointenergy.in'):p.unlink()
    progress(folder,'Preparing downloadable ZIP packages')
    hashes=package(folder,artifacts,report)
    result={'forcefield':'OpenFF Sage 2.2.1 (unconstrained)','charges':'AM1-BCC',
            'scope':SCOPE,'validation':report,'native_namd_validated':False,'sha256':hashes}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
