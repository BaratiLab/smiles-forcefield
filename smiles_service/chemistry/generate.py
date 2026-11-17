"""Generate Sage 2.2.1 + AM1-BCC packages; isolated-molecule validation setup."""
from pathlib import Path
import json, os, re
import numpy as np
from openff.toolkit import Molecule, ForceField
from rdkit import Chem
from openff.units import unit
from openff.interchange import Interchange
import openmm
from openmm import app

ROOT=Path(__file__).resolve().parents[1]
MOLECULES=ROOT/'molecules.json'

def main():
    ff=ForceField('openff_unconstrained-2.2.1.offxml')
    # Vacuum reference: no switching. Files are isolated molecule examples,
    # NOT the default condensed-phase Sage nonbonded protocol.
    for record in json.loads(MOLECULES.read_text()):
        out=ROOT/record['id']; out.mkdir(exist_ok=True)
        print('Parameterizing',record['id'],flush=True)
        rdmol=Chem.AddHs(Chem.MolFromSmiles(record['smiles']))
        mol=Molecule.from_rdkit(rdmol,hydrogens_are_explicit=True)
        (out/'atom_mapping.json').write_text(json.dumps([dict(index_zero_based=a.GetIdx(),smiles_atom_map=a.GetAtomMapNum(),element=a.GetSymbol()) for a in rdmol.GetAtoms()],indent=2))
        mol.name='LIG'
        mol.generate_conformers(n_conformers=1)
        mol.assign_partial_charges('am1bcc')
        ic=Interchange.from_smirnoff(ff,mol.to_topology(),charge_from_molecules=[mol])
        xyz=mol.conformers[0].m_as(unit.nanometer)
        if np.max(np.linalg.norm(xyz[:,None,:]-xyz[None,:,:],axis=-1))>=2.0:
            raise ValueError('Molecule exceeds the 2 nm isolated validation size limit.')
        ic.positions=(xyz-xyz.min(axis=0)+0.25)*unit.nanometer
        mol._conformers[0]=ic.positions
        mol.to_file(str(out/'molecule.sdf'),file_format='sdf')
        ic.box=np.eye(3)*6*unit.nanometer
        ic.collections['vdW'].switch_width=0*unit.nanometer
        for folder in ['gromacs','lammps','namd']:
            (out/folder).mkdir(exist_ok=True)
        ic.to_gromacs(str(out/'gromacs/molecule'),decimal=8)
        ic.to_lammps(str(out/'lammps/molecule'))
        ic.to_amber(str(out/'namd/molecule'))
        lp=out/'lammps/molecule_pointenergy.in'
        text=lp.read_text()
        text=re.sub(r'pair_style .*', 'pair_style lj/cut/coul/cut 25.0 25.0',text)
        text=re.sub(r'kspace_style .*\n','',text)
        text=text.replace('tail yes','tail no')
        text=re.sub(r'read_data .*','read_data molecule.lmp',text)
        lp.write_text(text)

        system=ic.to_openmm(combine_nonbonded_forces=True)
        for force in system.getForces():
            if isinstance(force,openmm.NonbondedForce):
                force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
                force.setUseSwitchingFunction(False)
        (out/'reference_system.xml').write_text(openmm.XmlSerializer.serialize(system))
        xyz=ic.positions.m_as(unit.nanometer)
        np.save(out/'positions_nm.npy',xyz)
        with (out/'molecule.pdb').open('w') as f:
            app.PDBFile.writeFile(ic.to_openmm_topology(),xyz*openmm.unit.nanometer,f)
        (out/'provenance.json').write_text(json.dumps(dict(record,forcefield='openff_unconstrained-2.2.1.offxml',charges='AmberTools AM1-BCC',charge_sum=float(mol.partial_charges.m_as(unit.elementary_charge).sum()),nonbonded_protocol='Isolated molecule vacuum; no switching; no constraints',atom_count=mol.n_atoms),indent=2))
        (out/'namd/singlepoint.namd').write_text('''# Isolated molecule single-point validation, not a production MD protocol.
amber on
parmfile molecule.prmtop
ambercoor molecule.inpcrd
readexclusions on
exclude scaled1-4
1-4scaling 0.8333333333333333
scnb 2.0
switching off
cutoff 100
pairlistdist 102
PME off
rigidBonds none
temperature 0
outputName singlepoint
outputEnergies 1
run 0
''')
        print('Exported',record['id'],flush=True)

if __name__=='__main__':main()
