"""Native GROMACS/LAMMPS plus independent AMBER reader comparison to OpenMM."""
from pathlib import Path
import json, os, subprocess
import numpy as np
import openmm as mm
from openmm import app,unit
ROOT=Path(__file__).resolve().parents[1]
BIN=Path(__import__('sys').executable).parent

def execute(argv,cwd,stdin=None):
    p=subprocess.run([str(a) for a in argv],cwd=cwd,input=stdin,text=True,capture_output=True,timeout=120)
    name=Path(str(argv[0])).name+'_'+str(argv[1] if len(argv)>1 else 'run').replace('/','_')
    (cwd/(name+'.log')).write_text(p.stdout+'\n'+p.stderr)
    if p.returncode: raise RuntimeError(f'Failed {argv}: see {cwd/name}.log')
    return p

def state(system,xyz):
    integrator=mm.VerletIntegrator(.001)
    ctx=mm.Context(system,integrator,mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(xyz*unit.nanometer)
    s=ctx.getState(getEnergy=True,getForces=True)
    return s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)

def numbers(path):
    return np.loadtxt([l for l in path.read_text().splitlines() if l and l[0] not in '#@'])

def main():
    rows=[]
    for out in sorted(ROOT.iterdir()):
        if not (out/'reference_system.xml').exists():continue
        print('Validating',out.name,flush=True)
        system=mm.XmlSerializer.deserialize((out/'reference_system.xml').read_text())
        xyz=np.load(out/'positions_nm.npy')
        e,f=state(system,xyz)
        # Independent AMBER topology reader; native NAMD not installed.
        amber=app.AmberPrmtopFile(str(out/'namd/molecule.prmtop'))
        ac=app.AmberInpcrdFile(str(out/'namd/molecule.inpcrd'))
        coordinate_error=float(np.max(np.abs(ac.positions.value_in_unit(unit.nanometer)-xyz)))
        if coordinate_error>1e-7: raise ValueError('AMBER coordinates changed during serialization')
        ae,af=state(amber.createSystem(nonbondedMethod=app.NoCutoff,constraints=None,rigidWater=False,removeCMMotion=False),xyz)
        rows.append(dict(molecule=out.name,route='AMBER readback (OpenMM; not native NAMD)',coordinate_max_error_nm=coordinate_error,energy_kj_mol=ae,energy_abs_error=abs(ae-e),force_max_abs_error=float(np.max(np.abs(af-f)))))
        gd=out/'gromacs'
        (gd/'validate.mdp').write_text('''integrator = md
nsteps = 0
dt = 0.001
constraints = none
cutoff-scheme = Verlet
nstlist = 1
rlist = 2.5
coulombtype = Cut-off
coulomb-modifier = None
rcoulomb = 2.5
vdwtype = Cut-off
vdw-modifier = None
rvdw = 2.5
DispCorr = no
pbc = xyz
nstenergy = 1
nstcalcenergy = 1
nstfout = 1
nstxout = 1
comm-mode = None
verlet-buffer-tolerance = -1
''')
        execute([BIN/'gmx','grompp','-f','validate.mdp','-c','molecule.gro','-p','molecule.top','-o','validate.tpr'],gd)
        execute([BIN/'gmx','mdrun','-s','validate.tpr','-deffnm','validate','-ntmpi','1','-ntomp','1'],gd)
        execute([BIN/'gmx','energy','-f','validate.edr','-o','energy.xvg'],gd,'Potential\n0\n')
        execute([BIN/'gmx','traj','-f','validate.trr','-s','validate.tpr','-of','forces.xvg','-fp'],gd,'0\n')
        ge=float(np.atleast_2d(numbers(gd/'energy.xvg'))[0,1])
        gf=np.atleast_2d(numbers(gd/'forces.xvg'))[0,1:].reshape(-1,3)
        # Coordinate serialization error quantified independently below.
        glines=(gd/'molecule.gro').read_text().splitlines()
        gxyz=np.array([[float(line[20+13*j:20+13*(j+1)]) for j in range(3)] for line in glines[2:2+len(xyz)]])
        gre,grf=state(system,np.array(gxyz))
        rows.append(dict(molecule=out.name,route='GROMACS native',energy_kj_mol=ge,energy_abs_error=abs(ge-gre),force_max_abs_error=float(np.max(np.abs(gf-grf))),coordinate_max_error_nm=float(np.max(np.abs(np.array(gxyz)-xyz)))))
        ld=out/'lammps'
        # Exporter writes interaction styles and coefficients; adapt its point-energy input.
        source=(ld/'molecule_pointenergy.in').read_text()
        source=source.replace('run 0','dump validation all custom 1 forces.dump id fx fy fz\ndump_modify validation sort id format float %.16g\nthermo_style custom step pe\nthermo_modify format float %.16g\nrun 0')
        (ld/'validate.in').write_text(source)
        lp=execute([BIN/'lmp','-in','validate.in'],ld)
        ls=lp.stdout.splitlines()
        le=None
        for i,line in enumerate(ls):
            if line.strip().split()==['Step','PotEng']:
                le=float(ls[i+1].split()[1])*4.184
        if le is None: raise ValueError('LAMMPS energy not found')
        lines=(ld/'forces.dump').read_text().splitlines()
        start=next(i for i,l in enumerate(lines) if l.startswith('ITEM: ATOMS'))+1
        lf=np.array([[float(v) for v in l.split()[1:4]] for l in lines[start:start+len(xyz)]])*41.84
        rows.append(dict(molecule=out.name,route='LAMMPS native',energy_kj_mol=le,energy_abs_error=abs(le-e),force_max_abs_error=float(np.max(np.abs(lf-f)))))
    for r in rows:
        r['passed']=r['energy_abs_error']<.01 and r['force_max_abs_error']<.1
    (ROOT/'validation.json').write_text(json.dumps(dict(energy_tolerance_kj_mol=.01,force_component_tolerance_kj_mol_nm=.1,scope='One isolated geometry per molecule; no native NAMD executable',results=rows),indent=2))
    print(json.dumps(rows,indent=2))
    if not all(r['passed'] for r in rows): raise SystemExit('Export consistency failure; see validation.json')
if __name__=='__main__':main()
