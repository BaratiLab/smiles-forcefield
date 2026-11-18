"""Export geometry-complete development corrections, isolated from public jobs."""
import argparse, io, json, os, re, sys, tarfile
from pathlib import Path

os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from rdkit import Chem
from openff.toolkit import Molecule, ForceField
from openff.interchange import Interchange
from openff.interchange.models import ProperTorsionKey, PotentialKey
from openff.interchange.components.potentials import Potential
from openff.units import unit
import openmm as mm
from openmm import unit as ommunit
from smiles_service.chemistry import validate


NAMD_TEMPLATE = """# Research example; native NAMD has not been run.
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
"""
EXAMPLE_NOTICE = """Experimental topology-specific Sage correction, fitted to archived QM scans.
See ../validation.json for native GROMACS/LAMMPS and AMBER readback checks.
Native NAMD has not been run; its singlepoint.namd is an untested template.
These isolated vacuum files are not a production MD protocol or an accuracy certification.
See patch-provenance.json for the exact affected torsions, source and fitted terms.
"""


def normalized(t):
    return min(tuple(t), tuple(t[::-1]))


def run(sage, archive, out):
    sage = Path(sage)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    audits = json.loads((sage / "summary.json").read_text())
    names = [r["molecule"] for r in audits]
    payload = {}
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            if (
                m.isfile()
                and m.name.endswith("/input.sdf")
                and any("/" + n + "/" in m.name for n in names)
            ):
                payload[m.name] = tf.extractfile(m).read()
    parity = []
    for audit in audits:
        name = audit["molecule"]
        folder = sage / name
        rdmol = next(
            iter(
                Chem.ForwardSDMolSupplier(
                    io.BytesIO(
                        next(v for k, v in payload.items() if "/" + name + "/" in k)
                    ),
                    removeHs=False,
                )
            )
        )
        for atom in rdmol.GetAtoms():
            atom.SetAtomMapNum(0)
        mol = Molecule.from_rdkit(rdmol, hydrogens_are_explicit=True)
        reference = mm.XmlSerializer.deserialize((folder / "patched.xml").read_text())
        baseline = mm.XmlSerializer.deserialize((folder / "baseline.xml").read_text())
        nb = next(f for f in baseline.getForces() if isinstance(f, mm.NonbondedForce))
        mol.partial_charges = (
            np.array(
                [
                    nb.getParticleParameters(i)[0].value_in_unit(
                        ommunit.elementary_charge
                    )
                    for i in range(mol.n_atoms)
                ]
            )
            * unit.elementary_charge
        )
        coords = np.load(folder / "coordinates_angstrom.npy")
        best = audit["selected"]
        for frame in (0, 8, 16):
            target = out / f"{name}-frame{frame:02d}"
            target.mkdir(exist_ok=True)
            ff = ForceField("openff_unconstrained-2.2.1.offxml")
            ic = Interchange.from_smirnoff(
                ff, mol.to_topology(), charge_from_molecules=[mol]
            )
            collection = ic.collections["ProperTorsions"]
            for path in best["paths"]:
                multiplicities = [
                    key.mult or 0
                    for key in collection.key_map
                    if normalized(key.atom_indices) == normalized(path)
                ]
                start = max(multiplicities, default=-1) + 1
                for j, (n, (a, b)) in enumerate(
                    zip(
                        best["harmonics"], np.array(best["coefficients"]).reshape(-1, 2)
                    )
                ):
                    tk = ProperTorsionKey(atom_indices=tuple(path), mult=start + j)
                    pk = PotentialKey(
                        id="validated-development-correction:" + name + ":" + str(path),
                        mult=start + j,
                        associated_handler="ProperTorsions",
                    )
                    collection.key_map[tk] = pk
                    collection.potentials[pk] = Potential(
                        parameters={
                            "k": float(np.hypot(a, b)) * unit.kilocalorie_per_mole,
                            "periodicity": n * unit.dimensionless,
                            "phase": float(np.arctan2(b, a)) * unit.radian,
                            "idivf": 1 * unit.dimensionless,
                        }
                    )
            xyz = coords[frame] * 0.1
            xyz = xyz - xyz.min(axis=0) + 0.25
            ic.positions = xyz * unit.nanometer
            ic.box = np.eye(3) * 6 * unit.nanometer
            ic.collections["vdW"].switch_width = 0 * unit.nanometer
            for engine in ["gromacs", "lammps", "namd"]:
                (target / engine).mkdir(exist_ok=True)
            ic.to_gromacs(str(target / "gromacs/molecule"), decimal=8)
            ic.to_lammps(str(target / "lammps/molecule"))
            ic.to_amber(str(target / "namd/molecule"))
            lp = target / "lammps/molecule_pointenergy.in"
            text = lp.read_text()
            text = re.sub(
                r"pair_style .*", "pair_style lj/cut/coul/cut 25.0 25.0", text
            )
            text = re.sub(r"kspace_style .*\n", "", text)
            text = text.replace("tail yes", "tail no")
            text = re.sub(r"read_data .*", "read_data molecule.lmp", text)
            lp.write_text(text)
            generated = ic.to_openmm(combine_nonbonded_forces=True)
            for system in [generated, reference]:
                for f in system.getForces():
                    if isinstance(f, mm.NonbondedForce):
                        f.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
                        f.setUseSwitchingFunction(False)
            e, f = validate.state(reference, xyz)
            ee, fforce = validate.state(generated, xyz)
            row = {
                "molecule": name,
                "frame": frame,
                "energy_error_kj_mol": abs(e - ee),
                "force_max_error_kj_mol_nm": float(np.max(np.abs(f - fforce))),
            }
            assert (
                row["energy_error_kj_mol"] < 1e-7
                and row["force_max_error_kj_mol_nm"] < 1e-6
            )
            parity.append(row)
            (target / "reference_system.xml").write_text(
                mm.XmlSerializer.serialize(reference)
            )
            np.save(target / "positions_nm.npy", xyz)
            (target / "patch-provenance.json").write_text(
                json.dumps(
                    {
                        "status": "development only; not automatically applied to public jobs",
                        "molecule": name,
                        "frame": frame,
                        "source_audit": audit,
                        "native_namd_validation": False,
                    },
                    indent=2,
                )
            )
            (target / "namd" / "singlepoint.namd").write_text(NAMD_TEMPLATE)
            (target / "README.txt").write_text(EXAMPLE_NOTICE)
            print("EXPORTED", target.name, flush=True)
    (out / "interchange-patch-parity.json").write_text(json.dumps(parity, indent=2))
    validate.ROOT = out
    validate.main()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sage", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.sage, a.archive, a.out)
