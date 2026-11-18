"""Check periodic residual correction energy and Cartesian forces in OpenMM."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from smiles_service.repair.core import Posterior


def validate():
    import openmm as mm
    from openmm import unit
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms

    hs = list(range(1, 7))
    a = np.linspace(-np.pi, np.pi, 40, endpoint=False)
    y = sum(np.cos(n * a) / (n + 1) + np.sin(n * a) / (n + 2) for n in hs)
    model = Posterior(a, y, hs)
    system = mm.System()
    for _ in range(4):
        system.addParticle(12)
    force = mm.PeriodicTorsionForce()
    for term in model.periodic_terms():
        force.addTorsion(
            0,
            1,
            2,
            3,
            term["periodicity"],
            term["phase_rad"],
            term["k_kcal_mol"] * 4.184,
        )
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    conf = Chem.Conformer(4)
    values = []
    pred = []
    max_force_error = 0

    def energy(x):
        context.setPositions(x * unit.nanometer)
        return (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )

    for phi in np.linspace(-2.8, 2.8, 7):
        xyz = np.array(
            [
                [0, 0.1, 0],
                [0, 0, 0],
                [0.1, 0, 0],
                [0.1, 0.1 * np.cos(phi), 0.1 * np.sin(phi)],
            ]
        )
        for i, p in enumerate(xyz):
            conf.SetAtomPosition(i, p.tolist())
        angle = rdMolTransforms.GetDihedralRad(conf, 0, 1, 2, 3)
        values.append(energy(xyz) / 4.184)
        pred.append(float(model.predict([angle])[0]))
        context.setPositions(xyz * unit.nanometer)
        f = (
            context.getState(getForces=True)
            .getForces(asNumpy=True)
            .value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        )
        eps = 1e-6
        fd = np.zeros_like(xyz)
        for i in range(4):
            for j in range(3):
                plus = xyz.copy()
                minus = xyz.copy()
                plus[i, j] += eps
                minus[i, j] -= eps
                fd[i, j] = -(energy(plus) - energy(minus)) / (2 * eps)
        max_force_error = max(max_force_error, float(np.max(np.abs(f - fd))))
    delta = np.array(values) - pred
    energyerror = float(np.max(np.abs(delta - delta.mean())))
    assert energyerror < 1e-9 and max_force_error < 1e-4
    return {
        "engine": "OpenMM " + mm.__version__,
        "platform": "Reference",
        "angles": 7,
        "relative_energy_max_error_kcal_mol": energyerror,
        "cartesian_force_fd_max_error_kj_mol_nm": max_force_error,
        "energy_tolerance_kcal_mol": 1e-9,
        "force_tolerance_kj_mol_nm": 1e-4,
        "scope": "correction term only, not full molecule/package validation",
        "gromacs_lammps_namd_repair_validation": False,
    }


if __name__ == "__main__":
    result = validate()
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
