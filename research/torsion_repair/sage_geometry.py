"""Geometry-complete Sage torsion correction prototype, not a released force field.

Features sum over EVERY proper-torsion instance assigned to the selected parent
parameter (or its verified local subset), on full Cartesian scan geometries.
"""
import argparse, hashlib, io, json, os, tarfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from openff.toolkit import Molecule, ForceField
from openff.interchange import Interchange
from openff.units import unit
import openmm as mm
from openmm import unit as ommunit
from smiles_service.repair.core import chemistry, verified_scopes


def normalize(path):
    return min(tuple(path), tuple(path[::-1]))


def matrix(mol, coordinates, paths, hs):
    conf = Chem.Conformer(mol.GetNumAtoms())
    rows = []
    for xyz in coordinates:
        for i, p in enumerate(xyz):
            conf.SetAtomPosition(i, p.tolist())
        angles = [rdMolTransforms.GetDihedralRad(conf, *path) for path in paths]
        rows.append(
            [sum(f(h * a) for a in angles) for h in hs for f in (np.cos, np.sin)]
        )
    return np.array(rows)


def fit(x, y, indices):
    xm = x[indices].mean(axis=0)
    ym = float(y[indices].mean())
    xx = x[indices] - xm
    coef = np.linalg.solve(
        np.eye(x.shape[1]) + xx.T @ xx / 0.01, xx.T @ (y[indices] - ym) / 0.01
    )
    return coef, ym + (x - xm) @ coef


def run(archive, data, names, out, limit=2, continue_on_error=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = json.loads(Path(names).read_text())[:limit]
    payload = {}
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if (
                member.isfile()
                and any("/" + n + "/" in member.name for n in names)
                and member.name.rsplit("/", 1)[-1]
                in ["input.sdf", "scan.xyz", "qdata.txt", "metadata.json"]
            ):
                payload[member.name] = tf.extractfile(member).read()
    results = []
    ff = ForceField("openff_unconstrained-2.2.1.offxml")
    failures = []
    for name in names:
        try:
            folder = out / name
            folder.mkdir(exist_ok=True)
            prefix = next(p.rsplit("/", 1)[0] for p in payload if "/" + name + "/" in p)
            raw = lambda key: payload[prefix + "/" + key]
            rdmol = next(
                iter(
                    Chem.ForwardSDMolSupplier(
                        io.BytesIO(raw("input.sdf")), removeHs=False
                    )
                )
            )
            for i, atom in enumerate(rdmol.GetAtoms()):
                atom.SetAtomMapNum(i + 1)
            toolkit_mol = Chem.Mol(rdmol)
            for atom in toolkit_mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mol = Molecule.from_rdkit(
                toolkit_mol, hydrogens_are_explicit=True, allow_undefined_stereo=False
            )
            charge_conformer_policy = "toolkit-generated conformer"
            try:
                mol.assign_partial_charges("am1bcc")
            except ValueError as error:
                if "RDKit conformer generation failed" not in str(error):
                    raise
                # Fixed geometry-only fallback; never change state or charge method.
                # Use a pool frame, not a held-out evaluation conformer.
                old_data = json.loads(
                    (Path(data) / "datasets" / f"{name}.json").read_text()
                )
                frame_index = int(old_data["pool"][0]["id"].rsplit("-", 1)[1])
                xyz_lines = raw("scan.xyz").decode().splitlines()
                atom_count = int(xyz_lines[0])
                assert atom_count == mol.n_atoms
                begin = frame_index * (atom_count + 2) + 2
                charge_xyz = np.array(
                    [
                        [float(v) for v in line.split()[1:4]]
                        for line in xyz_lines[begin : begin + atom_count]
                    ]
                )
                mol.assign_partial_charges(
                    "am1bcc", use_conformers=[charge_xyz * unit.angstrom]
                )
                charge_conformer_policy = "fixed first pool geometry fallback after RDKit conformer-generation failure"

            labels = ff.label_molecules(mol.to_topology())[0]["ProperTorsions"]
            driven = json.loads(raw("metadata.json"))["dihedrals"][0]
            parent = next(
                p for k, p in labels.items() if normalize(k) == normalize(driven)
            )
            broad = sorted(
                {normalize(k) for k, p in labels.items() if p.id == parent.id}
            )
            scopes = verified_scopes(Chem.MolToSmiles(rdmol), [i + 1 for i in driven])
            query = Chem.MolFromSmarts(scopes["local"])
            local = sorted(
                set(broad)
                & {
                    normalize(path)
                    for path in rdmol.GetSubstructMatches(
                        query, uniquify=False, maxMatches=10000
                    )
                }
            )
            if normalize(driven) not in local:
                raise ValueError("Driven torsion missing from local assignment")
            text = raw("scan.xyz").decode().splitlines()
            coords = []
            pos = 0
            while pos < len(text):
                if not text[pos].strip():
                    pos += 1
                    continue
                n = int(text[pos])
                rows = [r.split() for r in text[pos + 2 : pos + 2 + n]]
                assert [r[0] for r in rows] == [a.GetSymbol() for a in rdmol.GetAtoms()]
                coords.append(np.array([[float(v) for v in r[1:4]] for r in rows]))
                pos += n + 2
            coords = np.array(coords)
            qm = np.array(
                [
                    float(line.split()[1]) * 627.5094740631
                    for line in raw("qdata.txt").decode().splitlines()
                    if line.startswith("ENERGY ")
                ]
            )
            assert len(qm) == len(coords) == 24
            ic = Interchange.from_smirnoff(
                ff, mol.to_topology(), charge_from_molecules=[mol]
            )
            system = ic.to_openmm(combine_nonbonded_forces=True)
            for force in system.getForces():
                if isinstance(force, mm.NonbondedForce):
                    force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
                    force.setUseSwitchingFunction(False)
            integrator = mm.VerletIntegrator(0.001)
            context = mm.Context(
                system, integrator, mm.Platform.getPlatformByName("Reference")
            )
            baseline = []
            for xyz in coords:
                context.setPositions(xyz * 0.1 * ommunit.nanometer)
                baseline.append(
                    context.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(ommunit.kilocalorie_per_mole)
                )
            y = qm - np.array(baseline)
            np.save(folder / "qm_energy_kcal_mol.npy", qm)
            np.save(folder / "baseline_energy_kcal_mol.npy", np.array(baseline))
            (folder / "molecule.sdf").write_text(Chem.MolToMolBlock(rdmol) + "\n$$$$\n")
            # Existing retired conformer allocation retained; no new held-out claim.
            old = json.loads((Path(data) / "datasets" / f"{name}.json").read_text())
            pool = [int(r["id"].rsplit("-", 1)[1]) for r in old["pool"]]
            test = [int(r["id"].rsplit("-", 1)[1]) for r in old["test"]]
            candidates = []
            for scope, paths in [("parent", broad), ("specialized", local)]:
                for h in (2, 3, 4, 6):
                    hs = list(range(1, h + 1))
                    x = matrix(rdmol, coords, paths, hs)
                    loo = []
                    for i in pool:
                        _, prediction = fit(x, y, [j for j in pool if j != i])
                        loo.append(prediction[i] - y[i])
                    coef, prediction = fit(x, y, pool)
                    error = y[test] - prediction[test]
                    candidates.append(
                        {
                            "scope": scope,
                            "harmonics": hs,
                            "paths": paths,
                            "coefficients": coef.tolist(),
                            "loo_rmse": float(np.sqrt(np.mean(np.array(loo) ** 2))),
                            "retired_test_centered_rmse": float(np.std(error)),
                        }
                    )
            best = min(
                candidates,
                key=lambda r: (r["loo_rmse"], len(r["harmonics"]), r["scope"]),
            )
            patched = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
            patch = mm.PeriodicTorsionForce()
            for path in best["paths"]:
                for h, (a, b) in zip(
                    best["harmonics"], np.array(best["coefficients"]).reshape(-1, 2)
                ):
                    patch.addTorsion(
                        *path, h, float(np.arctan2(b, a)), float(np.hypot(a, b)) * 4.184
                    )
            patched.addForce(patch)
            integ2 = mm.VerletIntegrator(0.001)
            ctx2 = mm.Context(
                patched, integ2, mm.Platform.getPlatformByName("Reference")
            )
            patch_energies = []
            for xyz, base in zip(coords, baseline):
                ctx2.setPositions(xyz * 0.1 * ommunit.nanometer)
                patch_energies.append(
                    ctx2.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(ommunit.kilocalorie_per_mole)
                    - base
                )
            summed = matrix(rdmol, coords, best["paths"], best["harmonics"]) @ np.array(
                best["coefficients"]
            )
            delta = np.array(patch_energies) - summed
            parity = float(np.max(np.abs(delta - delta.mean())))
            assert parity < 1e-7
            (folder / "baseline.xml").write_text(mm.XmlSerializer.serialize(system))
            (folder / "patched.xml").write_text(mm.XmlSerializer.serialize(patched))
            np.save(folder / "coordinates_angstrom.npy", coords)
            result = {
                "molecule": name,
                "parent_parameter_id": parent.id,
                "parent_smirks": parent.smirks,
                "local_selector": scopes["local"],
                "all_parent_instances": broad,
                "local_instances": local,
                "selected": best,
                "candidates": candidates,
                "summed_feature_vs_patched_openmm_max_relative_energy_error_kcal_mol": parity,
                "baseline_retired_test_centered_rmse": float(np.std(y[test])),
                "forcefield": "openff_unconstrained-2.2.1.offxml",
                "forcefield_serialization_sha256": hashlib.sha256(
                    ff.to_string().encode()
                ).hexdigest(),
                "charge_model": "AmberTools AM1-BCC, fixed across frames",
                "charge_conformer_policy": charge_conformer_policy,
                "status": "retired training molecule development audit; no global OFFXML type replacement or engine export claimed",
                "geometry_source_sha256": hashlib.sha256(raw("scan.xyz")).hexdigest(),
            }
            (folder / "audit.json").write_text(json.dumps(result, indent=2))
            results.append(result)
            print("SAGE", name, parent.id, len(broad), len(local), parity, flush=True)
            del context, ctx2, integrator, integ2
        except Exception as error:
            failures.append(
                {
                    "molecule": name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            (out / "failures.json").write_text(json.dumps(failures, indent=2))
            print(
                "SAGE_FAILURE", name, type(error).__name__, str(error)[:200], flush=True
            )
            if not continue_on_error:
                raise
        (out / "summary.json").write_text(json.dumps(results, indent=2))
    (out / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--continue-on-error", action="store_true")
    a = p.parse_args()
    os.environ["PATH"] = (
        str(Path(__import__("sys").executable).parent) + os.pathsep + os.environ["PATH"]
    )
    run(a.archive, a.data, a.names, a.out, a.limit, a.continue_on_error)
