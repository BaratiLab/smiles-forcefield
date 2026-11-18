"""Build geometry-complete Sage training episodes, separate from UFF adapter data."""
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from smiles_service.repair.core import semantic_id, rules

SAGE_SYSTEM = """Propose at most two valid torsion repair programs as JSON {"programs":[...]}. Each program has EXACT action, scope, harmonics. action is revise/specialize/keep/abstain. scope broad means sum the correction over all verified instances of the Sage parent parameter in this molecule; scope local means the verified narrower environment subset and requires specialize. Harmonics is sorted unique integers in1..6. keep/abstain use broad and [1,2,3]. You never predict force constants. The background Sage force field and AM1-BCC charges are fixed. Tools fit numerical coefficients and support-only validation may reject your hypothesis. Fourier sums account for every affected torsion instance, not just the driven angle. Do not change chemical state or invent QM observations."""


def features(mol, xyz, paths):
    conf = Chem.Conformer(mol.GetNumAtoms())
    result = []
    for frame in xyz:
        for i, p in enumerate(frame):
            conf.SetAtomPosition(i, p.tolist())
        angles = np.array(
            [rdMolTransforms.GetDihedralRad(conf, *path) for path in paths]
        )
        result.append(
            [float(f(h * angles).sum()) for h in range(1, 7) for f in (np.cos, np.sin)]
        )
    return np.array(result)


def fit(x, y, ids):
    xm = x[ids].mean(axis=0)
    ym = y[ids].mean()
    xx = x[ids] - xm
    cov = np.linalg.solve(np.eye(x.shape[1]) + xx.T @ xx / 0.01, np.eye(x.shape[1]))
    coef = cov @ xx.T @ (y[ids] - ym) / 0.01
    return ym + (x - xm) @ coef, cov, coef


def build(sage, data, out, role="train", seeds=3):
    sage = Path(sage)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    episodes = []
    datasets = []
    base = {"action": "keep", "scope": "broad", "harmonics": [1, 2, 3]}
    programs = [base] + rules()
    for folder in sorted(sage.iterdir()):
        if not (folder / "audit.json").exists():
            continue
        audit = json.loads((folder / "audit.json").read_text())
        old = json.loads((Path(data) / "datasets" / f"{folder.name}.json").read_text())
        mol = Chem.MolFromMolFile(str(folder / "molecule.sdf"), removeHs=False)
        xyz = np.load(folder / "coordinates_angstrom.npy")
        qm = np.load(folder / "qm_energy_kcal_mol.npy")
        mm = np.load(folder / "baseline_energy_kcal_mol.npy")
        y = qm - mm
        matrices = {
            "broad": features(mol, xyz, audit["all_parent_instances"]),
            "local": features(mol, xyz, audit["local_instances"]),
        }
        pool = [int(r["id"].rsplit("-", 1)[1]) for r in old["pool"]]
        test = [int(r["id"].rsplit("-", 1)[1]) for r in old["test"]]
        datasets.append(
            {
                "molecule": folder.name,
                "parent": old["parent_smiles"],
                "scaffold": old["scaffold"],
                "pool_indices": pool,
                "retired_test_indices": test,
                "qm_energy_kcal_mol": qm.tolist(),
                "sage_energy_kcal_mol": mm.tolist(),
                "fourier_sum_matrices": {k: v.tolist() for k, v in matrices.items()},
                "parent_parameter_id": audit["parent_parameter_id"],
                "parent_paths": audit["all_parent_instances"],
                "local_paths": audit["local_instances"],
                "forcefield_serialization_sha256": audit[
                    "forcefield_serialization_sha256"
                ],
            }
        )
        for seed in range(seeds):
            chosen = sorted(
                pool,
                key=lambda i: hashlib.sha256(
                    (f"sage-initial-v1-{seed}:" + str(i) + ":" + folder.name).encode()
                ).hexdigest(),
            )[:4]
            x = matrices["broad"][:, :6]
            for _ in range(4):
                _, cov, _ = fit(x, y, chosen)
                xm = x[chosen].mean(axis=0)
                available = [i for i in pool if i not in chosen]
                best = min(
                    available,
                    key=lambda i: (-float((x[i] - xm) @ cov @ (x[i] - xm)), i),
                )
                chosen.append(best)
            hold = [i for i in pool if i not in chosen]
            basepred, basecov, basecoef = fit(x, y, chosen)
            scores = []
            for p in programs:
                columns = [2 * (h - 1) + j for h in p["harmonics"] for j in (0, 1)]
                xx = matrices[p["scope"]][:, columns]
                pred, _, _ = fit(xx, y, chosen)
                errors = y[hold] - pred[hold]
                loo = []
                for i in chosen:
                    pp, _, _ = fit(xx, y, [j for j in chosen if j != i])
                    loo.append(y[i] - pp[i])
                scores.append(
                    {
                        "program": p,
                        "support_loo_rmse": float(np.sqrt(np.mean(np.array(loo) ** 2))),
                        "holdout_centered_rmse": float(np.std(errors)),
                    }
                )
            baseline = scores[0]
            valid = [
                r
                for r in scores[1:]
                if r["support_loo_rmse"] < 0.98 * baseline["support_loo_rmse"]
                and r["holdout_centered_rmse"]
                < 0.98 * baseline["holdout_centered_rmse"]
            ]
            best = sorted(
                valid,
                key=lambda r: (
                    r["holdout_centered_rmse"],
                    len(r["program"]["harmonics"]),
                    semantic_id(r["program"]),
                ),
            )[:2]
            state = {
                "schema_version": "sage-geometry-repair-v1",
                "smiles": old["molecule"]["smiles"],
                "driven_torsion": old["molecule"]["torsions"][0],
                "background": "Sage2.2.1 unconstrained, fixed AM1-BCC, unswitched vacuum",
                "parent_parameter_id": audit["parent_parameter_id"],
                "parent_smirks": audit["parent_smirks"],
                "verified_local_selector": audit["local_selector"],
                "parent_instance_count": len(audit["all_parent_instances"]),
                "local_instance_count": len(audit["local_instances"]),
                "observations": [
                    {
                        "relative_correction_kcal_mol": round(
                            float(y[i] - y[chosen].mean()), 6
                        ),
                        "baseline_residual_kcal_mol": round(
                            float(y[i] - basepred[i]), 6
                        ),
                        "parent_fourier_sums": np.round(
                            matrices["broad"][i], 6
                        ).tolist(),
                        "local_fourier_sums": np.round(
                            matrices["local"][i], 6
                        ).tolist(),
                    }
                    for i in chosen
                ],
                "baseline_posterior_stdev": np.sqrt(np.diag(basecov)).tolist(),
            }
            episodes.append(
                {
                    "molecule": folder.name,
                    "parent": old["parent_smiles"],
                    "scaffold": old["scaffold"],
                    "seed": seed,
                    "state": state,
                    "support_indices": chosen,
                    "internal_teacher_holdout_indices": hold,
                    "candidate_utilities": scores,
                    "answer": {
                        "programs": [r["program"] for r in best]
                        or [
                            {
                                "action": "abstain",
                                "scope": "broad",
                                "harmonics": [1, 2, 3],
                            }
                        ]
                    },
                }
            )
    (out / "datasets.json").write_text(json.dumps(datasets, indent=2))
    (out / "episodes.json").write_text(json.dumps(episodes, indent=2))
    (out / "train.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": SAGE_SYSTEM},
                        {
                            "role": "user",
                            "content": json.dumps(r["state"], sort_keys=True),
                        },
                        {"role": "assistant", "content": json.dumps(r["answer"])},
                    ]
                }
            )
            + "\n"
            for r in episodes
        )
    )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "status": "geometry-complete Sage development supervision; no adapter trained on this corpus yet",
                "parents": len(datasets),
                "episodes": len(episodes),
                "source": role
                + " parents only, fixed16pool/8retiredtest conformer allocation",
                "teacher_labels": "remaining internal pool points; retired test labels stored for replay but not used for teacher selection",
                "scope_semantics": "parent or local summed proper-torsion correction, numerical coefficients shared across affected instances; not the old environment-conditioned-prior surrogate",
                "not_interchangeable_with_old_adapter": "The current completed64-update adapter was trained on UFF residuals and a different scope meaning. It cannot be relabeled as a Sage-trained model. Separate training/validation needed.",
                "independence": str(seeds)
                + " episodes per parent; repeated episodes are correlated; no new chemical generalization claimed",
            },
            indent=2,
        )
    )
    print("Sage episodes", len(episodes), "parents", len(datasets))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sage", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--role", choices=["train", "validation"], default="train")
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    build(a.sage, a.data, a.out, a.role, a.seeds)
