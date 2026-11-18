"""Geometry-complete numerical development comparators before Sage LLM training."""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sage_episodes import fit
from smiles_service.repair.core import rules, semantic_id

PROGRAMS = rules()
BASE = {"action": "keep", "scope": "broad", "harmonics": [1, 2, 3]}
FIXED = [
    {"action": "revise", "scope": "broad", "harmonics": [1, 2]},
    {"action": "specialize", "scope": "local", "harmonics": [1, 2, 3, 4]},
]


def vector(state):
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128).GetFingerprint(
        Chem.MolFromSmiles(state["smiles"])
    )
    bits = np.zeros(128)
    DataStructs.ConvertToNumpyArray(fp, bits)
    obs = state["observations"]
    y = np.array([r["relative_correction_kcal_mol"] for r in obs])
    res = np.array([r["baseline_residual_kcal_mol"] for r in obs])
    px = np.array([r["parent_fourier_sums"] for r in obs])
    lx = np.array([r["local_fourier_sums"] for r in obs])
    return np.r_[
        bits,
        np.std(y),
        np.std(res),
        np.ptp(y),
        state["parent_instance_count"],
        state["local_instance_count"],
        px.mean(0),
        px.std(0),
        lx.mean(0),
        lx.std(0),
    ]


def score(ep, programs):
    lookup = {
        (r["program"]["scope"], tuple(r["program"]["harmonics"])): r
        for r in ep["candidate_utilities"]
    }
    base = ep["candidate_utilities"][0]
    choices = [base]
    for p in programs:
        r = lookup[p["scope"], tuple(p["harmonics"])]
        if r["support_loo_rmse"] < 0.98 * base["support_loo_rmse"]:
            choices.append(r)
    return min(
        choices,
        key=lambda r: (
            r["support_loo_rmse"],
            len(r["program"]["harmonics"]),
            semantic_id(r["program"]),
        ),
    )


def run(train, validation, out):
    train = Path(train)
    validation = Path(validation)
    tr = json.loads((train / "episodes.json").read_text())
    va = json.loads((validation / "episodes.json").read_text())
    assert not {r["scaffold"] for r in tr} & {r["scaffold"] for r in va}
    assert not {r["parent"] for r in tr} & {r["parent"] for r in va}
    target = np.array(
        [
            [
                r["candidate_utilities"][0]["holdout_centered_rmse"]
                - s["holdout_centered_rmse"]
                for s in r["candidate_utilities"][1:]
            ]
            for r in tr
        ]
    )
    learner = ExtraTreesRegressor(
        n_estimators=128,
        min_samples_leaf=3,
        max_features=0.5,
        random_state=20261119,
        n_jobs=2,
    ).fit(np.array([vector(r["state"]) for r in tr]), target)
    records = []
    for ep in va:
        utility = learner.predict([vector(ep["state"])])[0]
        idx = sorted(range(10), key=lambda i: (-utility[i], semantic_id(PROGRAMS[i])))[
            :2
        ]
        learned = [PROGRAMS[i] for i in idx if utility[i] > 0]
        for policy, programs in [
            ("bayes_parent_H3", []),
            ("fixed_two", FIXED),
            ("learned_two", learned),
            ("cv_harmonics", [p for p in PROGRAMS if p["scope"] == "broad"]),
            ("cv_library", PROGRAMS),
        ]:
            selected = score(ep, programs)
            records.append(
                {
                    "molecule": ep["molecule"],
                    "scaffold": ep["scaffold"],
                    "policy": policy,
                    "internal_holdout_rmse": selected["holdout_centered_rmse"],
                    "selected": selected["program"],
                }
            )
    policies = sorted({r["policy"] for r in records})
    summary = {
        p: float(
            np.mean([r["internal_holdout_rmse"] for r in records if r["policy"] == p])
        )
        for p in policies
    }
    result = {
        "status": "actual Sage geometry-complete development validation, not fresh chemistry and not LLM performance",
        "train_parents": len({r["parent"] for r in tr}),
        "train_episodes": len(tr),
        "validation_parents": len({r["parent"] for r in va}),
        "summary": summary,
        "records": records,
        "scope": "same8support+8internalpoolholdout; program acceptance uses observed-onlyLOO; coefficients fitted to sums over all affected torsions; all targetQM alreadyretired",
        "fitting_archive_overlap": "These molecules belong to the published Sage fitting archive; this validates mechanics/headroom rather than new chemical accuracy",
    }
    Path(out).write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.train, a.validation, a.out)
