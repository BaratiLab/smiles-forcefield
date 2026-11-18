"""Shared development episode construction: identical to the deployed v1 pilot."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from experiment import arrays, read, make_library, prior_function, dump
from smiles_service.repair.core import (
    Posterior,
    verified_scopes,
    acquire,
    rules,
    loo_score,
    semantic_id,
)


def support_indices(records, seed=0):
    a, _ = arrays(records)
    ids = [r["id"] for r in records]
    salt = "initial-v1:" if seed == 0 else f"initial-v1-seed-{seed}:"
    chosen = sorted(
        range(len(a)),
        key=lambda i: hashlib.sha256((salt + ids[i]).encode()).hexdigest(),
    )[:4]
    # Acquisition is covariance-only; do not even load unseen labels into this initializer.
    model = Posterior(a[chosen], np.zeros(4), [1, 2, 3])
    return chosen + acquire(
        model, a, [i for i in range(len(a)) if i not in chosen], ids, 4
    )


def state_for(d, lib, seed=0):
    m = d["molecule"]
    a, y = arrays(d["pool"])
    chosen = support_indices(d["pool"], seed)
    scopes = verified_scopes(m["smiles"], m["torsions"][0])
    model = Posterior(a[chosen], y[chosen], [1, 2, 3])
    res = y[chosen] - model.predict(a[chosen])
    state = {
        "schema_version": "torsion-repair-v1",
        "smiles": m["smiles"],
        "torsion": m["torsions"][0],
        "verified_scopes": scopes,
        "chemistry": m["chemistry"],
        "observations": [
            {
                "angle_rad": float(a[i]),
                "relative_correction_kcal_mol": float(y[i] - y[chosen].mean()),
                "residual_kcal_mol": float(res[j]),
            }
            for j, i in enumerate(chosen)
        ],
        "posterior_standard_deviations": np.sqrt(np.diag(model.cov)).tolist(),
        "training_analogue_counts": {
            s: sum(r["scopes"][s] == scopes[s] for r in lib) for s in scopes
        },
        "scope_meaning": "broad/local choose shared prior families. Output applies only to the driven torsion instance. Local fallback is zero prior if no training analogue exists.",
    }
    return state, chosen


def build(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    split = json.loads((Path(data) / "split.json").read_text())["splits"]
    lib = make_library(data, split["train"])
    dump(out / "library.json", lib)
    validation = sorted(
        split["validation"],
        key=lambda n: hashlib.sha256(
            ("repair-validation-v2:" + n).encode()
        ).hexdigest(),
    )[:12]
    records = {"train": [], "validation": []}
    candidates = [
        {"action": "keep", "scope": "broad", "harmonics": [1, 2, 3]}
    ] + rules()
    for group, names in [("train", split["train"]), ("validation", validation)]:
        for n in names:
            d = read(data, n)
            others = [r for r in lib if r["parent"] != d["parent_smiles"]]
            prior = prior_function(
                others,
                verified_scopes(d["molecule"]["smiles"], d["molecule"]["torsions"][0]),
            )
            for seed in range(3) if group == "train" else [0]:
                state, chosen = state_for(d, others, seed)
                a, y = arrays(d["pool"])
                hold = [i for i in range(len(a)) if i not in chosen]
                scores = []
                for p in candidates:
                    fit = Posterior(a[chosen], y[chosen], p["harmonics"], prior(p))
                    e = y[hold] - fit.predict(a[hold])
                    scores.append(
                        {
                            "program": p,
                            "holdout_centered_rmse": float(
                                np.sqrt(np.mean((e - e.mean()) ** 2))
                            ),
                            "support_loo_rmse": loo_score(
                                a[chosen], y[chosen], p["harmonics"], prior(p)
                            ),
                        }
                    )
                base = scores[0]
                eligible = [
                    r
                    for r in scores[1:]
                    if r["support_loo_rmse"] < 0.98 * base["support_loo_rmse"]
                    and r["holdout_centered_rmse"]
                    < 0.98 * base["holdout_centered_rmse"]
                ]
                best = sorted(
                    eligible,
                    key=lambda r: (
                        r["holdout_centered_rmse"],
                        len(r["program"]["harmonics"]),
                        semantic_id(r["program"]),
                    ),
                )[:2]
                records[group].append(
                    {
                        "molecule": n,
                        "parent": d["parent_smiles"],
                        "scaffold": d["scaffold"],
                        "seed": seed,
                        "state": state,
                        "support_indices": chosen,
                        "teacher_holdout_indices": hold,
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
    assert not {r["parent"] for r in records["train"]} & {
        r["parent"] for r in records["validation"]
    }
    assert not {r["scaffold"] for r in records["train"]} & {
        r["scaffold"] for r in records["validation"]
    }
    for group, rows in records.items():
        dump(out / f"{group}.json", rows)
    dump(
        out / "protocol.json",
        {
            "status": "development pilot; all cohorts retired",
            "data": str(Path(data).resolve()),
            "original_split_sha256": hashlib.sha256(
                (Path(data) / "split.json").read_bytes()
            ).hexdigest(),
            "training_parents": split["train"],
            "validation_parents": validation,
            "updates": 64,
            "checkpoints": [0, 32, 64],
            "seed": 20261119,
            "learning_rate": 0.0001,
            "adapter": {
                "num_layers": 2,
                "rank": 8,
                "scale": 8.0,
                "dropout": 0.0,
                "keys": ["self_attn.q_proj", "self_attn.v_proj"],
            },
            "selection": "lowest accepted-program centered residual RMSE on 12 validation parents at 8 support + 8 internal pool holdout, tie earliest; also retain best nonzero checkpoint for diagnostic comparison",
            "loss": "completion-only program-token cross entropy; training samples parents uniformly, seed episode cycles; no A-E target",
            "fresh_labels_used": False,
            "validation_labels": "internal pool holdout only; dataset test field is not read",
            "planned_evaluation": "24 existing pilot molecules, cached matched base proposals only after byte-identical state checks; single training seed, development only",
            "training_state_alignment": "same four hash-initial points + four covariance-based Bayes acquisitions as deployed v1; seed0 exactly identical; no target-parent prior retrieval",
            "numerical": "BF16 base floats, FP32 LoRA and loss; per-tensor finite checks before every optimizer step; clip norm1; full adapter and optimizer snapshots",
        },
    )
    print("Aligned episodes", {g: len(r) for g, r in records.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    build(a.data, a.out)
