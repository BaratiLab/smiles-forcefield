"""Build development-only program supervision from training parents only."""
import argparse, json
from pathlib import Path
import numpy as np
from experiment import read, arrays, make_library, prior_function, dump
from llm_mlx import SYSTEM
from smiles_service.repair.core import (
    Posterior,
    verified_scopes,
    rules,
    loo_score,
    semantic_id,
)


def build(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    split = json.loads((Path(data) / "split.json").read_text())["splits"]
    names = split["train"]
    lib = make_library(data, names)
    episodes = []
    training = []
    for name in names:
        d = read(data, name)
        m = d["molecule"]
        a, y = arrays(d["pool"])
        scopes = verified_scopes(m["smiles"], m["torsions"][0])
        others = [r for r in lib if r["parent"] != d["parent_smiles"]]
        prior = prior_function(others, scopes)
        for seed in (0, 1):
            ids = np.random.default_rng(seed).permutation(len(a))
            support = ids[:8]
            holdout = ids[8:]
            model = Posterior(a[support], y[support], [1, 2, 3])
            res = y[support] - model.predict(a[support])
            state = {
                "schema_version": "torsion-repair-v1",
                "smiles": m["smiles"],
                "torsion": m["torsions"][0],
                "verified_scopes": scopes,
                "chemistry": m["chemistry"],
                "observations": [
                    {
                        "angle_rad": float(a[i]),
                        "relative_correction_kcal_mol": float(y[i] - y[support].mean()),
                        "residual_kcal_mol": float(res[j]),
                    }
                    for j, i in enumerate(support)
                ],
                "posterior_standard_deviations": np.sqrt(np.diag(model.cov)).tolist(),
                "training_analogue_counts": {
                    s: sum(r["scopes"][s] == scopes[s] for r in others) for s in scopes
                },
                "scope_meaning": "broad/local choose shared prior families. Output applies only to the driven torsion instance. Local fallback is zero prior if no training analogue exists.",
            }
            candidates = [
                {"action": "keep", "scope": "broad", "harmonics": [1, 2, 3]}
            ] + rules()
            scores = []
            for p in candidates:
                fit = Posterior(a[support], y[support], p["harmonics"], prior(p))
                e = y[holdout] - fit.predict(a[holdout])
                loss = float(np.sqrt(np.mean((e - e.mean()) ** 2)))
                scores.append(
                    {
                        "program": p,
                        "holdout_centered_rmse": loss,
                        "support_loo_rmse": loo_score(
                            a[support], y[support], p["harmonics"], prior(p)
                        ),
                    }
                )
            baseline = scores[0]
            eligible = [
                r
                for r in scores[1:]
                if r["support_loo_rmse"] < 0.98 * baseline["support_loo_rmse"]
                and r["holdout_centered_rmse"]
                < 0.98 * baseline["holdout_centered_rmse"]
            ]
            best = sorted(
                eligible,
                key=lambda r: (
                    r["holdout_centered_rmse"],
                    len(r["program"]["harmonics"]),
                    semantic_id(r["program"]),
                ),
            )[:2]
            answer = {
                "programs": [r["program"] for r in best]
                or [{"action": "abstain", "scope": "broad", "harmonics": [1, 2, 3]}]
            }
            episodes.append(
                {
                    "parent": d["parent_smiles"],
                    "molecule": name,
                    "scaffold": d["scaffold"],
                    "seed": seed,
                    "support_ids": [d["pool"][i]["id"] for i in support],
                    "internal_teacher_holdout_ids": [
                        d["pool"][i]["id"] for i in holdout
                    ],
                    "candidate_utilities": scores,
                    "answer": answer,
                }
            )
            training.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": json.dumps(state, sort_keys=True)},
                        {"role": "assistant", "content": json.dumps(answer)},
                    ]
                }
            )
    dump(out / "episodes.json", episodes)
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in training))
    dump(
        out / "manifest.json",
        {
            "status": "development supervision, no repair fine-tuning executed",
            "independent_parents": len(names),
            "episodes": len(episodes),
            "source_split": "train only",
            "target_parent_excluded_from_prior_library": True,
            "old_validation_and_test_labels_used": False,
            "internal_teacher_holdouts": "remaining training-parent pool points; never the dataset test field",
            "limitation": "190 correlated episodes are not 190 independent molecules; support states use two random seeds, pilot uses fixed Bayes initialization; align episode distribution before paper training",
        },
    )
    print("Teacher episodes", len(episodes))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    build(a.data, a.out)
