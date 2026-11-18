"""Select non-LLM development comparators on the same checkpoint-selection task."""
import argparse, json
from pathlib import Path
import numpy as np
from controls import Learned, Mixture, FIXED, PROGRAMS
from experiment import read, arrays, prior_function, dump
from smiles_service.repair.core import select


def run(aligned, out):
    aligned = Path(aligned)
    proto = json.loads((aligned / "protocol.json").read_text())
    episodes = json.loads((aligned / "train.json").read_text())
    validation = json.loads((aligned / "validation.json").read_text())
    lib = json.loads((aligned / "library.json").read_text())
    learner = Learned(episodes)
    records = []
    for row in validation:
        d = read(proto["data"], row["molecule"])
        a, y = arrays(d["pool"])
        chosen = row["support_indices"]
        hold = row["teacher_holdout_indices"]
        state = row["state"]
        prior = prior_function(lib, state["verified_scopes"])
        policies = {
            "bayes_fixed": [],
            "fixed_two": FIXED,
            "learned_two": learner.predict(state),
            "retrieval_two": learner.retrieve(state),
            "bma_two": FIXED,
            "cv_library": PROGRAMS,
            "bma_library": PROGRAMS,
        }
        for name, programs in policies.items():
            if name.startswith("bma"):
                model = Mixture(programs, a[chosen], y[chosen], prior)
            else:
                model, _ = select(
                    programs,
                    state["smiles"],
                    state["torsion"],
                    a[chosen],
                    y[chosen],
                    prior,
                )
            error = y[hold] - model.predict(a[hold])
            records.append(
                {
                    "molecule": row["molecule"],
                    "scaffold": row["scaffold"],
                    "policy": name,
                    "rmse": float(np.std(error)),
                }
            )
    summary = {
        p: float(np.mean([r["rmse"] for r in records if r["policy"] == p]))
        for p in policies
    }
    eligible = ["fixed_two", "learned_two", "retrieval_two", "bma_two"]
    best = min(eligible, key=lambda p: (summary[p], p))
    dump(
        out,
        {
            "status": "development validation at8observed+8internalpoolholdout; same task as LoRA checkpoint selection",
            "summary": summary,
            "strongest_matched_budget_non_llm": best,
            "candidate_counts": "matched policies at most2 nontrivial; library references10",
            "records": records,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aligned", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.aligned, a.out)
