"""Chemistry-process validation for MLX-generated programs; support labels only."""
import argparse, json
from pathlib import Path
from experiment import read, arrays, prior_function, dump
from smiles_service.repair.core import select
import numpy as np


def score(episodes, library, data, plans, out):
    rows = json.loads(Path(episodes).read_text())
    lib = json.loads(Path(library).read_text())
    plans = json.loads(Path(plans).read_text())
    result = []
    for r in rows:
        d = read(data, r["molecule"])
        a, y = arrays(d["pool"])
        chosen = r["support_indices"]
        hold = r["teacher_holdout_indices"]
        p = prior_function(lib, r["state"]["verified_scopes"])
        model, audit = select(
            plans[r["molecule"]]["programs"],
            r["state"]["smiles"],
            r["state"]["torsion"],
            a[chosen],
            y[chosen],
            p,
        )
        e = y[hold] - model.predict(a[hold])
        result.append(
            {
                "molecule": r["molecule"],
                "rmse": float(np.sqrt(np.mean((e - e.mean()) ** 2))),
                "audit": audit,
            }
        )
    dump(
        out,
        {
            "mean_validation_rmse": float(np.mean([r["rmse"] for r in result])),
            "molecules": result,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--plans", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    score(a.episodes, a.library, a.data, a.plans, a.out)
