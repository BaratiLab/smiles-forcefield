"""Score generated Sage programs; compiler/acceptance never see evaluation labels."""
import argparse, json
from pathlib import Path
import numpy as np
from experiment import dump
from sage_runtime import select, subset


def score(aligned, plans, out):
    aligned = Path(aligned)
    rows = json.loads((aligned / "validation.json").read_text())
    datasets = {
        r["molecule"]: r for r in json.loads((aligned / "datasets.json").read_text())
    }
    plans = json.loads(Path(plans).read_text())
    records = []
    for r in rows:
        d = datasets[r["molecule"]]
        y = np.array(d["qm_energy_kcal_mol"]) - d["sage_energy_kcal_mol"]
        matrices = d["fourier_sum_matrices"]
        ids = r["support_indices"]
        hold = r["internal_teacher_holdout_indices"]
        model, audit = select(
            plans[r["molecule"]]["programs"], r["state"], subset(matrices, ids), y[ids]
        )
        error = y[hold] - model.predict(subset(matrices, hold))
        records.append(
            {
                "molecule": r["molecule"],
                "scaffold": r["scaffold"],
                "rmse": float(np.std(error)),
                "audit": audit,
            }
        )
    dump(
        out,
        {
            "mean_validation_rmse": float(np.mean([r["rmse"] for r in records])),
            "molecules": records,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aligned", required=True)
    p.add_argument("--plans", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    score(a.aligned, a.plans, a.out)
