"""Primary full-trajectory validation; evaluator outcomes never reach selection."""
import argparse, json
from pathlib import Path
import numpy as np
from trajectory_utility import Controller, rollout, subset


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
        hold = d["retired_test_indices"]
        result = rollout(
            Controller(d["fourier_sum_matrices"], y, r["state"]),
            plans[r["molecule"]]["programs"],
            r["support_indices"],
            d["pool_indices"],
            subset(d["fourier_sum_matrices"], hold),
            y[hold],
        )
        records.append(dict(molecule=r["molecule"], scaffold=r["scaffold"], **result))
    Path(out).write_text(
        json.dumps(
            dict(
                mean_validation_auc=float(np.mean([r["auc"] for r in records])),
                molecules=records,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aligned")
    p.add_argument("--plans")
    p.add_argument("--out")
    a = p.parse_args()
    score(a.aligned, a.plans, a.out)
