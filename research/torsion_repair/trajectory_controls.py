"""Train-only whole-trajectory controls, with exact matched SFT exposures."""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sage_compare import vector
from sage_controls import matched_rows, FIXED, Mixture, Learned
from program_language import PROGRAMS
from trajectory_utility import Controller, rollout, subset, BUDGETS


def run(aligned, out):
    aligned = Path(aligned)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    load = lambda n: json.loads((aligned / (n + ".json")).read_text())
    proto = load("protocol")
    train = load("train")
    val = load("validation")
    language = load("language")
    data = {r["molecule"]: r for r in load("datasets")}
    used = matched_rows(train, proto)
    models = {}
    retrieval = {}
    for tag, rows in [("matched", used), ("full", train)]:
        x = [vector(r["state"]) for r in rows]
        models[tag] = ExtraTreesRegressor(
            n_estimators=128,
            min_samples_leaf=3,
            max_features=0.5,
            random_state=proto["seed"],
            n_jobs=2,
        ).fit(x, [r["plan_costs"] for r in rows])
        retrieval[tag] = Learned(rows, True)
    classifier = ExtraTreesClassifier(
        n_estimators=128,
        min_samples_leaf=3,
        max_features=0.5,
        random_state=proto["seed"],
        n_jobs=2,
    ).fit([vector(r["state"]) for r in used], [r["target_plan_index"] for r in used])
    costs = np.mean([r["plan_costs"] for r in used], axis=0)
    best = min(range(len(language)), key=lambda i: (costs[i], language[i]))
    (out / "training-global-plan.json").write_text(
        json.dumps(
            dict(
                selected=json.loads(language[best]),
                mean_training_auc=costs.tolist(),
                matched_episodes=[(r["molecule"], r["seed"]) for r in used],
            ),
            indent=2,
        )
    )
    records = []
    for ep in val:
        d = data[ep["molecule"]]
        y = np.array(d["qm_energy_kcal_mol"]) - d["sage_energy_kcal_mol"]
        mat = d["fourier_sum_matrices"]
        hold = d["retired_test_indices"]
        evalx = subset(mat, hold)
        policies = {
            "bayes": [],
            "fixed_two": FIXED,
            "global_matched": json.loads(language[best])["programs"],
            "labels_matched": json.loads(
                language[int(classifier.predict([vector(ep["state"])])[0])]
            )["programs"],
            "cv_library": PROGRAMS,
            "cv_harmonics": [p for p in PROGRAMS if p["scope"] == "broad"],
            "bma_two": FIXED,
            "bma_library": PROGRAMS,
        }
        for tag in models:
            pred = models[tag].predict([vector(ep["state"])])[0]
            i = min(range(len(language)), key=lambda i: (pred[i], language[i]))
            policies["learned_" + tag] = json.loads(language[i])["programs"]
            policies["retrieval_" + tag] = retrieval[tag].retrieve(ep["state"])
        ctrl = Controller(mat, y, ep["state"])
        for name, programs in policies.items():
            if name.startswith("bma"):
                ids = list(ep["support_indices"])
                errors = []
                trace = []
                for budget in BUDGETS:
                    m = Mixture(subset(mat, ids), y[ids], programs)
                    errors.append(float(np.std(y[hold] - m.predict(evalx))))
                    trace.append(dict(budget=budget, selected_indices=list(ids)))
                    if budget < 16:
                        ids += m.acquire(
                            mat, [i for i in d["pool_indices"] if i not in ids], 2
                        )
                result = dict(
                    auc=float(np.trapezoid(errors, dx=2) / 8),
                    errors=errors,
                    trace=trace,
                )
            else:
                result = rollout(
                    ctrl,
                    programs,
                    ep["support_indices"],
                    d["pool_indices"],
                    evalx,
                    y[hold],
                )
            records.append(
                dict(
                    molecule=ep["molecule"],
                    scaffold=ep["scaffold"],
                    policy=name,
                    **result,
                )
            )
    summary = {
        p: float(np.mean([r["auc"] for r in records if r["policy"] == p]))
        for p in sorted(policies)
    }
    (out / "scores.json").write_text(
        json.dumps(
            dict(summary=summary, records=records, held_out_evaluated=False), indent=2
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aligned")
    p.add_argument("--out")
    a = p.parse_args()
    run(a.aligned, a.out)
