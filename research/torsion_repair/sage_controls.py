"""Validation controls for actual Sage sums; same observed data and proposal cap."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import ExtraTreesRegressor
from rdkit import Chem, DataStructs
from sage_compare import vector
from sage_runtime import Gaussian, BASE, select, subset, columns
from program_language import PROGRAMS, answers
from smiles_service.repair.core import semantic_id
from experiment import dump

FIXED = [
    {"action": "revise", "scope": "broad", "harmonics": [1, 2]},
    {"action": "specialize", "scope": "local", "harmonics": [1, 2, 3, 4]},
]


class Learned:
    def __init__(self, rows, labels_only=False):
        self.rows = rows
        self.labels_only = labels_only
        if labels_only:
            target = [
                [
                    float(
                        any(
                            a["action"] in ("revise", "specialize")
                            and a["scope"] == p["scope"]
                            and a["harmonics"] == p["harmonics"]
                            for a in r["answer"]["programs"]
                        )
                    )
                    for p in PROGRAMS
                ]
                for r in rows
            ]
        else:
            target = [
                [
                    r["candidate_utilities"][0]["holdout_centered_rmse"]
                    - v["holdout_centered_rmse"]
                    for v in r["candidate_utilities"][1:]
                ]
                for r in rows
            ]
        self.model = ExtraTreesRegressor(
            n_estimators=128,
            min_samples_leaf=3,
            max_features=0.5,
            random_state=20261120,
            n_jobs=2,
        ).fit([vector(r["state"]) for r in rows], target)
        self.fps = [
            Chem.RDKFingerprint(Chem.MolFromSmiles(r["state"]["smiles"])) for r in rows
        ]

    def predict(self, state):
        utility = self.model.predict([vector(state)])[0]
        order = sorted(
            range(10),
            key=lambda i: (-utility[i], json.dumps(PROGRAMS[i], sort_keys=True)),
        )[:2]
        threshold = 0.5 if self.labels_only else 0
        return [PROGRAMS[i] for i in order if utility[i] > threshold]

    def retrieve(self, state):
        fp = Chem.RDKFingerprint(Chem.MolFromSmiles(state["smiles"]))
        i = min(
            range(len(self.rows)),
            key=lambda j: (
                -DataStructs.TanimotoSimilarity(fp, self.fps[j]),
                self.rows[j]["molecule"],
                self.rows[j]["seed"],
            ),
        )
        return self.rows[i]["answer"]["programs"]


class Mixture:
    def __init__(self, matrices, y, programs):
        unique = {(p["scope"], tuple(p["harmonics"])): p for p in [BASE] + programs}
        self.models = []
        scores = []
        self.programs = [unique[k] for k in sorted(unique)]
        q = np.linalg.qr(np.column_stack([np.ones(len(y)), np.eye(len(y))[:, :-1]]))[0][
            :, 1:
        ]
        for p in self.programs:
            m = Gaussian(matrices, y, p)
            self.models.append(m)
            x = q.T @ m.x
            res = q.T @ y
            c = 0.01 * np.eye(len(y) - 1) + x @ x.T
            sign, ld = np.linalg.slogdet(c)
            if sign <= 0:
                raise ValueError("Invalid model evidence")
            scores.append(
                -0.5
                * (
                    res @ np.linalg.solve(c, res)
                    + ld
                    + (len(y) - 1) * np.log(2 * np.pi)
                )
            )
        self.weights = np.exp(scores - logsumexp(scores))

    def predict(self, matrices):
        return self.weights @ np.array([m.predict(matrices) for m in self.models])

    def acquire(self, matrices, available, batch=2):
        pred = np.array([m.predict(matrices) for m in self.models])
        mean = self.weights @ pred
        disagreement = self.weights @ ((pred - mean) ** 2)
        xs = [
            np.asarray(matrices[m.program["scope"]])[:, columns(m.program)] - m.xmean
            for m in self.models
        ]
        covs = [m.cov.copy() for m in self.models]
        remaining = list(available)
        chosen = []
        for _ in range(min(batch, len(remaining))):
            score = disagreement.copy()
            for w, x, c in zip(self.weights, xs, covs):
                score += w * np.einsum("ij,jk,ik->i", x, c, x)
            best = min(remaining, key=lambda i: (-float(score[i]), i))
            chosen.append(best)
            remaining.remove(best)
            for x, c in zip(xs, covs):
                v = c @ x[best]
                c -= np.outer(v, v) / (0.01 + x[best] @ v)
        return chosen


def matched_rows(rows, protocol):
    names = sorted({r["molecule"] for r in rows})
    schedule = np.random.default_rng(protocol["seed"]).permutation(names).tolist()
    lookup = {(r["molecule"], r["seed"]): r for r in rows}
    return [lookup[name, i % 3] for i, name in enumerate(schedule)]


def build_proposers(rows, protocol):
    used = matched_rows(rows, protocol)
    return {
        "full": Learned(rows),
        "matched": Learned(used),
        "labels": Learned(used, True),
    }


def proposals(state, learners):
    return {
        "bayes": [],
        "fixed_two": FIXED,
        "learned_matched": learners["matched"].predict(state),
        "labels_matched": learners["labels"].predict(state),
        "retrieval_matched": learners["matched"].retrieve(state),
        "learned_full": learners["full"].predict(state),
        "retrieval_full": learners["full"].retrieve(state),
        "cv_harmonics": [p for p in PROGRAMS if p["scope"] == "broad"],
        "cv_library": PROGRAMS,
        "bma_two": FIXED,
        "bma_library": PROGRAMS,
    }


def constant_plan(rows):
    """Posthoc diagnostic; choose solely on matched training episode utilities."""
    rankings = []
    for text in answers():
        programs = json.loads(text)["programs"]
        costs = []
        for row in rows:
            base = row["candidate_utilities"][0]
            lookup = {
                (v["program"]["scope"], tuple(v["program"]["harmonics"])): v
                for v in row["candidate_utilities"][1:]
            }
            choices = [base]
            for p in programs:
                if p["action"] in ("keep", "abstain"):
                    continue
                candidate = lookup[p["scope"], tuple(p["harmonics"])]
                if candidate["support_loo_rmse"] < 0.98 * base["support_loo_rmse"]:
                    choices.append(candidate)
            selected = min(
                choices,
                key=lambda r: (
                    r["support_loo_rmse"],
                    len(r["program"]["harmonics"]),
                    semantic_id(r["program"]),
                ),
            )
            costs.append(selected["holdout_centered_rmse"])
        rankings.append(
            {"programs": programs, "mean_training_rmse": float(np.mean(costs))}
        )
    rankings.sort(
        key=lambda r: (
            r["mean_training_rmse"],
            json.dumps(r["programs"], sort_keys=True),
        )
    )
    return rankings[0]["programs"], rankings


def run(aligned, out, plans_path=None, secondary=False):
    aligned = Path(aligned)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    proto = json.loads((aligned / "protocol.json").read_text())
    train = json.loads((aligned / "train.json").read_text())
    val = json.loads((aligned / "validation.json").read_text())
    data = {
        r["molecule"]: r for r in json.loads((aligned / "datasets.json").read_text())
    }
    learners = build_proposers(train, proto)
    global_programs, global_rankings = constant_plan(matched_rows(train, proto))
    dump(
        out / "training-global-plan.json",
        {
            "status": "supplemental diagnostic added after partial48-step outputs; recipe selected solely on95training episodes, no validation scores in selection",
            "selected": global_programs,
            "rankings": global_rankings,
        },
    )
    plans = json.loads(Path(plans_path).read_text()) if plans_path else {}
    rows = []
    audits = {}
    curve_rows = []
    for ep in val:
        d = data[ep["molecule"]]
        y = np.array(d["qm_energy_kcal_mol"]) - d["sage_energy_kcal_mol"]
        matrices = d["fourier_sum_matrices"]
        chosen = ep["support_indices"]
        hold = ep["internal_teacher_holdout_indices"]
        policies = proposals(ep["state"], learners)
        policies["global_matched"] = global_programs
        if plans:
            policies["llm"] = plans[ep["molecule"]]["programs"]
        fixed = list(chosen)
        while len(fixed) < 16:
            model = Gaussian(subset(matrices, fixed), y[fixed], BASE)
            fixed += model.acquire(
                matrices, [i for i in d["pool_indices"] if i not in fixed], 2
            )
        for policy, programs in policies.items():
            if policy.startswith("bma"):
                model = Mixture(subset(matrices, chosen), y[chosen], programs)
                audit = {"weights": model.weights.tolist(), "programs": model.programs}
            else:
                model, audit = select(
                    programs, ep["state"], subset(matrices, chosen), y[chosen]
                )
            e = y[hold] - model.predict(subset(matrices, hold))
            rows.append(
                {
                    "molecule": ep["molecule"],
                    "scaffold": ep["scaffold"],
                    "policy": policy,
                    "rmse": float(np.std(e)),
                }
            )
            audits[ep["molecule"] + "::" + policy] = audit
            if secondary:
                for mode in ("adaptive", "fixed_queries"):
                    ids = list(chosen)
                    errors = []
                    for budget in (8, 10, 12, 14, 16):
                        if mode == "fixed_queries":
                            ids = fixed[:budget]
                        if policy.startswith("bma"):
                            m = Mixture(subset(matrices, ids), y[ids], programs)
                        else:
                            m, _ = select(
                                programs, ep["state"], subset(matrices, ids), y[ids]
                            )
                        test = d["retired_test_indices"]
                        error = y[test] - m.predict(subset(matrices, test))
                        errors.append(float(np.std(error)))
                        ids += m.acquire(
                            matrices, [i for i in d["pool_indices"] if i not in ids], 2
                        )
                    curve_rows.append(
                        {
                            "molecule": ep["molecule"],
                            "scaffold": ep["scaffold"],
                            "policy": policy,
                            "mode": mode,
                            "rmse_auc": float(np.trapezoid(errors, dx=2) / 8),
                            "budgets": [8, 10, 12, 14, 16],
                            "rmse_curve": errors,
                        }
                    )
    summary = {
        p: float(np.mean([r["rmse"] for r in rows if r["policy"] == p]))
        for p in sorted({r["policy"] for r in rows})
    }
    eligible = [
        "fixed_two",
        "learned_matched",
        "labels_matched",
        "retrieval_matched",
        "bma_two",
    ]
    best = min(eligible, key=lambda p: (summary[p], p))
    dump(
        out / "primary.json",
        {
            "summary": summary,
            "strongest_matched_control": best,
            "strongest_extended_matched_control": min(
                eligible + ["global_matched"], key=lambda p: (summary[p], p)
            ),
            "global_control_status": "supplemental posthoc diagnostic, training-only recipe selection; original primary comparison unchanged",
            "records": rows,
            "status": "development validation only,12scaffolds,8support+8internalpoolholdout",
            "matched_supervision": "95 exact parent/seed episodes in the frozen model schedule; full references285episodes",
            "target_formats": "regret learner gets cost vectors; labels learner/retrieval get the same teacher answer programs as SFT",
            "feature_information": "all numerical feature transforms use only the same observed Sage state",
        },
    )
    dump(out / "audits.json", audits)
    if secondary:
        dump(out / "secondary-curves.json", curve_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aligned", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--plans")
    p.add_argument("--secondary", action="store_true")
    a = p.parse_args()
    run(a.aligned, a.out, a.plans, a.secondary)
