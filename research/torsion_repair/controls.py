"""Stronger development controls, sharing observations, priors and compiler."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import ExtraTreesRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from experiment import read, arrays, prior_function, dump
from aligned import state_for
from smiles_service.repair.core import Posterior, design, rules, select, acquire

BASE = {"action": "keep", "scope": "broad", "harmonics": [1, 2, 3]}
FIXED = [
    {"action": "revise", "scope": "broad", "harmonics": [1, 2]},
    {"action": "specialize", "scope": "local", "harmonics": [1, 2, 3, 4]},
]
PROGRAMS = rules()


def features(state):
    mol = Chem.MolFromSmiles(state["smiles"])
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128).GetFingerprint(
        mol
    )
    bits = np.zeros(128)
    DataStructs.ConvertToNumpyArray(fp, bits)
    obs = state["observations"]
    a = np.array([r["angle_rad"] for r in obs])
    y = np.array([r["relative_correction_kcal_mol"] for r in obs])
    res = np.array([r["residual_kcal_mol"] for r in obs])
    fit = Posterior(a, y, list(range(1, 7)))
    numeric = [
        np.std(y),
        np.ptp(y),
        np.std(res),
        np.ptp(res),
        *fit.mean,
        *state["posterior_standard_deviations"],
        np.log1p(state["training_analogue_counts"]["broad"]),
        np.log1p(state["training_analogue_counts"]["local"]),
    ]
    return np.r_[bits, numeric]


class Learned:
    def __init__(self, episodes, labels_only=False):
        x = np.array([features(r["state"]) for r in episodes])
        targets = []
        for r in episodes:
            base = r["candidate_utilities"][0]["holdout_centered_rmse"]
            targets.append(
                [
                    base - v["holdout_centered_rmse"]
                    for v in r["candidate_utilities"][1:]
                ]
            )
        if labels_only:
            targets = [
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
                for r in episodes
            ]
        self.labels_only = labels_only
        self.model = ExtraTreesRegressor(
            n_estimators=128,
            min_samples_leaf=3,
            max_features=0.5,
            random_state=20261119,
            n_jobs=2,
        ).fit(x, targets)
        self.episodes = episodes
        self.fps = [
            Chem.RDKFingerprint(Chem.MolFromSmiles(r["state"]["smiles"]))
            for r in episodes
        ]

    def predict(self, state):
        utility = self.model.predict([features(state)])[0]
        indices = sorted(
            range(len(PROGRAMS)),
            key=lambda i: (-utility[i], json.dumps(PROGRAMS[i], sort_keys=True)),
        )[:2]
        return (
            [PROGRAMS[i] for i in indices if utility[i] >= 0.5]
            if self.labels_only
            else [PROGRAMS[i] for i in indices if utility[i] > 0]
        )

    def retrieve(self, state):
        fp = Chem.RDKFingerprint(Chem.MolFromSmiles(state["smiles"]))
        i = min(
            range(len(self.fps)),
            key=lambda j: (
                -DataStructs.TanimotoSimilarity(fp, self.fps[j]),
                self.episodes[j]["molecule"],
                self.episodes[j]["seed"],
            ),
        )
        return self.episodes[i]["answer"]["programs"]


class Mixture:
    def __init__(self, programs, a, y, prior):
        # De-duplicate the baseline representation, independently of action wording.
        unique = {(p["scope"], tuple(p["harmonics"])): p for p in [BASE] + programs}
        self.models = []
        scores = []
        self.programs = [unique[k] for k in sorted(unique)]
        # An orthonormal contrast removes the free molecular energy zero.
        q = np.linalg.qr(np.column_stack([np.ones(len(y)), np.eye(len(y))[:, :-1]]))[0][
            :, 1:
        ]
        for p in self.programs:
            mu = prior(p)
            model = Posterior(a, y, p["harmonics"], mu)
            self.models.append(model)
            x = q.T @ design(a, p["harmonics"])
            res = q.T @ y - x @ mu
            c = 0.01 * np.eye(len(y) - 1) + x @ x.T
            sign, ld = np.linalg.slogdet(c)
            if sign <= 0:
                raise ValueError("Invalid model evidence covariance")
            scores.append(
                -0.5
                * (
                    res @ np.linalg.solve(c, res)
                    + ld
                    + (len(y) - 1) * np.log(2 * np.pi)
                )
            )
        self.weights = np.exp(np.array(scores) - logsumexp(scores))

    def predict(self, a):
        return self.weights @ np.array([m.predict(a) for m in self.models])

    def acquire(self, a, available, ids, batch=2):
        covs = [m.cov.copy() for m in self.models]
        xs = [design(a, m.hs) - m.xmean for m in self.models]
        pred = np.array([m.predict(a) for m in self.models])
        mean = self.weights @ pred
        disagreement = self.weights @ ((pred - mean) ** 2)
        remaining = list(available)
        chosen = []
        for _ in range(min(batch, len(remaining))):
            score = disagreement.copy()
            for w, x, cov in zip(self.weights, xs, covs):
                score += w * np.einsum("ij,jk,ik->i", x, cov, x)
            best = min(remaining, key=lambda i: (-score[i], ids[i]))
            chosen.append(best)
            remaining.remove(best)
            for x, cov in zip(xs, covs):
                v = cov @ x[best]
                cov -= np.outer(v, v) / (0.01 + x[best] @ v)
        return chosen


def run(data, aligned, out, names_path, plans_path=None, history_path=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lib = json.loads((Path(aligned) / "library.json").read_text())
    episodes = json.loads((Path(aligned) / "train.json").read_text())
    learner = Learned(episodes)
    matched = None
    if history_path:
        used = {
            (r["molecule"], r["episode_seed"])
            for r in map(json.loads, Path(history_path).read_text().splitlines())
        }
        matching = [r for r in episodes if (r["molecule"], r["seed"]) in used]
        assert len(matching) == len(used)
        matched = Learned(matching)
        labels_matched = Learned(matching, labels_only=True)
    names = json.loads(Path(names_path).read_text())
    plans = json.loads(Path(plans_path).read_text()) if plans_path else {}
    rows = []
    audits = {}
    for index, n in enumerate(names, 1):
        d = read(data, n)
        state, initial = state_for(d, lib)
        a, y = arrays(d["pool"])
        at, yt = arrays(d["test"])
        prior = prior_function(lib, state["verified_scopes"])
        ids = [r["id"] for r in d["pool"]]
        policies = {
            "bayes_fixed": [],
            "fixed_two": FIXED,
            "cv_library": PROGRAMS,
            "cv_harmonics": [p for p in PROGRAMS if p["scope"] == "broad"],
            "specialization_only": [
                {"action": "specialize", "scope": "local", "harmonics": [1, 2, 3]}
            ],
            "learned_two": learner.predict(state),
            "retrieval_two": learner.retrieve(state),
            "bma_two": FIXED,
            "bma_library": PROGRAMS,
        }
        if matched is not None:
            policies["learned_matched_two"] = matched.predict(state)
            policies["retrieval_matched_two"] = matched.retrieve(state)
            policies["labels_matched_two"] = labels_matched.predict(state)
        if n in plans:
            policies["llm_two"] = plans[n]["programs"]
        fixed = initial.copy()
        while len(fixed) < 16:
            base = Posterior(a[fixed], y[fixed], [1, 2, 3], prior(BASE))
            fixed += acquire(
                base, a, [i for i in range(len(a)) if i not in fixed], ids, 2
            )
        for policy, programs in policies.items():
            for mode in ("adaptive", "fixed_queries"):
                chosen = initial.copy()
                errs = []
                steps = []
                for budget in [8, 10, 12, 14, 16]:
                    if mode == "fixed_queries":
                        chosen = fixed[:budget]
                    if policy.startswith("bma_"):
                        model = Mixture(programs, a[chosen], y[chosen], prior)
                        record = {
                            "programs": model.programs,
                            "weights": model.weights.tolist(),
                            "selection": "Gaussian marginal evidence after projecting out molecule offset; uniform prior over unique representations",
                        }
                    else:
                        model, record = select(
                            programs,
                            state["smiles"],
                            state["torsion"],
                            a[chosen],
                            y[chosen],
                            prior,
                        )
                    e = yt - model.predict(at)
                    error = float(np.sqrt(np.mean((e - e.mean()) ** 2)))
                    errs.append(error)
                    steps.append(
                        {
                            "budget": budget,
                            "error": error,
                            "selected_ids": [ids[i] for i in chosen],
                            **record,
                        }
                    )
                    if mode == "adaptive":
                        available = [i for i in range(len(a)) if i not in chosen]
                        chosen += (
                            model.acquire(a, available, ids, 2)
                            if policy.startswith("bma_")
                            else acquire(model, a, available, ids, 2)
                        )
                key = n + "::" + policy + "::" + mode
                audits[key] = steps
                rows.append(
                    {
                        "molecule": n,
                        "scaffold": d["scaffold"],
                        "policy": policy,
                        "mode": mode,
                        "rmse_auc": float(np.trapezoid(errs, dx=2) / 8),
                        "final_rmse": errs[-1],
                    }
                )
        print("CONTROL", index, len(names), n, flush=True)
    dump(out / "metrics.json", rows)
    dump(out / "audit.json", audits)
    summary = {
        p
        + "::"
        + m: float(
            np.mean(
                [r["rmse_auc"] for r in rows if r["policy"] == p and r["mode"] == m]
            )
        )
        for p in sorted({r["policy"] for r in rows})
        for m in ("adaptive", "fixed_queries")
    }
    dump(out / "summary.json", summary)
    dump(
        out / "protocol.json",
        {
            "status": "retired development cohort",
            "parents": len(names),
            "training_parents": len({r["molecule"] for r in episodes}),
            "supervision": "full references: 285 aligned training-only episodes; matched policies use exactly the model training-history episodes",
            "matched_episodes": len(matching) if matched is not None else None,
            "target_formats": "regret predictors use full utility vectors from the same QM data; labels_matched uses only membership in the exact teacher answer programs, threshold0.5. All have at most2 proposals.",
            "features": "ExtraTrees derives an H6 fit and summary features from observed support data, not additional QM labels. LLM sees the original state; representations differ.",
            "learned": "ExtraTrees 128 trees, leaf3, maxfeatures .5, seed20261119; regress internal-holdout RMSE improvements over all 10 programs, propose at most two positive predictions",
            "retrieval": "nearest training RDK fingerprint, deterministic molecule/episode tie break; transfer its at-most-two teacher programs",
            "fixed_two": "broad H2 and local H4",
            "bma": "Gaussian marginal evidence, intercept contrast; acquisition uses mixture predictive variance, not exact cross-model mutual information",
            "candidate_budget": "Fixed/learned/retrieval/LLM at most two nontrivial programs; exhaustive CV/BMA ten candidates are larger-library references",
            "init": "exact same seed0 eight-point support as v1 pilot",
            "fixed_query_ablation": "all methods reuse the baseline Bayes trajectory; adaptive recomputes queries within each model",
            "evaluation": "old test conformers, development only; no new QM and no generalization claim",
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--aligned", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--plans")
    p.add_argument("--history")
    a = p.parse_args()
    run(a.data, a.aligned, a.out, a.names, a.plans, a.history)
