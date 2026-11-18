"""Development-only replay. No current molecule is a fresh test molecule."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smiles_service.repair.core import (
    Posterior,
    design,
    verified_scopes,
    select,
    rules,
    acquire,
)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def arrays(records):
    return np.array([r["angles_rad"][0] for r in records]), np.array(
        [r["qm_energy_kcal_mol"] - r["baseline_energy_kcal_mol"] for r in records]
    )


def read(data, n):
    return json.loads((Path(data) / "datasets" / f"{n}.json").read_text())


def make_library(data, names):
    lib = []
    for n in names:
        d = read(data, n)
        m = d["molecule"]
        a, y = arrays(d["pool"])
        try:
            scopes = verified_scopes(m["smiles"], m["torsions"][0])
        except ValueError:
            continue
        model = Posterior(a, y, list(range(1, 7)))
        lib.append(
            {
                "id": n,
                "scopes": scopes,
                "coefficients": model.mean.tolist(),
                "parent": d["parent_smiles"],
                "scaffold": d["scaffold"],
            }
        )
    return lib


def prior_function(lib, scopes):
    def prior(p):
        rows = [r for r in lib if r["scopes"][p["scope"]] == scopes[p["scope"]]]
        if not rows:
            return np.zeros(2 * len(p["harmonics"]))
        v = np.mean([r["coefficients"] for r in rows], axis=0)
        return np.array([v[2 * (h - 1) + j] for h in p["harmonics"] for j in (0, 1)])

    return prior


def prepare(data, out, limit):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    split = json.loads((Path(data) / "split.json").read_text())["splits"]
    lib = make_library(data, split["train"])
    dump(out / "library.json", lib)
    # Selection independent of outcomes; these are ALL retired development molecules.
    names = sorted(
        split["test"],
        key=lambda n: hashlib.sha256(("repair-pilot-v1:" + n).encode()).hexdigest(),
    )[:limit]
    states = {}
    selections = {}
    for n in names:
        d = read(data, n)
        m = d["molecule"]
        a, y = arrays(d["pool"])
        scopes = verified_scopes(m["smiles"], m["torsions"][0])
        ids = [r["id"] for r in d["pool"]]
        chosen = sorted(
            range(len(a)),
            key=lambda i: hashlib.sha256(("initial-v1:" + ids[i]).encode()).hexdigest(),
        )[:4]
        model = Posterior(a[chosen], y[chosen], [1, 2, 3])
        chosen += acquire(
            model, a, [i for i in range(len(a)) if i not in chosen], ids, 4
        )
        model = Posterior(a[chosen], y[chosen], [1, 2, 3])
        res = y[chosen] - model.predict(a[chosen])
        selections[n] = chosen
        states[n] = {
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
    dump(out / "states.json", states)
    dump(out / "initial-selections.json", selections)
    dump(
        out / "protocol.json",
        {
            "status": "exploratory development pilot, not fresh confirmation",
            "data": str(Path(data).resolve()),
            "training_names": split["train"],
            "evaluation_names": names,
            "initial_queries": 8,
            "budgets": [8, 10, 12, 14, 16],
            "llm_calls_per_task": 1,
            "maximum_llm_nontrivial_proposals": 2,
            "noise": 0.1,
            "alpha": 1.0,
            "acceptance_relative_loo_gain": 0.02,
            "scope": "QM-minus-UFF driven-instance residual; specialization selects training-only environment prior, not production force-field atom typing",
            "prior_library_sha256": hashlib.sha256(
                (out / "library.json").read_bytes()
            ).hexdigest(),
            "limitations": [
                "single acquisition seed",
                "single untuned open model",
                "old angle-policy LoRA deliberately not reused",
                "old cohorts retired",
                "prior fit uses training pool only",
                "no new QM or off-grid forces",
                "no production repaired packages",
                "pilot counts smaller than proposed paper protocol",
            ],
        },
    )
    print(
        "Prepared", len(states), "states and", len(lib), "training parents", flush=True
    )


def centered_rmse(y, p):
    e = np.array(y) - np.array(p)
    return float(np.sqrt(np.mean((e - e.mean()) ** 2)))


def replay(out):
    out = Path(out)
    proto = json.loads((out / "protocol.json").read_text())
    states = json.loads((out / "states.json").read_text())
    lib = json.loads((out / "library.json").read_text())
    initial = json.loads((out / "initial-selections.json").read_text())
    plans = (
        json.loads((out / "llm-plans.json").read_text())
        if (out / "llm-plans.json").exists()
        else {}
    )
    rows = []
    audit = {}
    for n, state in states.items():
        d = read(proto["data"], n)
        a, y = arrays(d["pool"])
        at, yt = arrays(d["test"])
        ids = [r["id"] for r in d["pool"]]
        prior = prior_function(lib, state["verified_scopes"])
        policies = {
            "bayes_fixed": [],
            "rules_two": [
                {"action": "revise", "scope": "broad", "harmonics": [1, 2]},
                {"action": "specialize", "scope": "local", "harmonics": [1, 2, 3, 4]},
            ],
            "cv_library": rules(),
        }
        if n in plans:
            policies["llm_two"] = plans[n]["programs"]
        for policy, programs in policies.items():
            chosen = initial[n].copy()
            errs = []
            steps = []
            for budget in proto["budgets"]:
                model, record = select(
                    programs,
                    state["smiles"],
                    state["torsion"],
                    a[chosen],
                    y[chosen],
                    prior,
                )
                error = centered_rmse(yt, model.predict(at))
                errs.append(error)
                steps.append(
                    {
                        "budget": budget,
                        "evaluation_centered_rmse": error,
                        "selected_ids": [ids[i] for i in chosen],
                        **record,
                    }
                )
                chosen += acquire(
                    model, a, [i for i in range(len(a)) if i not in chosen], ids, 2
                )
            rows.append(
                {
                    "molecule": n,
                    "scaffold": d["scaffold"],
                    "policy": policy,
                    "rmse_auc": float(
                        np.trapezoid(errs, x=proto["budgets"])
                        / (proto["budgets"][-1] - proto["budgets"][0])
                    ),
                    "final_rmse": errs[-1],
                }
            )
            audit[n + "::" + policy] = steps
    dump(out / "metrics.json", rows)
    dump(out / "replay-audit.json", audit)
    summary = {
        p: {
            "molecules": sum(r["policy"] == p for r in rows),
            "mean_rmse_auc": float(
                np.mean([r["rmse_auc"] for r in rows if r["policy"] == p])
            ),
        }
        for p in sorted({r["policy"] for r in rows})
    }
    summary[
        "interpretation"
    ] = "Development pilot only; CV enumerates ten proposals, rules/LLM at most two. Compare separately; CV is a stronger-library baseline, not equal proposal compute. Rule candidates are fixed before this replay; no LLM-specific tools."
    dump(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "replay"])
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=24)
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.data, args.out, args.limit)
    else:
        replay(args.out)
