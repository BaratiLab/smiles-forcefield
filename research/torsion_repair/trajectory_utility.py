"""Full-horizon Sage plan utilities, with label-blind numerical decisions.

The evaluator's holdout energies are never supplied to the model/controller.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from program_language import answers, PROGRAMS
from sage_runtime import Gaussian, BASE, subset, loo
from smiles_service.repair.core import semantic_id, compile_program

BUDGETS = (8, 10, 12, 14, 16)


class Controller:
    def __init__(self, matrices, pool_y, state):
        self.matrices = matrices
        self.y = np.asarray(pool_y)
        self.state = state
        self.cache = {}
        self.programs = [BASE] + PROGRAMS
        for p in self.programs:
            compile_program(p, state["smiles"], state["driven_torsion"])

    def select(self, programs, indices):
        # No evaluator labels enter this method or the acquisition rule.
        key = tuple(sorted(indices))
        observed = subset(self.matrices, indices)
        y = self.y[indices]
        if key not in self.cache:
            self.cache[key] = {}
        cache = self.cache[key]
        rows = []
        for p in [BASE] + programs:
            semantic = (p["scope"], tuple(p["harmonics"]))
            if semantic not in cache:
                cache[semantic] = (Gaussian(observed, y, p), loo(observed, y, p))
            model, score = cache[semantic]
            rows.append((p, model, score))
        base = rows[0]
        best = min(
            rows, key=lambda r: (r[2], len(r[0]["harmonics"]), semantic_id(r[0]))
        )
        if best[2] >= 0.98 * base[2]:
            best = base
        return best[1], best[0], best[2]


def rollout(
    controller,
    programs,
    initial,
    pool_indices,
    evaluation_matrices,
    evaluation_y,
    mode="adaptive",
    fixed=None,
):
    chosen = list(initial)
    errors = []
    trace = []
    for budget in BUDGETS:
        if mode == "fixed_queries":
            chosen = list(fixed[:budget])
        model, selected, score = controller.select(programs, chosen)
        error = np.asarray(evaluation_y) - model.predict(evaluation_matrices)
        errors.append(float(np.std(error)))
        trace.append(
            {
                "budget": budget,
                "selected_indices": list(chosen),
                "program": selected,
                "support_loo_rmse": score,
            }
        )
        if budget < 16 and mode == "adaptive":
            chosen += model.acquire(
                controller.matrices, [i for i in pool_indices if i not in chosen], 2
            )
    return {
        "auc": float(np.trapezoid(errors, dx=2) / 8),
        "errors": errors,
        "trace": trace,
    }


def build(source, out):
    source = Path(source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((source / "protocol.json").read_text())
    train = json.loads((source / "train.json").read_text())
    val = json.loads((source / "validation.json").read_text())
    datasets = json.loads((source / "datasets.json").read_text())
    data = {r["molecule"]: r for r in datasets}
    language = answers()
    records = []
    current = None
    controller = None
    for index, ep in enumerate(train, 1):
        d = data[ep["molecule"]]
        if current != ep["molecule"]:
            y = np.array(d["qm_energy_kcal_mol"]) - d["sage_energy_kcal_mol"]
            controller = Controller(d["fourier_sum_matrices"], y, ep["state"])
            current = ep["molecule"]
        test = d["retired_test_indices"]
        eval_x = subset(d["fourier_sum_matrices"], test)
        eval_y = y[test]
        costs = []
        for text in language:
            result = rollout(
                controller,
                json.loads(text)["programs"],
                ep["support_indices"],
                d["pool_indices"],
                eval_x,
                eval_y,
            )
            costs.append(result["auc"])
        abstain = next(
            i
            for i, text in enumerate(language)
            if json.loads(text)["programs"][0]["action"] == "abstain"
        )
        base = costs[abstain]
        best = min(
            range(len(language)),
            key=lambda i: (
                costs[i],
                len(json.loads(language[i])["programs"]),
                sum(len(p["harmonics"]) for p in json.loads(language[i])["programs"]),
                language[i],
            ),
        )
        gain = (base - costs[best]) / max(base, 1e-8)
        target = best if gain > 0.01 else abstain
        record = {
            k: v for k, v in ep.items() if k not in ("candidate_utilities", "answer")
        }
        record.update(
            plan_costs=costs,
            baseline_auc=base,
            oracle_auc=costs[best],
            answer=json.loads(language[target]),
            target_plan_index=target,
            relative_target_gain=(base - costs[target]) / max(base, 1e-8),
            utility_weight=1
            + min(2, max(0, (base - costs[target]) / max(base, 1e-8)) / 0.1),
        )
        records.append(record)
        print(
            "TEACHER",
            index,
            len(train),
            ep["molecule"],
            "seed",
            ep["seed"],
            "gain",
            round(record["relative_target_gain"], 4),
            flush=True,
        )
    (out / "train.json").write_text(json.dumps(records, indent=2))
    (out / "validation.json").write_text(json.dumps(val, indent=2))
    (out / "datasets.json").write_text(json.dumps(datasets, indent=2))
    (out / "language.json").write_text(json.dumps(language, indent=2))
    system = (
        protocol["system_prompt"]
        + " Observations are E_QM minus E_Sage, centered on the observed support mean; they are not absolute QM torsion profiles. Choose a plan to minimize the entire error-versus-QM-budget curve from8 to16 observations. A hypothesis can be valuable later even when it is not accepted at8points. Keep plausible alternatives for later Bayesian selection. Abstain only when no alternative plan is useful across the full trajectory."
    )
    new = {
        "status": "frozen trajectory-utility development experiment; external held-out cohort forbidden",
        "baseline": protocol["baseline"],
        "system_prompt": system,
        "seed": 20261121,
        "updates": 95,
        "checkpoints": [0, 48, 95],
        "adapter": protocol["adapter"],
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "clip_norm": 1.0,
        "dtype": "BF16 base floats; FP32 adapter/loss",
        "episode_schedule": "one shuffled pass of95distinct parents; seed=stepmod3; all285episodes available",
        "supervision": "all56 plan AUCs per training episode; true training-molecule8conformers outside16point acquisition pool evaluate complete adaptive rollout at8/10/12/14/16; no validation outcomes in teacher",
        "target": "best full-trajectory AUC plan, abstain if gain<=1%; ties fewer programs then fewer terms then canonical identity; no initial-fit benefit filter",
        "loss": "Utility-weighted mean full-vocabulary completion-token cross entropy, weight=1+min(2,relativegain/.1). Utility-supervised imitation, not RL. Architecture, decoder, optimizer unchanged from initial-fit study.",
        "decoder": "Existing deterministic greedy constrained token decoding over identical 56 canonical plans; same decoder for zero update and all trained checkpoints.",
        "selection": "lowest mean whole-trajectory adaptive error-budgetAUC on12validation parents, include zero update, tieearliest",
        "controls": "Bayes; samebudget fixed_two/BMA_two; matched95trajectory-utility regressor and answer classifier; retrieval; training-global best constant plan; full285references; largerCV/BMA10references",
        "primary_comparisons": "Bayes and strongest matched-budget non-LLM selected on validation; paired descriptive97.5%scaffold intervals, not confirmatory",
        "ablation": "Prior initial-fit Sage study is historical reference; changed prompt, seed, targets and weighting prevent isolated causal attribution.",
        "held_out_access": "No external300molecule energy/force or outcome-record access, regardless of result",
        "single_seed_limit": "one seed; all validation data previously inspected and development-only",
        "budgets": list(BUDGETS),
        "source_train_sha256": hashlib.sha256(
            (source / "train.json").read_bytes()
        ).hexdigest(),
        "train_sha256": hashlib.sha256((out / "train.json").read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(
            (out / "validation.json").read_bytes()
        ).hexdigest(),
    }
    (out / "protocol.json").write_text(json.dumps(new, indent=2))
    print("TEACHER_COMPLETE", len(records), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    build(a.source, a.out)
