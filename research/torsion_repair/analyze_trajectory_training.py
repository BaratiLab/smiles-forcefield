"""Exploratory, paired validation analysis of whole-trajectory Sage training."""
import argparse, json
from pathlib import Path
import numpy as np


def analyze(root):
    root = Path(root)
    training = root / "training-v1"
    complete = json.loads((training / "COMPLETE.json").read_text())
    controls = json.loads((root / "controls/scores.json").read_text())
    proto = json.loads((root / "aligned-v1/protocol.json").read_text())
    policies = {
        p: {r["molecule"]: r for r in controls["records"] if r["policy"] == p}
        for p in controls["summary"]
    }
    for step in proto["checkpoints"]:
        scores = json.loads(
            (training / f"validation-{step:03d}/scores.json").read_text()
        )
        policies[f"llm_{step:03d}"] = {r["molecule"]: r for r in scores["molecules"]}
    selected = f"llm_{complete['selected']['step']:03d}"
    trained = f"llm_{complete['best_nonzero']['step']:03d}"
    matched = [
        "fixed_two",
        "global_matched",
        "labels_matched",
        "learned_matched",
        "retrieval_matched",
        "bma_two",
    ]
    means = {
        p: float(np.mean([r["auc"] for r in rows.values()]))
        for p, rows in policies.items()
    }
    strong = min(matched, key=lambda p: (means[p], p))
    names = sorted(policies["bayes"])
    scaffolds = [policies["bayes"][n]["scaffold"] for n in names]
    groups = sorted(set(scaffolds))
    rng = np.random.default_rng(20261121)
    draws = rng.integers(0, len(groups), (20000, len(groups)))
    comparisons = {}
    for llm in dict.fromkeys([selected, trained]):
        for ref in ["bayes", strong, "cv_library", "learned_full", "llm_000"]:
            delta = np.array(
                [
                    np.mean(
                        [
                            policies[llm][n]["auc"] - policies[ref][n]["auc"]
                            for n, s in zip(names, scaffolds)
                            if s == g
                        ]
                    )
                    for g in groups
                ]
            )
            samples = delta[draws].mean(axis=1)
            comparisons[llm + " minus " + ref] = {
                "mean_delta": float(delta.mean()),
                "descriptive_97_5_percent_interval": np.quantile(
                    samples, [0.0125, 0.9875]
                ).tolist(),
                "molecule_wins": sum(
                    policies[llm][n]["auc"] < policies[ref][n]["auc"] - 1e-10
                    for n in names
                ),
                "molecule_ties": sum(
                    abs(policies[llm][n]["auc"] - policies[ref][n]["auc"]) <= 1e-10
                    for n in names
                ),
            }
    inference = {}
    for step in proto["checkpoints"]:
        plans = json.loads((training / f"validation-{step:03d}/plans.json").read_text())
        inference[str(step)] = {
            "mean_seconds_per_plan": float(
                np.mean([p["seconds"] for p in plans.values()])
            ),
            "abstentions": sum(
                not p["programs"]
                or all(v["action"] in ("abstain", "keep") for v in p["programs"])
                for p in plans.values()
            ),
            "parse_failures": sum(p["parse_error"] is not None for p in plans.values()),
            "proposals": {name: p["programs"] for name, p in plans.items()},
            "unique_plans": len(
                {json.dumps(p["programs"], sort_keys=True) for p in plans.values()}
            ),
            "broad_base_selected_events": sum(
                t["program"]["scope"] == "broad"
                and t["program"]["harmonics"] == [1, 2, 3]
                for r in policies[f"llm_{step:03d}"].values()
                for t in r["trace"]
            ),
            "total_selection_events": 5 * len(plans),
        }
    result = dict(
        inference=inference,
        mean_validation_auc=means,
        selected_policy=selected,
        best_trained_policy=trained,
        strongest_matched_control=strong,
        comparisons=comparisons,
        validation_molecules=len(names),
        validation_scaffolds=len(groups),
        held_out_evaluated=False,
        status="Previously inspected development validation; checkpoint-selected, one training seed, exploratory intervals. No confirmatory claim.",
    )
    (root / "analysis.json").write_text(json.dumps(result, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [
        "bayes",
        "llm_000",
        trained,
        "global_matched",
        "learned_matched",
        "bma_two",
        "cv_library",
        "learned_full",
    ]
    show = list(dict.fromkeys(show))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#2878a6" if p.startswith("llm") else "#89969b" for p in show]
    labels = {
        "bayes": "Bayes (parent H3)",
        "llm_000": "Untuned LLM",
        trained: f"Best trained LLM (step {complete['best_nonzero']['step']})",
        "global_matched": "Train-global plan (95 episodes)",
        "learned_matched": "Utility regressor (95 episodes)",
        "bma_two": "BMA (two proposals)",
        "cv_library": "CV (10 representations)",
        "learned_full": "Utility regressor (285 episodes)",
    }
    ax.barh([labels[p] for p in show], [means[p] for p in show], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Mean centered RMSE AUC (kcal/mol; lower is better)")
    ax.set_title("Sage whole-trajectory planner: 12 development validation scaffolds")
    for i, p in enumerate(show):
        ax.text(means[p] + 0.005, i, f"{means[p]:.3f}", va="center")
    ax.set_xlim(0, max(means[p] for p in show) * 1.15)
    fig.tight_layout()
    fig.savefig(
        root / "validation-comparison.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    lines = [
        "# Sage whole-trajectory utility fine-tuning",
        "",
        f"Completed {complete['trained_updates']} local LoRA updates. Selected checkpoint: {complete['selected']['step']}; best nonzero checkpoint: {complete['best_nonzero']['step']}. The external 300-molecule held-out cohort was not evaluated.",
        "",
        "| Policy | Mean validation AUC (kcal/mol) |",
        "|---|---:|",
    ]
    lines += [
        f"| {p} | {v:.6f} |" for p, v in sorted(means.items(), key=lambda x: x[1])
    ]
    lines += [
        "",
        f"Best trained checkpoint versus untuned LLM: {means[trained]-means['llm_000']:+.6f}; versus Bayes: {means[trained]-means['bayes']:+.6f}; versus strongest matched control: {means[trained]-means[strong]:+.6f}. Negative differences favor training; point estimates alone do not establish superiority.",
        "",
        "Lower is better. Each value averages the normalized centered-RMSE curve over budgets 8, 10, 12, 14 and 16. Twelve validation parents/scaffolds; no training parent or scaffold overlaps. These are previously inspected development data, not a fresh held-out test.",
        "",
        f"Strongest matched control: `{strong}`. Controls trained on full-trajectory costs use exactly the same 95 parent/episode pairs as SFT. Full-data learners see 285 episodes from the same 95 parents and QM conformers, providing more support-initialization supervision rather than more unique QM labels; CV/BMA library controls can inspect ten representations instead of two.",
        "",
        "## Paired differences",
        "",
        "Negative differences favor the LLM. Intervals are descriptive 97.5% scaffold bootstrap intervals, with no correction for checkpoint/model selection.",
    ]
    for label, r in comparisons.items():
        lo, hi = r["descriptive_97_5_percent_interval"]
        lines += [
            "",
            f"- {label}: {r['mean_delta']:+.6f} [{lo:+.6f}, {hi:+.6f}]; wins {r['molecule_wins']}/12, ties {r['molecule_ties']}/12.",
        ]
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "Training is utility-weighted supervised imitation of train-only oracle plans, not reinforcement learning or direct optimization through Bayesian acquisition. The model proposes scope/harmonic hypotheses; the numerical Bayesian fitter supplies parameters and decides acceptance from observed labels. No QM oracle labels are exposed at inference.",
        "",
        "One seed, 95 updates, and a small repeatedly inspected validation panel cannot establish generalization. The changed prompt, schedule seed and weighting also prevent attributing differences solely to the teacher horizon. Greedy constrained decoding remains a potential source of bias; complete-plan scoring would be a separate experiment. A training-data oracle bound is not an achievable validation result.",
        "",
        "The external held-out cohort remains sealed. Neither a paper-level superiority claim nor promotion into the public website follows from this pilot.",
        "",
        "## Artifacts",
        "",
        "`aligned-v1/` contains all 56 plan costs per training episode, split data, language and frozen protocol. `training-v1/` contains adapter and optimizer checkpoints, generated plans, full acquisition traces, numerical guards, hashes and training history. `controls/` contains stronger-control traces and the train-selected global plan. Base model weights are excluded.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    analyze(a.root)
