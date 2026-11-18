"""Secondary validation replay analysis, after primary checkpoint selection."""
import argparse, json
from pathlib import Path
import numpy as np


def run(selected, untuned, out):
    rows = json.loads(Path(selected).read_text())
    old = json.loads(Path(untuned).read_text())
    names = sorted({r["molecule"] for r in rows})
    policies = sorted({r["policy"] for r in rows})
    lookup = {(r["molecule"], r["policy"], r["mode"]): r for r in rows}
    oldlookup = {
        (r["molecule"], r["mode"]): r["rmse_auc"] for r in old if r["policy"] == "llm"
    }
    scaffolds = sorted({r["scaffold"] for r in rows})
    clusters = [
        np.array(
            [
                i
                for i, n in enumerate(names)
                if lookup[n, "llm", "adaptive"]["scaffold"] == s
            ]
        )
        for s in scaffolds
    ]
    rng = np.random.default_rng(20261121)
    samples = [
        np.concatenate(
            [clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]
        )
        for _ in range(20000)
    ]
    result = {}
    for mode in ["adaptive", "fixed_queries"]:
        selected_values = np.array([lookup[n, "llm", mode]["rmse_auc"] for n in names])
        summary = {
            p: float(np.mean([lookup[n, p, mode]["rmse_auc"] for n in names]))
            for p in policies
        }
        summary["untuned_llm"] = float(np.mean([oldlookup[n, mode] for n in names]))
        comparisons = {}
        for policy in policies + ["untuned_llm"]:
            reference = np.array(
                [
                    oldlookup[n, mode]
                    if policy == "untuned_llm"
                    else lookup[n, policy, mode]["rmse_auc"]
                    for n in names
                ]
            )
            difference = selected_values - reference
            comparisons[policy] = {
                "mean_difference": float(difference.mean()),
                "relative_change": float(selected_values.mean() / reference.mean() - 1),
                "descriptive_97_5pct_scaffold_interval": np.quantile(
                    [difference[i].mean() for i in samples], [0.0125, 0.9875]
                ).tolist(),
            }
        eligible = [
            "fixed_two",
            "bma_two",
            "learned_matched",
            "labels_matched",
            "retrieval_matched",
        ]
        result[mode] = {
            "summary": summary,
            "strongest_frozen_matched_control": min(
                eligible, key=lambda p: (summary[p], p)
            ),
            "comparisons": comparisons,
        }
    final = {
        "status": "secondary development validation, never used for checkpoint selection; no external held-out outcomes",
        "molecules": len(names),
        "scaffolds": len(scaffolds),
        "metric": "centered energy RMSE averaged over budgets8/10/12/14/16 on the other8old validation conformers",
        "modes": result,
        "interpretation": "Fine-tuned proposals generated once at8points. Abstention retains Bayes representation at later budgets. Primary teacher/selection target was initial8point fit, not full-trajectory utility; this is an objective mismatch to investigate, not a causal ablation.",
        "uncertainty": "One fine-tuning seed and12scaffolds; descriptive intervals after model selection. Supplemental global constant-plan control is posthoc and training-only selected.",
    }
    Path(out).write_text(json.dumps(final, indent=2))
    print(json.dumps(result["adaptive"]["summary"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selected", required=True)
    p.add_argument("--untuned", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.selected, a.untuned, a.out)
