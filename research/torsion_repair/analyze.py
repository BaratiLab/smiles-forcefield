"""Summarize retired-data development runs with scaffold-cluster diagnostics."""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np


def analyze(folder):
    folder = Path(folder)
    rows = json.loads((folder / "metrics.json").read_text())
    audit = json.loads((folder / "replay-audit.json").read_text())
    lookup = {(r["molecule"], r["policy"]): r for r in rows}
    policies = sorted({r["policy"] for r in rows})
    result = {
        "status": "exploratory, retired data, not a fresh confirmation",
        "policies": {},
    }
    for p in policies:
        names = sorted(r["molecule"] for r in rows if r["policy"] == p)
        v = np.array([lookup[n, p]["rmse_auc"] for n in names])
        b = np.array([lookup[n, "bayes_fixed"]["rmse_auc"] for n in names])
        diff = v - b
        scaffolds = sorted({lookup[n, p]["scaffold"] for n in names})
        clusters = [
            np.array([i for i, n in enumerate(names) if lookup[n, p]["scaffold"] == s])
            for s in scaffolds
        ]
        rng = np.random.default_rng(20261118)
        draws = []
        for _ in range(5000):
            ids = np.concatenate(
                [clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]
            )
            draws.append(float(diff[ids].mean()))
        choices = Counter(
            json.dumps(audit[n + "::" + p][0]["selected"]["program"], sort_keys=True)
            for n in names
        )
        result["policies"][p] = {
            "molecules": len(names),
            "scaffolds": len(scaffolds),
            "mean_rmse_auc_kcal_mol": float(v.mean()),
            "relative_change_vs_bayes": float(v.mean() / b.mean() - 1),
            "paired_difference_95pct_scaffold_interval": np.quantile(
                draws, [0.025, 0.975]
            ).tolist(),
            "first_step_selected_programs": dict(choices),
            "rejected_programs_over_all_steps": sum(
                len(step["rejected"]) for n in names for step in audit[n + "::" + p]
            ),
        }
    if (folder / "llm-plans.json").exists():
        plans = json.loads((folder / "llm-plans.json").read_text())
        result["llm"] = {
            "calls": len(plans),
            "mean_seconds": float(np.mean([r["seconds"] for r in plans.values()])),
            "parse_errors": sum(r["parse_error"] is not None for r in plans.values()),
            "distinct_proposal_sets": len(
                {json.dumps(r["programs"], sort_keys=True) for r in plans.values()}
            ),
        }
    result[
        "caution"
    ] = "Intervals describe development replay only; one acquisition seed. CV has ten candidates versus at most two rules/LLM candidates. LLM subset must not be compared with full-cohort means."
    (folder / "analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    analyze(a.out)
