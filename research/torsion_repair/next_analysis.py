"""Exploratory summaries of next-stage development experiments."""
import argparse, json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np


def summarize(folder):
    rows = json.loads((Path(folder) / "metrics.json").read_text())
    lookup = {(r["molecule"], r["policy"], r["mode"]): r for r in rows}
    names = sorted({r["molecule"] for r in rows})
    policies = sorted({r["policy"] for r in rows})
    scaffolds = sorted({r["scaffold"] for r in rows})
    clusters = [
        np.array(
            [
                i
                for i, n in enumerate(names)
                if lookup[n, "bayes_fixed", "adaptive"]["scaffold"] == s
            ]
        )
        for s in scaffolds
    ]
    rng = np.random.default_rng(20261119)
    samples = [
        np.concatenate(
            [clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]
        )
        for _ in range(5000)
    ]
    base = np.array([lookup[n, "bayes_fixed", "adaptive"]["rmse_auc"] for n in names])
    result = {}
    for p in policies:
        result[p] = {}
        for mode in ["adaptive", "fixed_queries"]:
            value = np.array([lookup[n, p, mode]["rmse_auc"] for n in names])
            diff = value - base
            draws = [float(diff[ids].mean()) for ids in samples]
            result[p][mode] = {
                "mean_rmse_auc_kcal_mol": float(value.mean()),
                "relative_change_vs_bayes": float(value.mean() / base.mean() - 1),
                "paired_difference_vs_bayes_95pct_scaffold_interval": np.quantile(
                    draws, [0.025, 0.975]
                ).tolist(),
            }
        diff = np.array(
            [
                lookup[n, p, "adaptive"]["rmse_auc"]
                - lookup[n, p, "fixed_queries"]["rmse_auc"]
                for n in names
            ]
        )
        result[p]["adaptive_minus_fixed_query_mean"] = float(diff.mean())
        result[p]["adaptive_minus_fixed_query_95pct_scaffold_interval"] = np.quantile(
            [diff[ids].mean() for ids in samples], [0.025, 0.975]
        ).tolist()
    return {
        "molecules": len(names),
        "scaffolds": len(scaffolds),
        "policies": result,
        "status": "retired data, one acquisition seed; exploratory cluster intervals",
    }


def run(root):
    root = Path(root)
    controls = summarize(
        root
        / (
            "controls168-matched"
            if (root / "controls168-matched").exists()
            else "controls168"
        )
    )
    pilot = summarize(
        root
        / (
            "trained-controls24-matched"
            if (root / "trained-controls24-matched").exists()
            else "trained-controls24"
        )
    )
    training = json.loads((root / "training-v2/COMPLETE.json").read_text())
    history = [
        json.loads(line)
        for line in (root / "training-v2/training.jsonl").read_text().splitlines()
    ]
    plans = json.loads((root / "trained-plans24/llm-plans.json").read_text())
    audit = json.loads((root / "trained-controls24/audit.json").read_text())
    first = [audit[n + "::llm_two::adaptive"][0] for n in plans]
    selected = Counter(
        json.dumps(r["selected"]["program"], sort_keys=True) for r in first
    )
    result = {
        "status": "next-milestone development results; fresh QM cohort remains unscored",
        "controls168": controls,
        "trained_pilot24": pilot,
        "training": training,
        "training_detail": {
            "actual_unique_training_parents_used": len(
                {r["molecule"] for r in history}
            ),
            "available_training_parents": 95,
            "available_episodes": 285,
            "optimizer_updates": len(history),
            "initial_loss": history[0]["loss"],
            "final_loss": history[-1]["loss"],
            "preclip_gradient_norm_min": min(r["gradient_norm"] for r in history),
            "preclip_gradient_norm_max": max(r["gradient_norm"] for r in history),
            "all_updates_clipped": all(r["gradient_norm"] > 1 for r in history),
            "comment": "Losses are from different examples; do not interpret endpoint difference as a controlled learning curve. Large gradients were finite and clipped, but optimization is not proved robust.",
        },
        "trained_proposals": {
            "calls": len(plans),
            "mean_seconds": float(np.mean([r["seconds"] for r in plans.values()])),
            "parse_errors": sum(r["parse_error"] is not None for r in plans.values()),
            "unique_proposal_sets": len(
                {json.dumps(r["programs"], sort_keys=True) for r in plans.values()}
            ),
            "invalid_programs_at_first_step": sum(len(r["rejected"]) for r in first),
            "total_programs": sum(len(r["programs"]) for r in plans.values()),
            "selected_at_first_step": dict(selected),
        },
        "validation_controls": json.loads(
            (root / "validation-controls.json").read_text()
        ),
        "fresh_inventory": json.loads((root / "inventory/inventory.json").read_text()),
        "native_validation": json.loads(
            (root / "repaired-export-validation/validation.json").read_text()
        ),
    }
    if (root / "sage-validation-controls.json").exists():
        result["sage_validation_controls"] = json.loads(
            (root / "sage-validation-controls.json").read_text()
        )
    (root / "analysis.json").write_text(json.dumps(result, indent=2))
    print("Summarized controls, trained adapter and validation gate")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    run(a.root)
