"""Validation-only comparison; never opens the external held-out cohort."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from sage_controls import matched_rows


def run(aligned, training, controls, out):
    aligned = Path(aligned)
    training = Path(training)
    controls = Path(controls)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    proto = json.loads((aligned / "protocol.json").read_text())
    done = json.loads((training / "COMPLETE.json").read_text())
    history = [
        json.loads(line)
        for line in (training / "training.jsonl").read_text().splitlines()
    ]
    train = json.loads((aligned / "train.json").read_text())
    expected = matched_rows(train, proto)
    assert [(r["molecule"], r["episode_seed"]) for r in history] == [
        (r["molecule"], r["seed"]) for r in expected
    ]
    assert len(history) == 95 and len({r["molecule"] for r in history}) == 95
    table = json.loads((controls / "primary.json").read_text())
    names = sorted({r["molecule"] for r in table["records"]})
    lookup = {(r["molecule"], r["policy"]): r["rmse"] for r in table["records"]}
    rng = np.random.default_rng(20261120)
    sample = rng.integers(0, len(names), (20000, len(names)))
    checkpoints = {}
    selection = json.loads((training / "validation.json").read_text())
    for checkpoint in selection:
        step = checkpoint["step"]
        records = json.loads(
            (training / f"validation-{step:03d}" / "scores.json").read_text()
        )["molecules"]
        values = {r["molecule"]: r["rmse"] for r in records}
        mean = float(np.mean(list(values.values())))
        compare = {}
        for policy in table["summary"]:
            difference = np.array([values[n] - lookup[n, policy] for n in names])
            draws = difference[sample].mean(1)
            reference = table["summary"][policy]
            compare[policy] = {
                "mean_difference_kcal_mol": float(difference.mean()),
                "relative_change": float(mean / reference - 1),
                "paired_descriptive_97_5pct_interval": np.quantile(
                    draws, [0.0125, 0.9875]
                ).tolist(),
            }
        plans = json.loads(
            (training / f"validation-{step:03d}" / "plans.json").read_text()
        )
        checkpoints[str(step)] = {
            "mean_validation_rmse": mean,
            "comparisons": compare,
            "parse_errors": sum(r["parse_error"] is not None for r in plans.values()),
            "compiler_rejections": sum(len(r["audit"]["rejected"]) for r in records),
            "calls": len(plans),
            "mean_seconds": float(np.mean([r["seconds"] for r in plans.values()])),
            "selected_programs": {
                r["molecule"]: r["audit"]["selected"]["program"] for r in records
            },
        }
    best = done["selected"]["step"]
    trained = done["best_nonzero"]["step"]
    strongest = table["strongest_matched_control"]
    best_values = checkpoints[str(best)]
    trained_values = checkpoints[str(trained)]
    result = {
        "status": "development validation only; selected checkpoint and control comparisons are exploratory, not confirmatory inference",
        "training": done,
        "actual_training_parents": 95,
        "available_training_episodes": len(train),
        "validation_parents": len(names),
        "validation_scaffolds": len({r["scaffold"] for r in table["records"]}),
        "controls": table["summary"],
        "strongest_matched_control": strongest,
        "checkpoints": checkpoints,
        "best_trained_mean_better_than_bayes": trained_values["comparisons"]["bayes"][
            "mean_difference_kcal_mol"
        ]
        < 0,
        "best_trained_mean_better_than_strongest_control": trained_values[
            "comparisons"
        ][strongest]["mean_difference_kcal_mol"]
        < 0,
        "selected_mean_better_than_bayes": best_values["comparisons"]["bayes"][
            "mean_difference_kcal_mol"
        ]
        < 0,
        "selected_mean_better_than_strongest_control": best_values["comparisons"][
            strongest
        ]["mean_difference_kcal_mol"]
        < 0,
        "held_out_access": "External300-molecule energy/force outcomes were never loaded or scored",
        "uncertainty_caution": "Scaffold bootstrap12groups, one training seed. Intervals do not adjust for checkpoint/control selection on the same validation data and do not prove generalization.",
        "loss_and_optimization": {
            "initial_example_loss": history[0]["loss"],
            "final_example_loss": history[-1]["loss"],
            "max_preclip_gradient_norm": max(r["gradient_norm"] for r in history),
            "clipped_updates": sum(r["gradient_norm"] > 1 for r in history),
            "mean_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in history])),
            "statement": "Different examples at each step; endpoint losses are not a controlled learning curve. Large finite gradients are clipped, not evidence of robust optimization.",
        },
        "protocol_sha256": hashlib.sha256(
            (aligned / "protocol.json").read_bytes()
        ).hexdigest(),
    }
    # Explicit paired FT-versus-base diagnostic.
    base = {
        r["molecule"]: r["rmse"]
        for r in json.loads((training / "validation-000/scores.json").read_text())[
            "molecules"
        ]
    }
    for step in checkpoints:
        values = {
            r["molecule"]: r["rmse"]
            for r in json.loads(
                (training / f"validation-{int(step):03d}/scores.json").read_text()
            )["molecules"]
        }
        d = np.array([values[n] - base[n] for n in names])
        checkpoints[step]["versus_untuned"] = {
            "mean_difference": float(d.mean()),
            "paired_descriptive_97_5pct_interval": np.quantile(
                d[sample].mean(1), [0.0125, 0.9875]
            ).tolist(),
        }
    (out / "analysis.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "selected_step": best,
                "best_trained_step": trained,
                "bayes": table["summary"]["bayes"],
                "strongest_control": strongest,
                "selected_rmse": best_values["mean_validation_rmse"],
                "best_trained_rmse": trained_values["mean_validation_rmse"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aligned", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--controls", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.aligned, a.training, a.controls, a.out)
