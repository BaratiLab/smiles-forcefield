"""Freeze a separate actual-Sage program training/validation experiment."""
import argparse, hashlib, json, shutil
from pathlib import Path
from sage_episodes import SAGE_SYSTEM
from program_language import canonical, answers


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(train, val, out):
    train = Path(train)
    val = Path(val)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    tr = json.loads((train / "episodes.json").read_text())
    va = json.loads((val / "episodes.json").read_text())
    assert len({r["parent"] for r in tr}) == 95 and len(va) == 12
    for key in ("molecule", "parent", "scaffold"):
        assert not {r[key] for r in tr} & {r[key] for r in va}
    for r in tr:
        assert canonical(r["answer"]["programs"]) in answers()
    for name, rows in [("train", tr), ("validation", va)]:
        (out / (name + ".json")).write_text(json.dumps(rows, indent=2))
    datasets = json.loads((train / "datasets.json").read_text()) + json.loads(
        (val / "datasets.json").read_text()
    )
    (out / "datasets.json").write_text(json.dumps(datasets, indent=2))
    proto = {
        "status": "frozen development validation experiment; external held-out cohort forbidden",
        "baseline": "Sage2.2.1 unconstrained, fixed AM1-BCC, geometry-complete parent/local Fourier sums",
        "system_prompt": SAGE_SYSTEM,
        "model": "existing Qwen2.5-7B-Instruct4bit, newly initialized LoRA; no UFF adapter reused",
        "seed": 20261120,
        "updates": 95,
        "checkpoints": [0, 48, 95],
        "episode_schedule": "one shuffled full pass over95 distinct parents; choose seed=stepmod3; retain285 available episodes",
        "adapter": {
            "num_layers": 2,
            "rank": 8,
            "scale": 8.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
        },
        "learning_rate": 0.00005,
        "weight_decay": 0.01,
        "clip_norm": 1.0,
        "dtype": "BF16 base floats; FP32 adapter/loss",
        "loss": "completion-only canonical JSON program SFT, equally weighted parents",
        "decoder": "same finite-program trie for base and all fine-tuned checkpoints; no option labels;56 legal canonical one/two-program answers",
        "selection": "lowest molecule-mean accepted-program centered RMSE on12 validation parents,8support+8internalpoolholdout; include zero-update, earliest exact tie",
        "primary_controls": "fixed Bayes parent H3, fixed_two, learned_two, retrieval_two, BMA_two; exact-used-episode matched variants plus285-episode references; larger-libraryCV/BMA separately",
        "primary_win": "report point mean improvement versus Bayes and strongest matched non-LLM selected on validation, plus paired scaffold bootstrap97.5%intervals; validation alone is not confirmatory generalization",
        "secondary": "retired validation conformer error-budgetAUC8/10/12/14/16, fixed versus adaptive queries after checkpoint freeze",
        "held_out_access": "No reads from QCArchive outcome endpoints, no300-cohort energy/force access, even if validation improves",
        "single_seed_limit": "one fine-tuning seed; paper requires additional independent seeds",
        "source_train_sha256": digest(train / "episodes.json"),
        "source_validation_sha256": digest(val / "episodes.json"),
        "train_sha256": digest(out / "train.json"),
        "validation_sha256": digest(out / "validation.json"),
        "program_language_size": len(answers()),
    }
    (out / "protocol.json").write_text(json.dumps(proto, indent=2))
    print(
        "Prepared Sage training:",
        len(tr),
        "episodes;",
        len(va),
        "validation;",
        len(answers()),
        "valid answers",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.train, a.validation, a.out)
