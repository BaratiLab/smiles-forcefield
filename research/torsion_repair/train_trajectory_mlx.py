"""Whole-trajectory utility-weighted Sage SFT with matched constrained decoding."""
import argparse, hashlib, importlib.metadata, json, subprocess, time
from pathlib import Path
import numpy as np
from program_language import canonical, answers, Processor


def hashfile(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse(raw):
    try:
        s = raw.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(s)
        if (
            set(value) != {"programs"}
            or not isinstance(value["programs"], list)
            or len(value["programs"]) > 2
        ):
            raise ValueError("Invalid envelope")
        return value["programs"], None
    except (ValueError, TypeError) as e:
        return [], str(e)


def prompt_for(tok, state, system):
    return tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(state, sort_keys=True)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def run(aligned, model_path, chem_python, out):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load, generate
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.sample_utils import make_sampler

    aligned = Path(aligned)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((aligned / "protocol.json").read_text())
    train = json.loads((aligned / "train.json").read_text())
    val = json.loads((aligned / "validation.json").read_text())
    seed = protocol["seed"]
    mx.random.seed(seed)
    model, tok = load(model_path, tokenizer_config={"trust_remote_code": False})
    model.set_dtype(mx.bfloat16)
    model.freeze()
    config = protocol["adapter"]
    linear_to_lora_layers(
        model,
        config["num_layers"],
        {k: v for k, v in config.items() if k != "num_layers"},
    )
    flat = tree_flatten(model.trainable_parameters())
    assert flat and all(k.endswith(("lora_a", "lora_b")) for k, v in flat)
    provenance = {
        "protocol_sha256": hashfile(aligned / "protocol.json"),
        "train_sha256": hashfile(aligned / "train.json"),
        "validation_sha256": hashfile(aligned / "validation.json"),
        "source_sha256": hashfile(__file__),
        "prompt_source_sha256": hashfile(
            Path(__file__).with_name("program_language.py")
        ),
        "model_hashes": {
            p.name: hashfile(p) for p in Path(model_path).glob("*.safetensors")
        },
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "trainable_parameters": sum(v.size for k, v in flat),
        "trainable_names": [k for k, v in flat],
        "dtype": "BF16 base; FP32 LoRA/loss",
        "fresh_test_used": False,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": 2,
                "lora_parameters": {
                    k: v for k, v in config.items() if k != "num_layers"
                },
            },
            indent=2,
        )
    )
    optimizer = optim.AdamW(
        learning_rate=protocol["learning_rate"], weight_decay=protocol["weight_decay"]
    )

    def loss_fn(m, x, targets, start, weight):
        hidden = m.model(x[None])[:, start:, :]
        logits = m.lm_head(hidden).astype(mx.float32)
        return weight * nn.losses.cross_entropy(logits[0], targets, reduction="mean")

    grad_fn = nn.value_and_grad(model, loss_fn)
    history = []
    validation = []
    started = time.monotonic()

    def validate(step):
        model.eval()
        folder = out / f"validation-{step:03d}"
        folder.mkdir()
        plans = {}
        for r in val:
            prompt = prompt_for(tok, r["state"], protocol["system_prompt"])
            t = time.monotonic()
            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            max_tokens = (
                max(
                    len(tok.encode(s + "<|im_end|>", add_special_tokens=False))
                    for s in answers()
                )
                + 2
            )
            raw = generate(
                model,
                tok,
                prompt=prompt_ids,
                max_tokens=max_tokens,
                logits_processors=[Processor(tok, len(prompt_ids))],
                sampler=make_sampler(temp=0),
                verbose=False,
            )
            programs, error = parse(raw)
            plans[r["molecule"]] = {
                "programs": programs,
                "parse_error": error,
                "raw_response": raw,
                "seconds": time.monotonic() - t,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            mx.clear_cache()
            print("VALIDATION_INFER", step, r["molecule"], programs, error, flush=True)
        (folder / "plans.json").write_text(json.dumps(plans, indent=2))
        subprocess.run(
            [
                chem_python,
                str(Path(__file__).with_name("score_trajectory_validation.py")),
                "--aligned",
                str(aligned),
                "--plans",
                str(folder / "plans.json"),
                "--out",
                str(folder / "scores.json"),
            ],
            check=True,
        )
        scores = json.loads((folder / "scores.json").read_text())
        record = {"step": step, "mean_validation_auc": scores["mean_validation_auc"]}
        validation.append(record)
        (out / "validation.json").write_text(json.dumps(validation, indent=2))
        print("VALIDATION", json.dumps(record), flush=True)

    # Save zero adapter so the no-training control is a valid deployment candidate.
    mx.save_safetensors(str(out / "adapters-000.safetensors"), dict(flat))
    validate(0)
    names = sorted({r["molecule"] for r in train})
    rng = np.random.default_rng(seed)
    schedule = []
    while len(schedule) < protocol["updates"]:
        schedule.extend(rng.permutation(names).tolist())
    for step, name in enumerate(schedule[: protocol["updates"]], 1):
        r = next(
            r for r in train if r["molecule"] == name and r["seed"] == (step - 1) % 3
        )
        prompt = prompt_for(tok, r["state"], protocol["system_prompt"])
        prefix = tok.encode(prompt, add_special_tokens=False)
        completion = tok.encode(
            canonical(r["answer"]["programs"]) + "<|im_end|>", add_special_tokens=False
        )
        ids = prefix + completion
        model.train()
        t = time.monotonic()
        loss, grads = grad_fn(
            model,
            mx.array(ids[:-1]),
            mx.array(completion),
            len(prefix) - 1,
            r["utility_weight"],
        )
        mx.eval(loss, grads)
        bad = [
            k for k, v in tree_flatten(grads) if not bool(mx.all(mx.isfinite(v)).item())
        ]
        if not np.isfinite(float(loss.item())) or bad:
            (out / "numerical_failure.json").write_text(
                json.dumps({"step": step, "loss": float(loss.item()), "nonfinite": bad})
            )
            raise ValueError("Nonfinite before update")
        grads, norm = optim.clip_grad_norm(grads, protocol["clip_norm"])
        mx.eval(norm)
        if not np.isfinite(float(norm.item())):
            raise ValueError("Nonfinite gradient norm")
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        if any(
            not bool(mx.all(mx.isfinite(v)).item())
            for _, v in tree_flatten(model.trainable_parameters())
        ):
            raise ValueError("Nonfinite adapter after optimizer update")
        record = {
            "step": step,
            "molecule": name,
            "episode_seed": r["seed"],
            "loss": float(loss.item()),
            "gradient_norm": float(norm.item()),
            "prompt_tokens": len(prefix),
            "completion_tokens": len(completion),
            "utility_weight": r["utility_weight"],
            "seconds": time.monotonic() - t,
        }
        history.append(record)
        with (out / "training.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print("TRAIN", json.dumps(record), flush=True)
        mx.clear_cache()
        if step in protocol["checkpoints"]:
            mx.save_safetensors(
                str(out / f"adapters-{step:03d}.safetensors"),
                dict(tree_flatten(model.trainable_parameters())),
            )
            mx.save_safetensors(
                str(out / f"optimizer-{step:03d}.safetensors"),
                dict(tree_flatten(optimizer.state)),
            )
            validate(step)
    best = min(validation, key=lambda r: (r["mean_validation_auc"], r["step"]))
    nonzero = min(
        [r for r in validation if r["step"] > 0],
        key=lambda r: (r["mean_validation_auc"], r["step"]),
    )
    import shutil

    shutil.copy2(
        out / f"adapters-{best['step']:03d}.safetensors", out / "adapters.safetensors"
    )
    shutil.copy2(
        out / f"adapters-{nonzero['step']:03d}.safetensors",
        out / "best-trained-adapters.safetensors",
    )
    norm = float(
        sum(
            float((v * v).sum().item())
            for k, v in tree_flatten(model.trainable_parameters())
            if k.endswith("lora_b")
        )
        ** 0.5
    )
    assert norm > 0
    result = {
        "selected": best,
        "best_nonzero": nonzero,
        "trained_updates": len(history),
        "nonzero_lora_b_norm": norm,
        "seconds": time.monotonic() - started,
        "selected_adapter_sha256": hashfile(out / "adapters.safetensors"),
        "trained_adapter_sha256": hashfile(out / "best-trained-adapters.safetensors"),
        "fresh_test_evaluated": False,
    }
    (out / "COMPLETE.json").write_text(json.dumps(result, indent=2))
    print("COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aligned", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--chem-python", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.aligned, a.model, a.chem_python, a.out)
