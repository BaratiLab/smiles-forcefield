"""Matched cached-base versus trained program proposer on retired pilot states."""
import argparse, hashlib, json, time
from pathlib import Path
from train_programs_mlx import parse, prompt_for, hashfile


def run(aligned, training, model_path, cached, out):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers

    aligned = Path(aligned)
    training = Path(training)
    cached = Path(cached)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    done = json.loads((training / "COMPLETE.json").read_text())
    proto = json.loads((aligned / "protocol.json").read_text())
    provenance = json.loads((training / "provenance.json").read_text())
    states = json.loads((cached / "states.json").read_text())
    base = json.loads((cached / "llm-plans.json").read_text())
    cache_manifest = json.loads((cached / "llm-manifest.json").read_text())
    assert done["trained_updates"] == proto["updates"]
    assert provenance["train_sha256"] == hashfile(
        aligned / "train.json"
    ) and provenance["validation_sha256"] == hashfile(aligned / "validation.json")
    assert done["trained_adapter_sha256"] == hashfile(
        training / "best-trained-adapters.safetensors"
    )
    assert (
        provenance["model_hashes"]
        == {p.name: hashfile(p) for p in Path(model_path).glob("*.safetensors")}
        == cache_manifest["model_files_sha256"]
    )
    model, tok = load(model_path, tokenizer_config={"trust_remote_code": False})
    model.set_dtype(mx.bfloat16)
    model.freeze()
    config = proto["adapter"]
    mx.random.seed(proto["seed"])
    linear_to_lora_layers(
        model,
        config["num_layers"],
        {k: v for k, v in config.items() if k != "num_layers"},
    )
    model.eval()
    manifest = {
        "status": "retired 24-molecule development diagnostic",
        "selection": done,
        "adapter_sha256": done["trained_adapter_sha256"],
        "states_sha256": hashfile(cached / "states.json"),
        "base_cache_manifest_sha256": hashfile(cached / "llm-manifest.json"),
        "model_hashes": provenance["model_hashes"],
        "source_sha256": hashfile(__file__),
        "scope": "best nonzero validation checkpoint for diagnostic comparison; deployment candidate may be the unchanged base if validation selects step0",
    }
    mp = out / "manifest.json"
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest
    else:
        mp.write_text(json.dumps(manifest, indent=2))
    # One exact zero-adapter regeneration checks the cached matched-base route.
    first = next(iter(states))
    prompt = prompt_for(tok, states[first])
    assert hashlib.sha256(prompt.encode()).hexdigest() == base[first]["prompt_sha256"]
    raw = generate(
        model,
        tok,
        prompt=prompt,
        max_tokens=180,
        sampler=make_sampler(temp=0),
        verbose=False,
    )
    if raw != base[first]["raw_response"]:
        raise ValueError(
            "Cached base regeneration differs; regenerate matched baseline rather than silently reusing it"
        )
    (out / "base-repeatability.json").write_text(
        json.dumps(
            {
                "molecule": first,
                "raw_response_exact_match": True,
                "prompt_sha256": base[first]["prompt_sha256"],
            },
            indent=2,
        )
    )
    mx.clear_cache()
    weights = mx.load(str(training / "best-trained-adapters.safetensors"))
    current = dict(tree_flatten(model.trainable_parameters()))
    assert set(weights) == set(current)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.trainable_parameters())
    plans = (
        json.loads((out / "llm-plans.json").read_text())
        if (out / "llm-plans.json").exists()
        else {}
    )
    (out / "requests").mkdir(exist_ok=True)
    for i, (n, state) in enumerate(states.items(), 1):
        if n in plans:
            continue
        prompt = prompt_for(tok, state)
        assert hashlib.sha256(prompt.encode()).hexdigest() == base[n]["prompt_sha256"]
        t = time.monotonic()
        raw = generate(
            model,
            tok,
            prompt=prompt,
            max_tokens=180,
            sampler=make_sampler(temp=0),
            verbose=False,
        )
        programs, error = parse(raw)
        record = {
            "programs": programs,
            "raw_response": raw,
            "parse_error": error,
            "seconds": time.monotonic() - t,
            "prompt_sha256": base[n]["prompt_sha256"],
        }
        plans[n] = record
        (out / "llm-plans.json").write_text(json.dumps(plans, indent=2))
        (out / "requests" / f"{n}.json").write_text(
            json.dumps({"state": state, "prompt": prompt}, indent=2)
        )
        print("INFER", i, len(states), n, programs, error, flush=True)
        mx.clear_cache()
    (out / "COMPLETE").write_text(
        "Retired development plans complete; no evaluator labels supplied to inference.\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aligned", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cached", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.aligned, a.training, a.model, a.cached, a.out)
