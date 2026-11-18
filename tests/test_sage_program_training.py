import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    p = ROOT / "research/torsion_repair"
    sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(
        "sage_test_" + name, p / (name + ".py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text.endswith("<|im_end|>"):
            return [ord(c) for c in text[:-10]] + [999]
        return [ord(c) for c in text]


def test_all_grammar_answers_compile_and_are_canonical():
    language = module("program_language")
    from smiles_service.repair.core import compile_program

    assert len(language.answers()) == 56
    for answer in language.answers():
        programs = json.loads(answer)["programs"]
        assert len(programs) <= 2
        assert language.canonical(programs[::-1]) == answer
        for p in programs:
            compile_program(p, "[CH3:1][CH2:2][CH2:3][CH3:4]", [1, 2, 3, 4])


def test_stop_token_absorbs_mlx_lookahead_but_rejects_invalid_prefix():
    language = module("program_language")
    tok = CharacterTokenizer()
    p = language.Processor(tok, 10)
    sequence = tok.encode(language.answers()[0] + "<|im_end|>")
    for i, t in enumerate(sequence):
        assert t in p.allowed(sequence[:i])
    assert p.allowed(sequence) == [999]
    assert p.allowed(sequence + [999]) == [999]
    with pytest.raises(ValueError):
        p.allowed([999])
    with pytest.raises(ValueError):
        p.allowed(sequence + [12])


def test_summed_feature_selection_ignores_unobserved_labels_and_energy_zero():
    runtime = module("sage_runtime")
    a = np.linspace(-3, 3, 16)
    x = np.column_stack([f(n * a) for n in range(1, 7) for f in (np.cos, np.sin)])
    matrices = {"broad": 2 * x, "local": x}
    y = np.cos(2 * a)
    state = {"smiles": "[CH3:1][CH2:2][CH2:3][CH3:4]", "driven_torsion": [1, 2, 3, 4]}
    p = {"action": "specialize", "scope": "local", "harmonics": [1, 2]}
    chosen = list(range(0, 16, 2))
    observed = runtime.subset(matrices, chosen)
    m, r = runtime.select([p], state, observed, y[chosen])
    m2, r2 = runtime.select([p], state, observed, y[chosen] + 100)
    assert r["selected"]["program"] == r2["selected"]["program"]
    np.testing.assert_allclose(
        m.predict(matrices), m2.predict(matrices) - 100, atol=1e-9
    )
    poisoned = y.copy()
    poisoned[1::2] += 1e6
    m3, r3 = runtime.select([p], state, observed, poisoned[chosen])
    assert r3 == r
    np.testing.assert_allclose(m3.predict(matrices), m.predict(matrices))
