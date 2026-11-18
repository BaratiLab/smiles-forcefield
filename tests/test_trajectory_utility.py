import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/torsion_repair"))
from trajectory_utility import Controller, rollout, subset
from sage_runtime import select


def fixture():
    rng = np.random.default_rng(145)
    matrices = {"broad": rng.normal(size=(24, 12)), "local": rng.normal(size=(24, 12))}
    y = matrices["local"][:, 0] * 0.6 + matrices["local"][:, 3] * 0.9
    state = {"smiles": "[CH3:1][CH2:2][CH2:3][CH3:4]", "driven_torsion": [1, 2, 3, 4]}
    programs = [{"action": "specialize", "scope": "local", "harmonics": [1, 2]}]
    return matrices, y, state, programs


def test_evaluation_outcomes_cannot_change_any_acquisition_decision():
    matrices, y, state, programs = fixture()
    initial = list(range(8))
    hold = list(range(16, 24))
    a = rollout(
        Controller(matrices, y, state),
        programs,
        initial,
        list(range(16)),
        subset(matrices, hold),
        y[hold],
    )
    poison = y.copy()
    poison[hold] = np.arange(8) * 1e6
    b = rollout(
        Controller(matrices, poison, state),
        programs,
        initial,
        list(range(16)),
        subset(matrices, hold),
        poison[hold],
    )
    assert a["trace"] == b["trace"]
    assert a["auc"] != b["auc"]
    assert [r["budget"] for r in a["trace"]] == [8, 10, 12, 14, 16]
    assert a["auc"] == np.trapezoid(a["errors"], dx=2) / 8
    assert all(not (set(r["selected_indices"]) & set(hold)) for r in a["trace"])


def test_fast_teacher_controller_matches_deployment_selection():
    matrices, y, state, programs = fixture()
    ids = [0, 2, 4, 6, 8, 10, 12, 14]
    fast, p, _ = Controller(matrices, y, state).select(programs, ids)
    reference, audit = select(programs, state, subset(matrices, ids), y[ids])
    assert p == audit["selected"]["program"]
    np.testing.assert_allclose(
        fast.predict(matrices), reference.predict(matrices), atol=1e-10
    )
