import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    directory = ROOT / "research/torsion_repair"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(
        "milestone_" + name, directory / (name + ".py")
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def fixture():
    return {
        "molecule": {
            "smiles": "[CH3:1][CH2:2][CH2:3][CH3:4]",
            "torsions": [[1, 2, 3, 4]],
            "chemistry": {},
        },
        "pool": [
            {
                "id": str(i),
                "angles_rad": [float(a)],
                "qm_energy_kcal_mol": float(np.cos(a)),
                "baseline_energy_kcal_mol": 0.0,
            }
            for i, a in enumerate(np.linspace(-3, 3, 16))
        ],
        "test": [],
    }


def test_alignment_exactly_matches_deployed_initializer_and_state(tmp_path):
    aligned = module("aligned")
    experiment = module("experiment")
    d = fixture()
    data = tmp_path / "data"
    (data / "datasets").mkdir(parents=True)
    (data / "datasets/m.json").write_text(json.dumps(d))
    (data / "split.json").write_text(
        json.dumps({"splits": {"train": [], "validation": [], "test": ["m"]}})
    )
    experiment.prepare(data, tmp_path / "old", 1)
    state, indices = aligned.state_for(d, [], 0)
    assert state == json.loads((tmp_path / "old/states.json").read_text())["m"]
    assert (
        indices
        == json.loads((tmp_path / "old/initial-selections.json").read_text())["m"]
    )
    for i, r in enumerate(d["pool"]):
        if i not in indices:
            r["qm_energy_kcal_mol"] += 1e5
    assert aligned.state_for(d, [], 0)[0] == state


def test_bma_energy_offset_and_candidate_order_invariance():
    controls = module("controls")
    a = np.linspace(-3, 3, 12)
    y = np.cos(2 * a)
    zero = lambda p: np.zeros(2 * len(p["harmonics"]))
    m = controls.Mixture(controls.FIXED, a, y, zero)
    other = controls.Mixture(controls.FIXED[::-1], a, y + 1000, zero)
    np.testing.assert_allclose(m.weights, other.weights, atol=1e-8)
    np.testing.assert_allclose(m.predict(a), other.predict(a) - 1000, atol=1e-8)
    ids = [str(i) for i in range(12)]
    assert m.acquire(a, [0, 2, 4], ids, 2) == other.acquire(a, [0, 2, 4], ids, 2)


def test_program_parser_rejects_over_budget_and_truncated_outputs():
    trainer = module("train_programs_mlx")
    assert trainer.parse('{"programs":[{}, {}, {}]}')[1]
    assert trainer.parse('{"programs":[')[1]
    assert trainer.parse('```json\n{"programs": []}\n```') == ([], None)


def test_structure_only_inventory_excludes_seen_fitted_and_incomplete(tmp_path):
    import io, tarfile
    from rdkit import Chem

    inventory = module("inventory")
    seen = tmp_path / "seen"
    seen.mkdir()
    (seen / "m.json").write_text(json.dumps({"molecule": {"smiles": "CCCC"}}))
    archive = tmp_path / "fit.tar.gz"
    sdf = (Chem.MolToMolBlock(Chem.MolFromSmiles("C1CCC1")) + "\n$$$$\n").encode()
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("fit/input.sdf")
        info.size = len(sdf)
        tf.addfile(info, io.BytesIO(sdf))
    entries = {
        k: {"attributes": {"canonical_isomeric_smiles": s}}
        for k, s in [
            ("seen", "CCCC"),
            ("fit", "C1CCC1"),
            ("fresh", "c1ccccc1"),
            ("bad", "C1CCCC1"),
        ]
    }
    (tmp_path / "entries.json").write_text(json.dumps(entries))
    (tmp_path / "status.json").write_text(
        json.dumps(
            [[k, "default", "error" if k == "bad" else "complete"] for k in entries]
        )
    )
    inventory.run(
        tmp_path / "entries.json",
        [seen],
        archive,
        tmp_path / "out",
        status=tmp_path / "status.json",
    )
    r = json.loads((tmp_path / "out/candidate-structures.json").read_text())
    assert [v["entry"] for v in r] == ["fresh"]
    assert (
        json.loads((tmp_path / "out/inventory.json").read_text())[
            "qm_outcome_records_retrieved"
        ]
        is False
    )


def test_optional_sage_coverage_and_geometry_sum(tmp_path):
    import pytest

    pytest.importorskip("openff.toolkit")
    from rdkit import Chem

    coverage = module("coverage")
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([{"entry": "m", "smiles": "CCCCC"}]))
    coverage.run(path, tmp_path / "coverage")
    assert (
        json.loads((tmp_path / "coverage/coverage.json").read_text())[
            "coverage_passed_parents"
        ]
        == 1
    )
    sage = module("sage_geometry")
    m = Chem.MolFromSmiles("CCCCC")
    xyz = np.array([[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 1], [2, 1, 0]]], float)
    paths = [[0, 1, 2, 3], [1, 2, 3, 4]]
    summed = sage.matrix(m, xyz, paths, [1, 2, 3])
    individual = sum(sage.matrix(m, xyz, [p], [1, 2, 3]) for p in paths)
    np.testing.assert_allclose(summed, individual, atol=1e-12)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(
        summed, sage.matrix(m, xyz @ rotation + 3, paths, [1, 2, 3]), atol=1e-12
    )
