import numpy as np
import pytest
from rdkit import Chem
from smiles_service.repair.core import (
    Posterior,
    SharedTypes,
    compile_program,
    verified_scopes,
    select,
    acquire,
)

S = "[CH3:1][CH2:2][CH2:3][CH3:4]"
T = [1, 2, 3, 4]
BASE = {"action": "keep", "scope": "broad", "harmonics": [1, 2, 3]}


def test_invalid_edits_and_unsupported_coefficients():
    for p in [
        dict(BASE, k=3),
        dict(BASE, harmonics=[1, 7]),
        dict(BASE, harmonics=[True]),
        dict(BASE, scope="local"),
        dict(BASE, action="invent"),
    ]:
        with pytest.raises(ValueError):
            compile_program(p, S, T)


def test_atom_order_map_and_torsion_reversal_invariance():
    original = verified_scopes(S, T)
    mol = Chem.MolFromSmiles(S)
    renumbered = Chem.RenumberAtoms(mol, list(reversed(range(4))))
    assert verified_scopes(Chem.MolToSmiles(renumbered, canonical=False), T) == original
    assert verified_scopes(S, T[::-1]) == original
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetAtomMapNum() + 20)
    assert verified_scopes(Chem.MolToSmiles(mol), [v + 20 for v in T]) == original


def test_aromatic_and_explicit_hydrogen_scopes():
    for s, t in [
        ("[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1", [1, 2, 3, 4]),
        (
            "[H:5][C:1]([H:6])([H:7])[C:2]([H:8])([H:9])[C:3]([H:10])([H:11])[H:4]",
            [5, 1, 2, 3],
        ),
    ]:
        assert compile_program(BASE, s, t)["matched_torsions"]


def test_proposal_and_observation_order_invariance_and_offsets():
    a = np.linspace(-3, 3, 12)
    y = np.cos(2 * a)
    p = {"action": "revise", "scope": "broad", "harmonics": [2]}
    m, r = select([BASE, p], S, T, a, y)
    m2, r2 = select([p, BASE], S, T, a[::-1], y[::-1] + 100)
    assert r["selected"] == r2["selected"]
    np.testing.assert_allclose(m.predict(a), m2.predict(a) - 100, atol=1e-10)


def test_portable_periodic_energy_and_derivative():
    a = np.linspace(-3, 3, 20)
    m = Posterior(a, 2 * np.cos(a) - 3 * np.sin(2 * a), [1, 2])
    terms = m.periodic_terms()

    def periodic(v):
        return sum(
            t["k_kcal_mol"] * (1 + np.cos(t["periodicity"] * v - t["phase_rad"]))
            for t in terms
        )

    e = periodic(a) - m.predict(a)
    np.testing.assert_allclose(e, e[0], atol=1e-10)
    numeric = (periodic(a + 1e-6) - periodic(a - 1e-6)) / (2e-6)
    analytic = sum(
        -t["k_kcal_mol"]
        * t["periodicity"]
        * np.sin(t["periodicity"] * a - t["phase_rad"])
        for t in terms
    )
    np.testing.assert_allclose(numeric, analytic, atol=1e-8)


def test_shared_type_defect_recovery_and_clean_control():
    for defect in [False, True]:
        train = []
        test = []
        for name, sign in [("a", 1), ("b", -1 if defect else 1)]:
            for i, angle in enumerate(np.linspace(-np.pi, np.pi, 32, endpoint=False)):
                r = {
                    "molecule": name,
                    "broad": "parent",
                    "local": name,
                    "angle": float(angle),
                    "y": float(sign * 3 * np.cos(2 * angle) + 12),
                }
                (test if i % 4 == 0 else train).append(r)
        broad = SharedTypes(train, [1, 2, 3])
        local = SharedTypes(train, [1, 2, 3], True)
        truth = np.array([r["y"] for r in test])
        be = np.sqrt(np.mean((broad.predict(test) - truth) ** 2))
        le = np.sqrt(np.mean((local.predict(test) - truth) ** 2))
        assert le < 0.01
        if defect:
            assert be > 2 and le < be / 100
        else:
            assert be < 0.01


def test_batch_acquisition_ignores_unobserved_labels():
    a = np.linspace(-3, 3, 16)
    m = Posterior(a[:8], np.cos(a[:8]), [1, 2, 3])
    ids = [str(i) for i in range(16)]
    result = acquire(m, a, list(range(8, 16)), ids, 4)
    assert len(set(result)) == 4 and set(result) <= set(range(8, 16))


def test_research_api_is_gated_and_does_not_apply_to_packages(monkeypatch):
    from fastapi.testclient import TestClient
    from smiles_service.api import app

    client = TestClient(app)
    body = {
        "smiles": S,
        "torsion": T,
        "observations": [
            {"angle_rad": float(a), "correction_kcal_mol": float(np.cos(a))}
            for a in np.linspace(-3, 3, 8)
        ],
        "programs": [{"action": "revise", "scope": "broad", "harmonics": [1]}],
    }
    monkeypatch.delenv("SMILES_API_KEY", raising=False)
    monkeypatch.delenv("SMILES_ENABLE_RESEARCH_REPAIR", raising=False)
    assert client.post("/api/research/torsion-repair/fit", json=body).status_code == 403
    monkeypatch.setenv("SMILES_ENABLE_RESEARCH_REPAIR", "1")
    response = client.post("/api/research/torsion-repair/fit", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["automatically_applied_to_downloads"] is False
    assert "downloads" not in response.json()
    body["broad_prior"] = [1e300] * 12
    assert client.post("/api/research/torsion-repair/fit", json=body).status_code == 422
    body.pop("broad_prior")
    monkeypatch.setenv("SMILES_API_KEY", "test-token")
    assert client.post("/api/research/torsion-repair/fit", json=body).status_code == 401
    assert (
        client.post(
            "/api/research/torsion-repair/fit",
            json=body,
            headers={"X-API-Key": "test-token"},
        ).status_code
        == 200
    )


def test_planner_state_hides_unacquired_and_evaluation_labels(tmp_path):
    import json
    from research.torsion_repair.experiment import prepare

    data = tmp_path / "data"
    (data / "datasets").mkdir(parents=True)
    (data / "split.json").write_text(
        json.dumps({"splits": {"train": [], "validation": [], "test": ["m"]}})
    )
    records = [
        {
            "id": str(i),
            "angles_rad": [float(a)],
            "qm_energy_kcal_mol": float(np.cos(a)),
            "baseline_energy_kcal_mol": 0.0,
        }
        for i, a in enumerate(np.linspace(-3, 3, 24))
    ]
    d = {
        "molecule": {"smiles": S, "torsions": [T], "chemistry": {}},
        "pool": records[:16],
        "test": records[16:],
    }
    path = data / "datasets/m.json"
    path.write_text(json.dumps(d))
    prepare(data, tmp_path / "first", 1)
    selected = json.loads((tmp_path / "first/initial-selections.json").read_text())["m"]
    for i, r in enumerate(d["pool"]):
        if i not in selected:
            r["qm_energy_kcal_mol"] += 10000
    for r in d["test"]:
        r["qm_energy_kcal_mol"] -= 999999
    path.write_text(json.dumps(d))
    prepare(data, tmp_path / "second", 1)
    assert (tmp_path / "first/states.json").read_text() == (
        tmp_path / "second/states.json"
    ).read_text()
