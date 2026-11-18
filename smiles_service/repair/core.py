"""Bounded torsion programs, verified scopes and Gaussian Fourier corrections.

Only isolated driven-torsion residual models are supported by this prototype.
SMIRKS are selectors (no reaction arrows); all numerical coefficients come from data.
"""
import hashlib
import json
import math
import numpy as np
from rdkit import Chem

VERSION = "torsion-repair-v1"
HARMONICS = (1, 2, 3, 4, 6)


def semantic_id(program):
    return hashlib.sha256(
        json.dumps(program, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def chemistry(smiles, torsion):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("One valid connected mapped molecule is required")
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if 0 in maps or len(set(maps)) != len(maps):
        raise ValueError("Every atom needs a unique positive map number")
    if len(torsion) != 4 or len(set(torsion)) != 4 or not set(torsion) <= set(maps):
        raise ValueError("Four distinct existing torsion map numbers required")
    ids = [maps.index(v) for v in torsion]
    bonds = [mol.GetBondBetweenAtoms(a, b) for a, b in zip(ids, ids[1:])]
    if any(b is None for b in bonds):
        raise ValueError("Torsion must be a connected four-atom path")
    return mol, ids


def verified_scopes(smiles, torsion):
    mol, ids = chemistry(smiles, torsion)

    def atom(i, detailed, tag):
        a = mol.GetAtomWithIdx(i)
        props = f"#{a.GetAtomicNum()};" + ("a" if a.GetIsAromatic() else "A")
        if detailed:
            props += (
                f";D{a.GetDegree()};H{a.GetTotalNumHs(includeNeighbors=True)};{a.GetFormalCharge():+d};"
                + ("R" if a.IsInRing() else "R0")
            )
        return "[" + props + ":" + str(tag) + "]"

    def pattern(order, detailed):
        text = atom(order[0], detailed, 1)
        for j, (a, b) in enumerate(zip(order, order[1:]), 2):
            text += (
                ":"
                if mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
                else {1.0: "-", 2.0: "=", 3.0: "#"}.get(
                    mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble(), "~"
                )
            ) + atom(b, detailed, j)
        return text

    # Orientation canonicalization prevents dependence on atom order or map labels.
    broad = min(pattern(ids, False), pattern(ids[::-1], False))
    local = min(pattern(ids, True), pattern(ids[::-1], True))

    # Restore tag order along either path (a full reversal leaves dihedral unchanged).
    def retag(s):
        q = Chem.MolFromSmarts(s)
        for j, a in enumerate(q.GetAtoms()):
            a.SetAtomMapNum(j + 1)
        return Chem.MolToSmarts(q)

    return {"broad": retag(broad), "local": retag(local)}


def compile_program(program, smiles, torsion):
    if not isinstance(program, dict) or set(program) != {
        "action",
        "scope",
        "harmonics",
    }:
        raise ValueError("Program must contain only action, scope, harmonics")
    action = program["action"]
    scope = program["scope"]
    hs = program["harmonics"]
    if action not in ("keep", "revise", "specialize", "abstain"):
        raise ValueError("Unknown repair action")
    if (
        not isinstance(hs, list)
        or not hs
        or any(type(h) is not int or h < 1 or h > 6 for h in hs)
        or hs != sorted(set(hs))
    ):
        raise ValueError("Harmonics must be sorted unique integers in 1..6")
    scopes = verified_scopes(smiles, torsion)
    if scope not in ("broad", "local") or (scope == "local") != (
        action == "specialize"
    ):
        raise ValueError("Only specialization may use verified local scope")
    if action in ("keep", "abstain") and hs != [1, 2, 3]:
        raise ValueError("Keep/abstain must retain baseline H=3")
    mol, ids = chemistry(smiles, torsion)
    q = Chem.MolFromSmarts(scopes[scope])
    matches = mol.GetSubstructMatches(q, uniquify=False, maxMatches=10000)
    paths = {min(tuple(m), tuple(m[::-1])) for m in matches}
    if min(tuple(ids), tuple(ids[::-1])) not in paths:
        raise ValueError("Scope does not match driven torsion")
    clean = {"action": action, "scope": scope, "harmonics": hs}
    return {
        "program": clean,
        "semantic_id": semantic_id(clean),
        "scope_smarts": scopes[scope],
        "matched_torsions": sorted(
            [[mol.GetAtomWithIdx(i).GetAtomMapNum() for i in m] for m in paths]
        ),
        "coefficient_count": 2 * len(hs),
        "application": "driven-instance residual correction only; not a global force-field type replacement",
    }


def design(angles, hs):
    a = np.asarray(angles, float).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite angles")
    return np.column_stack([f(n * a) for n in hs for f in (np.cos, np.sin)])


class Posterior:
    def __init__(self, angles, y, hs, prior=None, alpha=1.0, noise=0.1):
        self.hs = list(hs)
        self.alpha = float(alpha)
        self.noise = float(noise)
        x = design(angles, hs)
        y = np.asarray(y, float)
        if (
            y.shape != (len(x),)
            or not np.isfinite(y).all()
            or len(y) < 2
            or alpha <= 0
            or noise <= 0
        ):
            raise ValueError(
                "At least two finite observations and positive alpha/noise required"
            )
        self.prior = np.zeros(x.shape[1]) if prior is None else np.asarray(prior, float)
        if self.prior.shape != (x.shape[1],) or not np.isfinite(self.prior).all():
            raise ValueError("Invalid prior")
        self.xmean = x.mean(axis=0)
        self.ymean = float(y.mean())
        xc = x - self.xmean
        yc = y - self.ymean
        self.cov = np.linalg.solve(
            alpha * np.eye(x.shape[1]) + xc.T @ xc / noise**2, np.eye(x.shape[1])
        )
        self.mean = self.cov @ (alpha * self.prior + xc.T @ yc / noise**2)

    def predict(self, a):
        return self.ymean + (design(a, self.hs) - self.xmean) @ self.mean

    def variance(self, a):
        x = design(a, self.hs) - self.xmean
        return np.einsum("ij,jk,ik->i", x, self.cov, x)

    def periodic_terms(self):
        # a*cos(n phi)+b*sin(n phi) = k*cos(n phi-phase).
        # k*(1+cos(...)) adds an irrelevant constant per torsion.
        return [
            {
                "periodicity": n,
                "k_kcal_mol": float(math.hypot(a, b)),
                "phase_rad": float(math.atan2(b, a)),
            }
            for n, (a, b) in zip(self.hs, self.mean.reshape(-1, 2))
        ]


def loo_score(angles, y, hs, prior=None):
    a = np.asarray(angles)
    y = np.asarray(y)
    pred = []
    if len(y) < 4:
        raise ValueError(
            "At least four observations required for support-only selection"
        )
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        pred.append(Posterior(a[keep], y[keep], hs, prior).predict(a[i : i + 1])[0])
    # Absolute OOF residuals retain fold-specific centering errors.
    return float(np.sqrt(np.mean((np.array(pred) - y) ** 2)))


def select(programs, smiles, torsion, angles, y, priors=None, min_relative_gain=0.02):
    baseline = {"action": "keep", "scope": "broad", "harmonics": [1, 2, 3]}
    candidates = {semantic_id(baseline): baseline}
    rejected = []
    for p in programs:
        try:
            c = compile_program(p, smiles, torsion)
            candidates[c["semantic_id"]] = p
        except (ValueError, TypeError, KeyError) as e:
            rejected.append({"program": p, "reason": str(e)})
    rows = []
    for key, p in sorted(candidates.items()):
        prior = None if priors is None else priors(p)
        score = loo_score(angles, y, p["harmonics"], prior)
        rows.append({"program": p, "semantic_id": key, "support_loo_rmse": score})
    base = next(r for r in rows if r["program"] == baseline)
    best = min(
        rows,
        key=lambda r: (
            r["support_loo_rmse"],
            len(r["program"]["harmonics"]),
            r["semantic_id"],
        ),
    )
    if best["support_loo_rmse"] >= base["support_loo_rmse"] * (1 - min_relative_gain):
        best = base
    p = best["program"]
    prior = None if priors is None else priors(p)
    model = Posterior(angles, y, p["harmonics"], prior)
    return model, {
        "selected": compile_program(p, smiles, torsion),
        "support_loo_rmse": best["support_loo_rmse"],
        "baseline_support_loo_rmse": base["support_loo_rmse"],
        "candidates": rows,
        "rejected": rejected,
        "periodic_terms": model.periodic_terms(),
        "validation_status": "support-only model selection; external physical validation required",
    }


def rules():
    return [
        {
            "action": "revise" if scope == "broad" else "specialize",
            "scope": scope,
            "harmonics": list(range(1, h + 1)),
        }
        for scope in ("broad", "local")
        for h in HARMONICS
    ]


def acquire(model, angles, available, ids, batch=2):
    remaining = list(available)
    chosen = []
    cov = model.cov.copy()
    x = design(angles, model.hs) - model.xmean
    for _ in range(min(batch, len(remaining))):
        best = min(remaining, key=lambda i: (-float(x[i] @ cov @ x[i]), ids[i]))
        chosen.append(best)
        remaining.remove(best)
        v = cov @ x[best]
        cov -= np.outer(v, v) / (model.noise**2 + x[best] @ v)
    return chosen


class SharedTypes:
    """Fit genuine shared coefficient blocks, optionally splitting a parent type.

    Each record is one driven-instance observation with molecule, angle, y,
    environment (broad/local key). Multiple torsions per geometry are unsupported.
    Intercepts are nuisance offsets per molecule, projected out before fitting.
    """

    def __init__(self, records, hs, specialize=False, alpha=1.0, noise=0.1):
        if not records or alpha <= 0 or noise <= 0:
            raise ValueError("Records and positive hyperparameters required")
        self.hs = list(hs)
        self.specialize = specialize
        self.keys = sorted({r["local" if specialize else "broad"] for r in records})
        self.width = 2 * len(hs)
        self.offsets = {}
        x = self._design(records)
        y = np.array([r["y"] for r in records], float)
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite labels")
        self.xmeans = {}
        for name in sorted({r["molecule"] for r in records}):
            ids = np.array([r["molecule"] == name for r in records])
            self.offsets[name] = float(y[ids].mean())
            self.xmeans[name] = x[ids].mean(axis=0)
            x[ids] -= self.xmeans[name]
            y[ids] -= self.offsets[name]
        self.cov = np.linalg.solve(
            alpha * np.eye(x.shape[1]) + x.T @ x / noise**2, np.eye(x.shape[1])
        )
        self.mean = self.cov @ x.T @ y / noise**2

    def _design(self, records):
        x = np.zeros((len(records), self.width * len(self.keys)))
        for i, r in enumerate(records):
            key = r["local" if self.specialize else "broad"]
            if key not in self.keys:
                raise ValueError("Unseen type needs a validated fallback")
            j = self.keys.index(key) * self.width
            x[i, j : j + self.width] = design([r["angle"]], self.hs)[0]
        return x

    def predict(self, records):
        x = self._design(records)
        out = []
        for i, r in enumerate(records):
            name = r["molecule"]
            if name not in self.offsets:
                raise ValueError("New molecule requires support offset calibration")
            out.append(self.offsets[name] + (x[i] - self.xmeans[name]) @ self.mean)
        return np.array(out)
