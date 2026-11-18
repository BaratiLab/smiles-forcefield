"""Observed-label-only acceptance and Gaussian fitting for summed Sage features."""
import numpy as np
from smiles_service.repair.core import compile_program, semantic_id

BASE = {"action": "keep", "scope": "broad", "harmonics": [1, 2, 3]}


def columns(p):
    return [2 * (h - 1) + j for h in p["harmonics"] for j in (0, 1)]


class Gaussian:
    def __init__(self, matrices, y, program):
        self.program = program
        self.x = np.asarray(matrices[program["scope"]], float)[:, columns(program)]
        y = np.asarray(y, float)
        if len(y) < 3 or len(self.x) != len(y) or not np.isfinite(y).all():
            raise ValueError("Finite matching observations required")
        self.xmean = self.x.mean(0)
        self.ymean = float(y.mean())
        xx = self.x - self.xmean
        self.cov = np.linalg.solve(
            np.eye(xx.shape[1]) + xx.T @ xx / 0.01, np.eye(xx.shape[1])
        )
        self.mean = self.cov @ xx.T @ (y - self.ymean) / 0.01

    def predict(self, matrices):
        return (
            self.ymean
            + (
                np.asarray(matrices[self.program["scope"]])[:, columns(self.program)]
                - self.xmean
            )
            @ self.mean
        )

    def acquire(self, matrices, available, batch=2):
        x = (
            np.asarray(matrices[self.program["scope"]])[:, columns(self.program)]
            - self.xmean
        )
        cov = self.cov.copy()
        remaining = list(available)
        chosen = []
        for _ in range(min(batch, len(remaining))):
            best = min(remaining, key=lambda i: (-float(x[i] @ cov @ x[i]), i))
            chosen.append(best)
            remaining.remove(best)
            v = cov @ x[best]
            cov -= np.outer(v, v) / (0.01 + x[best] @ v)
        return chosen


def subset(matrices, indices):
    return {s: np.asarray(x)[indices] for s, x in matrices.items()}


def loo(matrices, y, p):
    errors = []
    for i in range(len(y)):
        keep = [j for j in range(len(y)) if j != i]
        m = Gaussian(subset(matrices, keep), np.asarray(y)[keep], p)
        errors.append(m.predict(subset(matrices, [i]))[0] - y[i])
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


def select(programs, state, matrices, y):
    candidates = {semantic_id(BASE): BASE}
    rejected = []
    for p in programs:
        try:
            compiled = compile_program(p, state["smiles"], state["driven_torsion"])
            candidates[compiled["semantic_id"]] = p
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({"program": p, "error": str(error)})
    rows = [
        {"program": p, "support_loo_rmse": loo(matrices, y, p)}
        for _, p in sorted(candidates.items())
    ]
    base = next(r for r in rows if r["program"] == BASE)
    best = min(
        rows,
        key=lambda r: (
            r["support_loo_rmse"],
            len(r["program"]["harmonics"]),
            semantic_id(r["program"]),
        ),
    )
    if best["support_loo_rmse"] >= 0.98 * base["support_loo_rmse"]:
        best = base
    return Gaussian(matrices, y, best["program"]), {
        "selected": best,
        "candidates": rows,
        "rejected": rejected,
    }
