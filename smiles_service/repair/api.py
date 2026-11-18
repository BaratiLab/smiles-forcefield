"""Opt-in research API, isolated from production job/download routes."""
import os
from typing import Annotated, Literal
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(
    prefix="/api/research/torsion-repair", tags=["Experimental torsion repair"]
)


class Program(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["keep", "revise", "specialize", "abstain"]
    scope: Literal["broad", "local"]
    harmonics: list[int] = Field(min_length=1, max_length=6)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    angle_rad: float = Field(ge=-3.141593, le=3.141593)
    correction_kcal_mol: float = Field(ge=-1e6, le=1e6)


PriorCoefficient = Annotated[float, Field(ge=-10000, le=10000)]


class RepairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    smiles: str = Field(min_length=1, max_length=4000)
    torsion: list[int] = Field(min_length=4, max_length=4)
    observations: list[Observation] = Field(min_length=8, max_length=32)
    programs: list[Program] = Field(default_factory=list, max_length=2)
    broad_prior: list[PriorCoefficient] | None = Field(
        default=None, min_length=12, max_length=12
    )
    local_prior: list[PriorCoefficient] | None = Field(
        default=None, min_length=12, max_length=12
    )


@router.post("/fit")
def fit(body: RepairRequest, x_api_key: str | None = Header(default=None)):
    from ..api import authorize

    authorize(x_api_key)
    if os.getenv("SMILES_ENABLE_RESEARCH_REPAIR") != "1":
        raise HTTPException(
            403,
            "Research torsion repair is disabled. Production packages use the validated baseline.",
        )
    import numpy as np
    from .core import select, chemistry

    try:
        mol, _ = chemistry(body.smiles, body.torsion)
        if mol.GetNumAtoms() > 256:
            raise ValueError("Research requests are limited to 256 explicit atoms")

        def prior(p):
            values = body.local_prior if p["scope"] == "local" else body.broad_prior
            if values is None:
                return np.zeros(2 * len(p["harmonics"]))
            return np.array(
                [values[2 * (h - 1) + j] for h in p["harmonics"] for j in (0, 1)]
            )

        a = np.array([r.angle_rad for r in body.observations])
        y = np.array([r.correction_kcal_mol for r in body.observations])
        if len(np.unique(np.round(a, 8))) < 8:
            raise ValueError("Eight distinct observed angles required")
        model, result = select(
            [p.model_dump() for p in body.programs],
            body.smiles,
            body.torsion,
            a,
            y,
            prior,
        )
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(422, str(e))
    return {
        **result,
        "experimental": True,
        "automatically_applied_to_downloads": False,
        "prior_provenance": "User-supplied coefficients, not independently audited; absent priors are zero",
        "scope_limit": "One driven-instance QM-minus-baseline correction. Not a complete force field or global SMIRKS replacement.",
        "fitted_observation_corrections_kcal_mol": model.predict(a).tolist(),
        "required_before_production": [
            "independent conformer/force validation",
            "baseline-specific parameter assignment audit",
            "native engine energy and force checks",
        ],
    }
