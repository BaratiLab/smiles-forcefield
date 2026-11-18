# SMILES Forcefield

A working research backend and website that accept SMILES, generate real GROMACS, LAMMPS, and NAMD-compatible force-field packages, and release ZIP downloads after consistency checks pass.

**Model:** OpenFF Sage 2.2.1, unconstrained, with AmberTools AM1-BCC charges. No LLM or QM refinement is used in this baseline.

**Validated routes:** native GROMACS and LAMMPS potential energies and atomic forces against OpenMM; independent AMBER topology readback for NAMD-compatible files. **Native NAMD is not run.** Every report and result page states this limitation.

## Run locally

Use the supplied Conda environment (Python 3.11 and native engines required):

```bash
micromamba create -f environment.yml
micromamba activate smiles-forcefield
pip install --no-deps -e .
export SMILES_DATA_DIR="$PWD/data"
export MPLCONFIGDIR="$PWD/mpl-cache"
export OMP_NUM_THREADS=1
uvicorn smiles_service.api:app --host 127.0.0.1 --port 8000
```

In a second terminal with the same environment and data directory:

```bash
python -m smiles_service.worker
```

Open http://127.0.0.1:8000 for the website and http://127.0.0.1:8000/docs for interactive API docs. Workers and API can be restarted independently. The SQLite queue is durable on one machine; jobs are claimed transactionally. Run one worker initially; adding workers increases chemistry CPU/memory usage.

## Docker deployment

```bash
cp .env.example .env
# Configure a private API key and exact CORS origin when deploying publicly.
docker compose up --build -d
```

The container ports bind to loopback by default. Put a TLS reverse proxy in front on a lab server or cloud VM; expose only the API service, never the job volume. Set public request-rate and request-body limits at that proxy. Native jobs run in a separate non-root worker container with CPU/memory limits. The Docker build and Linux dependency solve have not been tested in this local macOS session; the local Python service and chemistry pipeline were tested.

A 4-core machine with at least 8 GB RAM is a starting deployment configuration to benchmark, not a measured production capacity claim. Configure `SMILES_API_KEY` for a trusted shared-key service. Public anonymous operation needs a deployment-specific admission/rate policy. A shared API key is suitable for a trusted group, not per-user tenancy or billing.

## API

Optional authentication header: `X-API-Key`, enforced for submission, status, and downloads when `SMILES_API_KEY` is configured. Structure preview and health are public. Job IDs are random capability identifiers; there is no list-all-jobs endpoint. Treat job URLs as private when running without a shared key.

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"COc1ccccc1"}'
# Returns HTTP 202 with {id, status, status_url, molecule}.

curl http://127.0.0.1:8000/api/jobs/JOB_ID
# Poll until status is succeeded or failed.

curl -o forcefields.zip \
  http://127.0.0.1:8000/api/jobs/JOB_ID/downloads/all
```

Routes:

| Route | Behavior |
|---|---|
| `POST /api/preview` | Validate and canonicalize SMILES; return formula and 2D SVG |
| `POST /api/jobs` | Queue a generation job |
| `GET /api/jobs/{id}` | Status, stage, validation report, archive hashes and download URLs |
| `GET /api/jobs/{id}/downloads/{engine}` | `all`, `gromacs`, `lammps`, or `namd` ZIP |
| `GET /api/health` | API readiness and native engine availability (not worker liveness) |

A failed or incomplete job returns no downloads. A native engine missing or timing out fails the job; the service does not downgrade to an unvalidated package. NAMD readback is a defined checked route, not a claim of native validation. Expected input errors return 422; a full queue returns 429; a download before completion returns 409; expired downloads return 410.

## Supported inputs

One connected, neutral, closed-shell organic molecule containing H, C, N, O, F, P, S, Cl, Br, or I, with explicit stereochemistry. Maximum 40 heavy atoms and 100 total atoms. Molecules must also fit the conservative 2 nm validation-diameter limit after embedding. A molecule inside these limits may still fail charge assignment or parameter coverage; errors are reported instead of inventing values.

Protonation and tautomer choices belong to the submitted structure. The service does not silently neutralize ions, choose a tautomer, or resolve ambiguous stereochemistry. Atom maps supplied by clients are stripped before canonicalization; the returned package provides its own atom mapping.

## Scientific validation scope

One embedded, unminimized geometry per job. Reference OpenMM uses all isolated intramolecular interactions. Native GROMACS/LAMMPS use a 6 nm box, a 2.5 nm cutoff, no switching, and no long-range dispersion correction. Accepted molecule diameter is below 2 nm, avoiding intermolecular image interactions and intramolecular truncation for these isolated cases. LJ/Coulomb 1–4 scales are 0.5 and 1/1.2.

Thresholds: total potential-energy difference <0.01 kJ/mol and maximum Cartesian force-component difference <0.1 kJ/mol/nm. GROMACS mixed precision and coordinate serialization contribute to numerical differences. The AMBER coordinate roundtrip is also checked. Full reports and native logs are included in each download.

These are vacuum single-point examples, **not complete solvated production simulations**. Successful consistency tests do not establish chemical accuracy, MD stability, binding free-energy accuracy, or compatibility with arbitrary protein/water force fields. Multi-conformer and native NAMD validation remain future work.

Each ZIP includes the relevant engine folder, structures, provenance, atom mapping, README, and validation.json. The combined ZIP additionally includes the reference OpenMM system and exact coordinate array. Parameters are real classical force-field assignments; file-format choice does not change their identity into GAFF, CHARMM, or another model.

## GitHub and the website

The public repository is https://github.com/BaratiLab/smiles-forcefield and the website is https://baratilab.github.io/smiles-forcefield/. The complete source tree is published directly on the main branch. The original source ZIP is retained as an initial release snapshot. The public chemistry API is not provisioned yet. `web/` contains a standalone static copy of the same interface served by the API. `.github/workflows/pages.yml` publishes it to GitHub Pages after repository access and Pages configuration are established.

GitHub Pages cannot execute Python, AmberTools, GROMACS or LAMMPS. Host the API separately, then set its HTTPS address in the website's Connection settings or pass `?api=https://your-api-host` in the website URL. Set `SMILES_CORS_ORIGINS=https://baratilab.github.io` on the API. API keys are held only in the browser page's memory and sent in a header; do not hardcode a shared secret into a public Pages build.

Direct file-tree publication once GitHub CLI is authenticated:

```bash
# From this project directory; gh must be authenticated with lab write access.
git push -u origin main
# In repository Settings → Pages, choose GitHub Actions as source.
# Run the Publish website workflow if the initial push did not trigger it.
```

No live public API hosting is created by pushing the repository. The Docker configuration prepares that separate deployment.

## Tests and local evidence

```bash
pip install -e '.[test]'
pytest -q
```

API tests exercise input restrictions, authorization, download gating, queue capacity, expiration, and concurrent atomic job claims. CI runs those tests without heavyweight chemistry dependencies. `LOCAL_VERIFICATION.json` records actual local HTTP submissions for all five development molecules, their native validation reports, and all 20 successful ZIP checks. These are local evidence, not CI or deployed-server results.

The first release was tested on macOS ARM64 with the existing chemistry environment. Docker image building, Linux execution, and GitHub Actions execution require independent verification after publication.

## Operations

- Default maximum pending/running jobs: 20 (`SMILES_MAX_PENDING`).
- Default job timeout: 900 seconds (`SMILES_JOB_TIMEOUT`); worker terminates the child process group on timeout.
- Native engine subprocess timeout: 120 seconds.
- Default download retention: 86400 seconds (`SMILES_RETENTION_SECONDS`), cleaned by a running worker.
- Job files and worker logs live in `SMILES_DATA_DIR/{id}`; never commit them.
- If a worker is forcibly killed, a running record may remain interrupted. Inspect its log and resubmit; automatic retry is deliberately absent to avoid duplicate computations. Graceful operator restart should allow an active job to finish first.
- Back up SQLite and job volume together. SQLite is intended for one host, not a distributed network filesystem.

No remote chemistry services or LLM accounts are required. Dependency licenses remain their upstream licenses; inspect them for your deployment and redistribution needs. NAMD is not distributed.

## Acquisition research

[The acquisition experiment](research/acquisition/README.md) compares random sampling, greedy Bayesian information gain, and a private Ollama/Qwen controller on the five development molecules. It logs valid/failed model actions and matches initial observations and label budgets. The engineering dataset uses MMFF94 minus UFF at identical real geometries; it is explicitly **not QM data or a learned Sage force field**. The research controller is separate from the production API. Optional private CPU deployment settings are in `deploy/compose.research.yaml`.

## Experimental torsion-repair research

The [torsion-repair module](research/torsion_repair/README.md) implements verified specialization programs, harmonic selection, Bayesian fitting/acquisition and a local open-model proposer. The optional `/api/research/torsion-repair/fit` endpoint requires `SMILES_ENABLE_RESEARCH_REPAIR=1`; it returns research corrections and validation evidence without changing production force-field downloads. Current experiments are development-only QM-minus-UFF residual studies. No repaired Sage accuracy or public deployment claim is implied.
