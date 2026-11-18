# SMILES Forcefield

[Open the website](https://baratilab.github.io/smiles-forcefield/)

Generate real OpenFF Sage 2.2.1 / AM1-BCC packages for GROMACS, LAMMPS, and NAMD-compatible AMBER input. The complete API, web interface, tests, Docker configuration, and HTTPS cloud deployment files are in **smiles-forcefield-source.zip**.

## Run the backend

```bash
git clone https://github.com/BaratiLab/smiles-forcefield.git
cd smiles-forcefield
unzip smiles-forcefield-source.zip
cd smiles-forcefield
micromamba create -f environment.yml
micromamba activate smiles-forcefield
pip install --no-deps -e .
uvicorn smiles_service.api:app --host 127.0.0.1 --port 8000
```

In another terminal, activate the same environment and run `python -m smiles_service.worker` from the same directory. Open http://127.0.0.1:8000 for the application and /docs for the API specification. Alternatively, use `docker compose up --build` from the extracted project directory.

## Current deployment status

The GitHub Pages interface is live. The public chemistry API is **not deployed yet**; a cloud provider/server must be selected. In the website, Connection settings accepts the URL of a separately hosted API. The extracted `deploy/README.md` contains a single-VM deployment with HTTPS, private worker/storage, and submission limits.

## Validation

All five development molecules completed real local API jobs, with all 20 ZIP downloads checked. Native GROMACS and LAMMPS energies and forces were compared to OpenMM. NAMD-compatible AMBER files were independently read back, but **native NAMD was not run**. Checks cover one isolated vacuum geometry per molecule, not predictive accuracy or solvated production MD. Five API/queue tests pass locally.

Supported first-release inputs: one neutral, closed-shell organic molecule, specified stereochemistry, supported H/C/N/O/F/P/S/halogen elements, at most 40 heavy atoms and 100 total atoms. Full scientific limits and provenance are documented in the source archive.
