# Experimental torsion repair

This implementation gives an LLM a bounded program-proposal role and keeps fitting, model acceptance and query selection numerical. It is development software. Current replay data use QM-minus-UFF residuals and all inspected cohorts are retired. This is not a validated replacement for the production Sage force field.

## Implemented

- Verified broad four-atom and more specific degree/hydrogen/charge/ring/aromatic environment selectors, canonicalized under reversal and atom-map changes.
- Strict `keep`, `revise`, `specialize`, `abstain` programs with harmonics in 1..6; no coefficients supplied by the LLM. Harmonic maxima {1,2,3,4,6} are baseline library choices, while the compiler permits arbitrary subsets within 1..6.
- Gaussian coefficient fitting with molecule energy offsets projected out, support-only leave-one-out acceptance, explicit no-edit candidate and deterministic semantic tie breaking.
- Training-only environment-conditioned coefficient priors. In the QM replay, **specialization selects a local shared prior, followed by a target-specific fit**. It is not a demonstrated global force-field type split.
- `SharedTypes`, a separate joint coefficient fitter for genuinely shared versus split parameter blocks, tested on injected typing defects and clean controls. This synthetic mechanics test is not evidence of real force-field defects.
- Bayesian conditional acquisition after model selection, one local untuned Qwen call per pilot task, and a compiler gate for malformed proposals.
- Development-only teacher programs with candidate utility labels and abstention, ready for subsequent program-supervised training.
- Optional research API. It never changes production job packages.

## Run

Use the chemistry environment described by the repository `environment.yml`. From the repository root:

```bash
python research/torsion_repair/experiment.py prepare --data /path/to/qm-finetune/data-v1 --out /path/to/new-run --limit 24
python research/torsion_repair/llm_mlx.py --out /path/to/new-run --model /path/to/local/qwen2.5-7b-4bit
python research/torsion_repair/experiment.py replay --out /path/to/new-run
python research/torsion_repair/teacher.py --data /path/to/qm-finetune/data-v1 --out /path/to/new-teacher
python research/torsion_repair/validate_openmm.py /path/to/openmm-validation.json
pytest -q
```

MLX inference requires the separate Apple MLX environment already used by the prior experiment. The old acquisition-label adapter is intentionally not loaded. `states.json` and `requests/` expose exactly what the model saw. Hashes identify the model tensors, inputs and proposer source. Invalid or truncated responses fall back to numerical no-edit; failure latency remains recorded. Candidate ordering is absent from this prompt, but full semantic invariance of the free-text proposer has not yet been demonstrated.

## API

Set `SMILES_ENABLE_RESEARCH_REPAIR=1` to opt in, then `POST /api/research/torsion-repair/fit` with mapped `smiles`, four mapped `torsion` atoms, 8–32 distinct-angle `observations` (`angle_rad`, `correction_kcal_mol`), and up to two `programs`. The existing API-key setting applies. Optional `broad_prior` and `local_prior` arrays contain cosine/sine coefficients for n=1..6; API priors are user supplied and their provenance is not audited. Missing priors are zero. With no prior distinction, local scope alone does not change the fitted physics. The API runs fitting/validation only, not arbitrary LLM commands, new QM jobs or cloud instances.

Response terms follow `k*(1+cos(n*phi-phase))`, with k in kcal/mol and phase in radians. Their additive offset is physically irrelevant for one fixed-topology molecule. They apply only to the specified driven instance, even when its selector matches multiple torsions. Do not install the selector globally or multiply the fitted correction across all matches. Additional angles, consistent background energies, assignment accounting and independent validation are required for a full molecule.

OpenMM correction energy and Cartesian force checks validate the periodic-term representation. They do **not** validate repaired GROMACS/LAMMPS/NAMD packages or off-grid QM accuracy. See [OpenMM's convention](https://docs.openmm.org/7.7.0/userguide/theory/02_standard_forces.html) and [RDKit query documentation](https://www.rdkit.org/docs/RDKit_Book.html).

## Before a paper or public repair rollout

1. Pin a modern baseline and audit fitting-record overlap; evaluate the actual full MM-minus-QM residual including all affected torsion instances, not an isolated relaxed-scan surrogate.
2. Align training episodes with deployment acquisition states, split parents/scaffolds before episodes, build validation supervision separately and train program adapters with independent seeds. No repair adapter has been trained by this pilot.
3. Add Bayesian model averaging, learned non-LLM and retrieval-only proposal controls, then fixed-query and adaptive-query factorial comparisons. Equalize candidate slots/tools/priors; report the exhaustive CV library separately.
4. Freeze the protocol before acquiring an external untouched QM cohort. Report energy/force and off-grid/branch tests with scaffold-cluster uncertainty; current development numbers cannot support a fresh generalization claim.
5. Validate repaired full-system engine exports, install only validated parameter patches, and add per-job uncertainty/coverage provenance. Complete public-server security, abuse/compute limits and deployment verification before enabling public repair.

The experiment proposal in `../acquisition/next_experiment_proposal.json` remains the broader paper plan. This code implements its first development stage, not all confirmatory requirements.

## Completed first development run

See [the implementation report](results/REPORT.md), [pilot analysis](results/pilot-analysis.json), [168-molecule controls](results/controls-analysis.json) and [reproducibility ZIP](results/torsion-repair-reproducibility.zip). The untuned LLM was essentially tied with Bayes on 24 molecules; the fixed two-proposal numerical control improved error AUC by 8.92% on all 168 retired molecules. These results do not establish an LLM advantage. The archive contains source QM replay data and audit logs, excluding model weights. To replay on another machine, extract it and run `prepare` with the extracted `data-v1` directory into a new run directory rather than using the machine-specific path in archived protocol files.
