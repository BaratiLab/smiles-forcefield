# Torsion specialization and harmonic selection: implementation and development results

The bounded repair loop is implemented, tested and available in the repository. The current development result supports numerical model revision but does not establish an LLM advantage.

## What runs now

The LLM receives only verified molecular structure, acquired QM-minus-UFF observations, residuals, coefficient uncertainties and training-only analogue counts. It returns at most two structured programs: keep, revise harmonics, specialize the verified environment prior, or abstain. Harmonic subsets are bounded to 1..6. The compiler rejects unsupported edits; the model never supplies numerical force constants. Gaussian fitting estimates coefficients; support-only leave-one-out validation with a 2% gain gate chooses the representation, retaining no-edit. Bayesian conditional acquisition selects additional configurations. The pilot makes one model call after eight common support configurations, then uses budgets 8/10/12/14/16.

**Specialization scope:** the real-QM replay specializes a shared training-derived coefficient prior by local chemical environment, followed by a target-specific fit. It is not yet a global Sage/SMIRNOFF type split. A separate `SharedTypes` fitter actually pools/splits coefficient blocks across instances and passes synthetic injected-defect and clean-control tests. Those tests verify mechanics, not a discovered real force-field defect. All current real data are isolated driven-torsion residual fits; global selector matches must not be installed as though every matching torsion independently receives the full fitted correction.

## Results

Metric: mean centered held-out relative-energy RMSE averaged across the query-budget curve, in kcal/mol. This differs from the older normalized acquisition metric; do not compare the numeric scales. All these molecules were previously inspected and are now retired development data.

| Method | Same 24-molecule pilot | Full 168-molecule control cohort |
|---|---:|---:|
| Fixed-model Bayes | 1.618874 | 1.075444 |
| Fixed two-proposal baseline | 1.463532 | 0.979531 |
| CV: ten-proposal library | 1.486420 | 0.992738 |
| Untuned 7B LLM: up to two proposals | 1.618831 | Not run |

On 168 molecules (143 scaffold groups), the fixed two-proposal baseline improves on Bayes by 8.92%, and the exhaustive CV library by 7.69%. These are development diagnostics, not a fresh improvement claim. The baseline proposes broad H=2 and local H=4; it is a fixed representation heuristic, not a learned chemical reasoner. CV explores ten candidates and therefore has a larger proposal/fit budget; it is reported as a stronger-library reference. All policies share priors and numerical tools. This is one acquisition seed, not the planned multi-seed paper experiment.

The LLM is essentially tied with Bayes on its own 24-molecule subset (relative change -0.0027%). It retained H=3 in every proposal. 1 of 48 proposed programs was invalid (abstain with local scope); the compiler rejected it. All 24 responses parsed as JSON. Mean local generation time was 14.1 seconds per molecule. The original angle-label LoRA was not reused: this is an untuned Qwen2.5-7B 4-bit base model with BF16 floating tensors, and no new repair adapter has been trained. No improvement should be attributed to chemical reasoning from this result.

A useful learning target now exists: 190 repair-program episodes from 95 training parents, with two support splits per parent and training-only internal holdouts. The target parent is excluded from its own prior library. Teacher generation does not read the old validation/test labels or even the training-parent dataset `test` field. Episodes are correlated and do not increase the independent molecule count. Their random support-state distribution must be aligned with the Bayes-initialized deployment states before confirmatory training; a separate scaffold-grouped validation set and independent training seeds remain necessary.

## Validation and API

- 14 API/repair tests and 34 existing acquisition integrity tests passed locally.
- Tests cover program validation, atom ordering/map renumbering/torsion reversal, aromatic bonds and explicit mapped hydrogens, proposal ordering, energy offsets, hidden-label invariance, synthetic shared-type defect recovery and clean controls.
- OpenMM Reference relative-energy parity error: 8.9e-16 kcal/mol. Maximum Cartesian finite-difference force discrepancy: 5.6e-8 kJ/mol/nm over seven conformations and all six harmonics. This validates correction-term representation, not full-system GROMACS/LAMMPS/NAMD exports or QM forces.
- `POST /api/research/torsion-repair/fit` accepts mapped structure, observations and bounded proposals; it returns selected scope, terms, fit evidence and rejected programs. It is disabled unless `SMILES_ENABLE_RESEARCH_REPAIR=1`; the existing API-key control applies. It does not execute a public LLM command, run QM, buy compute, or modify downloadable baseline force fields. User-supplied priors are explicitly unaudited.

## Development history and reproducibility

Before the completed v2 pilot, tests exposed explicit mapped hydrogen removal and aromatic-bond selector errors. They were corrected and covered by regression tests. The incomplete v1 inference was interrupted; it is not a reported performance run. No new QM outcomes were purchased or calculated. Data come from the same pinned published OpenFF scans as the previous study; current residual background is UFF, not the website's Sage force field. This limits physical interpretation.

The archive includes code, tests, complete v2 requests/responses, numerical replay audits, the 168-molecule controls, teacher programs, source data needed for those replays, and SHA256 manifests. The local model weights are identified by hash and not duplicated. All former test labels used for these development summaries remain retired. The code and API have been added to the repository; this work does not claim that a cloud deployment was upgraded.

## Path to a paper and community release

1. Implement baseline-specific geometry-complete assignments on a pinned modern force field, summing all affected torsion instances. Audit training overlap and prevent a high-frequency scan fit from masquerading as a useful transferable repair.
2. Align and expand grouped development episodes; train program adapters using downstream utility and abstention with separate validation and multiple seeds. Compare equally equipped learned non-LLM, retrieval and rule proposers, plus CV and Bayesian model averaging. Separate fixed-query model revision from adaptive-query gains.
3. Inventory an external untouched QM cohort, deduplicate records/parents/scaffolds, freeze the protocol and power calculation, then evaluate once. Validate off-grid energies, conformer branches and forces. The proposed 300-molecule target is not an acquired dataset.
4. Validate repaired full-system exports and independent molecular behavior before enabling repaired public downloads. Add bounded queued inference, coverage/provenance, explicit validation states and deployment checks. The existing baseline generator can remain useful independently.

A publishable LLM contribution requires evidence that its proposals improve upon these strong controls. The first pilot establishes functioning infrastructure and a negative untuned-proposer result; it is not yet that paper claim.

Implementation details: `research/torsion_repair/README.md`. Broader paper design: `research/acquisition/next_experiment_proposal.json`. Community rollout gates: `docs/torsion-repair-release-plan.md`.
