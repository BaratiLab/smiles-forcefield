# Research and community release gates

The existing SMILES package generator and the new research repair service have distinct validation requirements. An export-consistent baseline package is not evidence that a learned correction improves QM or molecular dynamics accuracy.

## Current implementation

The repair compiler, Fourier fitter, shared-type splitting mechanics, local open-model proposer, QM replay, teacher-generation scripts and optional research API are available in source. Current experiments use retired published QM scan data with UFF residuals. The production pipeline does not install research patches. Enabling the research fit endpoint does not enable public LLM inference or QM execution.

## Paper experiments

The intended question is whether residual-aware LLM proposals improve model-structure repair over equally equipped rules and learned non-LLM generators. Before a confirmatory run:

- Implement actual baseline-specific shared torsion assignments and fit the sum of all affected instances on full Cartesian geometries. Audit baseline training overlap.
- Align program training and validation episodes with deployment support states; reserve scaffold groups before generation. Train independent seeds; include clean tasks and abstention.
- Compare fixed Bayes, two-proposal rules, an equal-budget learned non-LLM generator, retrieval, support CV and model averaging. Share tools, prior information and numerical solvers.
- Separate model revision with fixed configurations from revision with adaptive acquisition.
- Inventory an external uninspected QM cohort, deduplicate parents/scaffolds/records/geometries, freeze theory and endpoints, calculate power on development data and preregister the untouched comparison.
- Report molecular/scaffold uncertainty, QM evaluation counts/core-hours, LLM latency, force and off-grid errors, proposal validity and repair failures. Do not infer MD accuracy from a relaxed scan.

The larger draft protocol in `research/acquisition/next_experiment_proposal.json` specifies proposed effect-size gates and cohort targets; those are not achieved results.

## Public website

Keep the baseline package route usable independently. A future repair job needs explicit QM data provenance, selected chemical state, baseline/version metadata and coverage. Run the LLM as a bounded queued worker; reject malformed programs in the compiler and bound calls, observations, candidates and CPU time. Do not allow user commands, arbitrary file paths or unchecked model output to execute.

Expose research/validated status, source references, scope, error measurements and exact per-engine validation status on every job. Only release repaired downloads after geometry-complete term assignment, independent physical validation, native-engine energy/force parity and regression tests. Require authentication/rate limits, retention controls, dependency/security review and a verified deployment before enabling a public compute endpoint. New QM/cloud campaigns need a concrete cost specification and authorized budget.

This release plan does not claim the current web deployment was upgraded, that NAMD was executed, or that a new cloud server was provisioned.

## Implemented next-stage milestones

Aligned program-SFT, stronger controls, geometry-complete Sage data preparation and topology-specific corrected exports are now implemented. Their reports distinguish the UFF adapter experiment from the new Sage corpus. A selected unchanged model is a valid negative validation outcome; an updated adapter is not promoted merely because it was trained.

For community use, keep the current SMILES-to-baseline download path independent of optional QM-assisted repair. A refinement job must identify its QM observations and fixed chemical state. Reuse a verified cache where available; otherwise any new QM work is a separate queued computation with explicit resource accounting. SMILES alone do not supply residual evidence. Release an experimental correction only with its exact term assignments, background and validation provenance. The current research CLI can create corrected examples; it does not enable a public, unrestricted QM or LLM service.
