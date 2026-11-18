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

## Next milestones: aligned training, stronger controls and Sage geometry

The next-stage implementation lives alongside the original pilot without rewriting its results:

- `aligned.py`: shared eight-point initialization for training and deployment; 285 grouped development episodes and a separate 12-scaffold validation subset. Teacher retrieval excludes its own parent.
- `train_programs_mlx.py`: real completion-token program SFT on a frozen Qwen2.5-7B quantized backbone, BF16 base/FP32 LoRA, finite-gradient guards, checkpoint/optimizer saves, and downstream validation selection including the zero-update model. It is not an A–E classifier.
- `infer_programs_mlx.py`: retired-pilot evaluation of the best trained checkpoint, with a cached-base provenance check and an exact zero-adapter regeneration control. Deployment selection can still choose the unchanged model.
- `controls.py`: matched two-proposal fixed/learned/retrieval/BMA controls, larger CV/BMA libraries, and fixed-query versus adaptive-query comparisons. Model averaging uses Gaussian marginal evidence after removing the energy offset; its acquisition score is mixture variance, not exact model-discrimination mutual information.
- `sage_geometry.py`: fixed AM1-BCC Sage2.2.1 energies on archived full Cartesian geometries; verified parent/local assignment sets; Fourier columns sum over every affected torsion instance. It creates explicit OpenMM correction systems and records failures without inventing data.
- `export_patch.py`: translates those corrections into Interchange topology-specific terms, exports all three package formats, and invokes native GROMACS/LAMMPS and AMBER readback validation. This is a research CLI, separate from live public jobs.
- `sage_episodes.py` / `sage_compare.py`: prepare and evaluate actual Sage geometry-complete training/validation supervision. This corpus has different scope semantics and must not be mislabeled as the training data for the completed UFF adapter.
- `inventory.py` / `coverage.py`: structure-only exclusion, scan-status/specification checks and Sage topology coverage for a new outcome-blind candidate cohort. Completion metadata do not guarantee a valid complete 24-frame scan.

Example commands from the repository root:

```bash
python research/torsion_repair/aligned.py --data /path/to/data-v1 --out /path/to/aligned
python research/torsion_repair/train_programs_mlx.py --aligned /path/to/aligned --model /path/to/model --chem-python /path/to/chemistry/python --out /path/to/training
python research/torsion_repair/compare_validation_controls.py --aligned /path/to/aligned --out /path/to/validation-controls.json
python research/torsion_repair/controls.py --data /path/to/data-v1 --aligned /path/to/aligned --names /path/to/names.json --out /path/to/controls
python research/torsion_repair/sage_geometry.py --archive /path/to/targets.tar.gz --data /path/to/data-v1 --names /path/to/names.json --out /path/to/sage --limit 95 --continue-on-error
python research/torsion_repair/sage_episodes.py --sage /path/to/sage --data /path/to/data-v1 --out /path/to/sage-supervision --role train --seeds 3
python research/torsion_repair/export_patch.py --sage /path/to/two-example-sage --archive /path/to/targets.tar.gz --out /path/to/export-check
```

Use MLX Python only for the adapter scripts and chemistry Python for QM/force-field replay. AM1-BCC requires AmberTools on PATH **before OpenFF toolkit registry initialization**. Mapping labels are stripped from the toolkit copy to avoid artificial stereo ambiguity; the original index mapping is retained for scope verification. Real undefined stereochemistry remains an error, not an instruction to invent a state.

`--continue-on-error` records chemistry failures; it does not substitute a different force field, charge model, molecule or QM label. Local sharding partitions molecules only, preserving the physical calculation. Repaired term exports are topology-specific additions to a fixed background, not a globally reusable OFFXML chemical-type replacement.

See the next-stage results and model card under `results/next-milestones/`. Passing numerical/export checks is separate from demonstrating held-out chemical or MD accuracy. The live website remains on the baseline route until the research validation gate is met.

### Completed next-stage evidence

The [next-stage report](results/next-milestones/REPORT.md) and [adapter model card](results/next-milestones/MODEL_CARD.md) document the64-update program-SFT experiment, matched64-episode controls, full168-molecule ablations,95-parent actual-Sage training corpus and12-parent Sage validation corpus. Checkpoint selection kept the unchanged model. The300-molecule candidate test remains outcome-blind. [Corrected research example packages](results/next-milestones/corrected-sage-examples.zip) passed18 full-system engine/readback checks; native NAMD remains untested. The [reproducibility archive](results/next-milestones/next-milestones-reproducibility.zip) includes trained adapters, optimizer states, derived data and audit logs, excluding base-model weights. No model has yet been trained on the new Sage corpus.

For matched supervision controls, pass `--history /path/to/training/training.jsonl` to `controls.py`. Full-data controls consume all285 teacher episodes; matched controls consume exactly the64 actual model episodes. `labels_matched_two` uses only the teacher answer labels, while the regret predictor uses the full utility vector computed from the same QM data.

## Sage-only adapter experiment

`prepare_sage_training.py` freezes a separate actual-Sage run. `train_sage_mlx.py` initializes a new adapter on the original base model and trains canonical JSON programs. It does not load the older UFF adapter. The numerical layer in `sage_runtime.py` fits summed parent/local features; model acceptance sees observed support labels only.

Both the untuned and trained policies use the same `program_language.py` decoder: 56 canonical answers consisting of one/two programs from ten shared parent/local harmonic representations, or abstention. This avoids option-letter positions and enforces schema validity. It is a bounded library, not unrestricted SMIRKS discovery. A tested absorbing EOS state handles MLX's one-token lookahead. Validity enforcement is shared with the untuned control and must not be counted as a fine-tuning benefit.

The [frozen development protocol](sage_training_protocol.json) specifies one shuffled pass over all95 parents,95 updates, checkpoints0/48/95, and12 separate validation scaffolds. Checkpoint selection uses accepted-program centered relative-energy RMSE at8support+8internalpoolholdout and includes the zero-update model. Descriptor, answer-label and retrieval comparators use the exact same95 parent/seed episodes;285-episode versions and larger CV/BMA libraries are references. Gaussian fitting and priors are shared. Secondary fixed/adaptive error-budget curves on old validation conformers are scored after checkpoint selection and do not select the checkpoint.

```bash
python research/torsion_repair/prepare_sage_training.py --train /path/to/sage-supervision95 --validation /path/to/sage-supervision-validation12 --out /path/to/sage-aligned
python research/torsion_repair/sage_controls.py --aligned /path/to/sage-aligned --out /path/to/sage-controls
python research/torsion_repair/train_sage_mlx.py --aligned /path/to/sage-aligned --model /path/to/local/model --chem-python /path/to/chemistry/python --out /path/to/sage-training
python research/torsion_repair/sage_controls.py --aligned /path/to/sage-aligned --out /path/to/selected-validation --plans /path/to/sage-training/validation-NNN/plans.json --secondary
python research/torsion_repair/analyze_sage_training.py --aligned /path/to/sage-aligned --training /path/to/sage-training --controls /path/to/sage-controls --out /path/to/sage-report
```

The external300-molecule cohort is not opened by this experiment. Validation intervals are descriptive:12 groups, one training seed, and checkpoint/control selection on the same development data. A validation improvement does not establish fresh chemical or MD accuracy.

The Sage comparison also records a **supplemental training-global constant-plan diagnostic**. It searches the same56 possible answers using only the matched95 training episodes and the unchanged support-only acceptance gate. This was added after partial48-step outputs, so it is explicitly posthoc. The frozen primary controls and their values remain unchanged; their original files are preserved in the reproducibility bundle. Its purpose is to check whether a learned policy improves on a useful constant recipe, rather than to retune the model on validation outcomes.

### Completed actual-Sage fine-tuning

The [Sage-only report](results/sage-program-finetune/REPORT.md) and [model card](results/sage-program-finetune/MODEL_CARD.md) document the completed95-update run on95 parents. Initial validation RMSE fell from0.7315 for the untuned constrained LLM to0.6374, exactly matching Bayes. The trained planner abstained on all12 validation molecules and retained the default fitted parent-H3 correction. It did not outperform Bayes or the stronger controls.

The secondary learning-curve result is also negative: fine-tuned and Bayes both score0.6602, compared with0.6383 for the untuned LLM and0.5998 for the matched two-proposal Bayesian model-averaging control. The teacher/selection objective focused on the initial8-point repair, while a useful acquisition plan needs complete-trajectory utility. No checkpoint was reselected using the secondary metric, and the external300-molecule cohort was not accessed.

[Checkpoint/optimizer/data reproducibility bundle](results/sage-program-finetune/sage-program-finetune-reproducibility.zip). Base7B weights are excluded and identified by hash. `analyze_sage_trajectory.py` reproduces the secondary descriptive comparisons; the12-group intervals do not establish external chemical or MD accuracy. No experimental model was deployed to the live website.
