# Next milestones: completed development work and remaining evidence gates

The aligned repair-program fine-tuning run, stronger numerical controls, geometry-complete Sage data pipeline, corrected export validation and outcome-blind external cohort preparation are complete. The current adapter does not improve validation, so the fresh cohort has not been used and experimental refinement is not enabled on the live website.

## 1. Actual program-supervised training

We generated 285 aligned episodes from 95 original training parents and selected 12 separate validation scaffolds by an outcome-independent hash. The initializer exactly matches the deployed v1 pilot: four hash-initial points and four covariance-based Bayes selections. Own-parent examples are excluded from teacher prior retrieval. Internal remaining pool points provide teacher utilities; the dataset test field does not enter teacher selection or checkpoint scoring.

The local Qwen2.5-7B 4-bit model was trained with completion-only JSON program token loss, not A–E classification. Last-two-layer q/v LoRA has 180,224 trainable parameters. Base floating tensors are BF16, adapter and loss FP32. All 64 updates completed; adapter and optimizer files passed a separate finite-tensor check. This bounded run actually consumed 64 distinct parent episodes from the 285-episode pool, not all episodes or all 95 parents. It is one training seed.

| Checkpoint | Separate validation RMSE, kcal/mol |
|---|---:|
| Before training | 0.958627 |
| Update 32 | 0.958627 |
| Update 64 | 0.958627 |

The frozen lowest-error/earliest-tie rule selects **step 0**, the unchanged model. The best nonzero checkpoint is step32 and is retained only as a research diagnostic. On the same retired 24-molecule pilot it matches Bayes exactly: mean error-query AUC 1.618874 versus 1.618874. All molecules retain the numerical no-edit model. The trained generator produced 26 invalid programs out of 48 (primarily an abstention with local scope, sometimes extra fields); the compiler rejected them. JSON parsing alone is not a validity test. Mean generation time was 13.9 seconds per molecule. A zero-adapter repeat exactly reproduced a cached base response; model hashes and every prompt hash were checked before cached-base reuse.

This is a negative utility result, despite a genuine weight update. Teacher answers are mostly abstention (162/285 episodes); mode collapse and a token-loss/utility mismatch are plausible explanations, not isolated causal findings. All updates were clipped and some preclip gradients were very large (maximum 1.66e+09); finite tensors do not establish robust optimization. Loss endpoints came from different examples and are not a controlled learning curve. More data, cost-aware objectives and schema-constrained decoding warrant development tests before any fresh claim.

## 2. Strong controls and acquisition ablation

The full 168-parent, 143-scaffold retired cohort now has fixed two-program, learned descriptor/Morgan, retrieval, Bayesian model averaging and larger-library CV/BMA comparisons. Every policy has the same observations, shared coefficient priors and compiler. The fixed-query version reuses the complete Bayes trajectory; adaptive versions choose subsequent points using their revised model. Model averaging uses Gaussian marginal evidence with the energy offset projected out; mixture variance is an acquisition approximation, not exact cross-model mutual information.

Metric: centered held-out relative-energy RMSE averaged over query budgets 8/10/12/14/16, in kcal/mol. It is not the old standard-deviation-normalized AUC, and cohort means must not be compared across different molecule sets.

| Method, adaptive queries | Mean error AUC | Change vs Bayes |
|---|---:|---:|
| Fixed-model Bayes | 1.075444 | +0.00% |
| Fixed two proposals | 0.979531 | -8.92% |
| BMA over two proposals plus baseline | 0.982626 | -8.63% |
| CV, ten representations | 0.992738 | -7.69% |
| Learned utility predictor, full285 episodes | 1.042540 | -3.06% |
| Learned utility predictor, matched64 episodes | 1.067737 | -0.72% |
| Learned answer-label predictor, matched64 | 1.074221 | -0.11% |
| Retrieval, full285 episodes | 1.052363 | -2.15% |
| Retrieval, matched64 episodes | 1.083517 | +0.75% |
| BMA, ten representations | 1.193968 | +11.02% |

The two fixed proposals are broad H2 and local H4, with no-edit retained. The strongest full-data controls are references, not identical training budgets: the adapter consumed64 episodes while those predictors used285. Matched controls now use exactly its64 molecule/seed episodes. A matched answer-label predictor uses only teacher program membership; the matched regret predictor uses richer cost-vector labels derived from the same QM observations. Descriptor predictors compute an H6 fit from observed support; the LLM sees the original state. Information sources and QM data are shared, but representations and training objectives differ. All non-library proposers remain capped at two programs.

Much of the fixed-two gain persists with fixed queries (0.976563 versus0.979531 adaptive), so the improvement is primarily model revision in this replay, not evidence that LLM query selection helps. Large-library BMA performs worse; more hypotheses are not automatically beneficial. The small validation subset has a different method ordering: retrieval is strongest among the original matched-budget full-data controls. This reinforces the need to select methods before an external test, rather than choose a winner from the168 development results.

## 3. Actual Sage geometry and specialization

The real force-field pipeline now prepares **all95 training parents and all12 separate validation parents** with Sage2.2.1 unconstrained and fixed AmberTools AM1-BCC charges on each molecule's24 archived Cartesian frames. It stores the complete MM background, QM energies, coordinates, molecule graph, assignments and source hashes. The new feature matrix sums cosine/sine terms over **every proper-torsion instance assigned to the selected parent parameter**, or the verified narrower environment subset. This fixes the prior driven-angle-only surrogate for these data.

Two validation molecules initially failed RDKit conformer generation. The same AM1-BCC method was retried with the fixed first pool-frame geometry, without changing chemical state or selecting a geometry by its energy. Their initial failures and recovery provenance remain archived. No alternative charge model was silently substituted.

A new corpus contains285 actual-Sage training episodes plus12 validation episodes. Its specialization changes the set of affected torsion instances, rather than selecting the old environment-conditioned residual prior. These inputs and scope semantics differ from the completed UFF adapter. **No adapter has yet been trained on the new Sage corpus**, and the old adapter is not described as Sage-trained.

Initial geometry-complete validation controls at8support+8internalpoolholdout give:

| Numerical control | Mean Sage residual RMSE, kcal/mol |
|---|---:|
| Default parent H3 correction | 0.637374 |
| Fixed two proposals | 0.667138 |
| Learned two proposals | 0.646654 |
| Harmonic CV | 0.677407 |
| Parent/local CV library | 0.655757 |

Model revision does not improve this small validation cohort on average. This is a further development gate, not an external accuracy test: these molecules belong to the published fitting archive of the baseline force field. Higher harmonics may fit scan-specific nuisance variation; off-grid conformers and QM forces remain necessary. The default H3 correction is itself a fitted delta on Sage, not an unchanged-Sage accuracy number.

## 4. Corrected force-field packages and engine checks

Two development molecules were exported at three geometries each. Their patches are explicit topology-specific Interchange additions to the fixed background, not globally reusable OFFXML type replacements. In one example a parent type has two instances and the selected specialization patches only one. Across all95 training molecules, summed-feature energies agree with the explicit patched OpenMM system to numerical precision on all24 frames.

All18 full-system export checks passed: native GROMACS, native LAMMPS and independent AMBER readback for each of six geometries. Maximum energy discrepancy is8.45e-5 kJ/mol and maximum force-component discrepancy0.0176 kJ/mol/nm, below the pre-existing0.01/0.1 tolerances. Native NAMD is unavailable; its compatible AMBER package has readback validation only. Passing these tests proves export consistency, not QM force accuracy, thermodynamics or MD transferability.

## 5. New outcome-blind QM cohort

We queried the public QCArchive catalog and inspected only structure entries, completion statuses and specifications for dataset472, TorsionNet500 Re-optimization TorsionDrives v4.0. It has497complete,2error and1running records at the snapshot. The specification is Psi4 B3LYP-D3BJ/DZVP, consistent with the original target theory. No final energy/gradient outcome records were retrieved.

After excluding758 previously inspected parents and385 inspected scaffold groups, known fitting parents from the919-parent Sage archive, duplicate parents and incomplete statuses, **353 candidate parents across308 scaffold groups** remain. All353 pass the structure-only Sage vdW/bond/angle/proper coverage check. That does not guarantee charge convergence or24 valid final geometries.

A structure-hash selection seals **300 candidate test parents across262 scaffold groups**, with53 reserved. Membership, metadata, exclusions and hashes are saved. Missing/inconsistent scans will be reported with predeclared checks, not replaced based on performance. The planning power sensitivity uses only retired controls and assumptions about future variance; it is not a guarantee of80% power. Original TorsionNet/pretrained-model exposure is unknown, and the archive exclusion does not audit every possible charge/nonbonded fitting source.

The new test remains untouched because current validation does not support an LLM advantage. A UFF replay test and a geometry-complete Sage repair test are distinct tracks. Before using the new cohort for the paper, train/select a model on the actual Sage corpus, add appropriate validation coverage, freeze all endpoints and controls, and satisfy the development gate.

## 6. Community and paper status

The implementation, corrected example packages, model card, data/proposal/optimizer provenance, analysis and reproducibility bundle are public source deliverables. The live website has not been redeployed and research corrections are not automatically installed in public downloads. Baseline package generation remains independent of the experimental model.

Local checks:19 API/repair tests plus34 existing acquisition integrity tests passed. The optional OpenFF geometry/coverage test runs locally; minimal GitHub CI may skip it because it does not install the chemistry stack. The18 native/readback engine checks are a separate local result. Base-model weights are not bundled, but the trained adapters and exact hashes are included. Existing published B3LYP target data were reused; local AM1-BCC semiempirical charge calculations were performed. No new target-level QM campaign or paid cloud job was launched.

The remaining paper gates are scientific: successful actual-Sage program training against strong matched controls, independent/off-grid energy and force validation, multi-seed uncertainty, baseline-training overlap audits and comparisons appropriate to published graph-to-parameter force fields. A prospective QM campaign and public refinement deployment follow a successful frozen retrospective study. The current work establishes infrastructure and negative validation evidence; it does not establish an LLM accuracy contribution.
