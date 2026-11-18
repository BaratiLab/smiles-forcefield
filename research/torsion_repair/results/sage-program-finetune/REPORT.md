# Actual-Sage LLM fine-tuning and validation

A separate Qwen2.5-7B program adapter completed95 updates on geometry-complete Sage data. It was initialized from the original quantized base, not the earlier UFF adapter. The external 300-molecule cohort was not opened or scored.

## Frozen experiment

- 95 distinct training parents,285 available support episodes, one shuffled pass consuming95 exact parent/seed episodes.12 separate validation parents/scaffolds. No train/validation parent or scaffold overlap.
- Background: Sage2.2.1 unconstrained, fixed AM1-BCC charges. Features sum cosine/sine contributions over every affected parent/local proper-torsion instance. The numerical solver uses full-precision values and estimates all coefficients.
- Program actions choose broad/local representation and maximum harmonic order from1/2/3/4/6, with abstention. At most two proposals plus the unchanged model. This is a bounded10-representation library, not unrestricted chemical typing discovery.
- Canonical JSON completion-only SFT; last-two-layer q/v LoRA rank8, scale8,180224 trainable parameters. BF16 base floats, FP32 adapter/loss; AdamW5e-5, weight decay0.01, norm clip1. One training seed.
- Identical finite-program decoder for the base and trained checkpoints:56 canonical legal answers, no A–E options. Grammar enforcement is a shared engineering control, not a fine-tuning gain. Training remains standard full-vocabulary completion-token CE; grammar masking is applied at generation, so syntactic validity does not guarantee downstream utility.
- Checkpoints0/48/95. Selection minimizes molecule-mean centered energy RMSE on8 observed+8 internal pool holdout validation conformers, including step 0 and earliest exact tie. Support-only leave-one-out validation, with a fixed2% improvement gate, accepts proposals. Evaluation labels never enter that acceptance step.

The numerical runtime reproduced the previously recorded Bayes value before training. An initial decoder smoke attempt stopped before any update because MLX evaluates one token beyond EOS. An absorbing stop-token state fixed this integration issue and was regression-tested; no training or outcome-selected retry was hidden.

## Primary validation results

Metric: centered relative-energy RMSE, kcal/mol, averaged over12 validation parents. These are development molecules, including overlap with the background force field's fitting archive; they are not an independent chemical accuracy test.

| Checkpoint | Mean RMSE |
|---|---:|
| 0 | 0.731505 |
| 48 | 0.731931 |
| 95 | 0.637374 |

| Control | Mean RMSE |
|---|---:|
| bayes | 0.637374 |
| bma_library | 0.817374 |
| bma_two | 0.761714 |
| cv_harmonics | 0.677407 |
| cv_library | 0.655757 |
| fixed_two | 0.667138 |
| global_matched | 0.640684 |
| labels_matched | 0.637374 |
| learned_full | 0.649963 |
| learned_matched | 0.637374 |
| retrieval_full | 0.744754 |
| retrieval_matched | 0.672799 |

Selected deployment candidate: **step 95**, RMSE0.637374. Best nonzero checkpoint: **step 95**, RMSE0.637374. The strongest matched-budget non-LLM control selected on validation is `labels_matched` (0.637374).

Best trained versus Bayes: +0.00% relative change (negative is better); mean difference+0.000000, descriptive97.5%paired interval[0.0, 0.0].

Best trained versus strongest matched control: +0.00%; descriptive97.5%paired interval[0.0, 0.0].

Best trained versus identically decoded untuned LLM: mean difference-0.094131, descriptive97.5%paired interval[-0.3455689499978154, 0.003368074601548216].

Point-mean improvement flags: versusBayes=False; versusstrongestmatchedcontrol=False. These flags are not confirmatory hypothesis tests.

## Fairness and uncertainty

Matched learned, answer-label and retrieval controls consume exactly the same95 parent/seed teacher episodes used by the adapter. Regret predictors use full utility vectors derived from the same QM data; answer-label predictors and retrieval use the teacher program answers. All get the same observed state, graph and physics; numerical feature transforms and objectives differ. Full285-episode references and10-representation CV/BMA libraries are reported separately. BMA uses Gaussian marginal evidence and mixture-variance acquisition, not exact cross-model mutual information.

Scaffold bootstrap intervals use20000 resamples of12 independent validation groups. There is one training seed, and both checkpoint and comparator selection use the same validation set. The intervals are descriptive and do not adjust for that selection. A lower validation mean is not evidence of external generalization or MD accuracy.

Large gradients were finite and clipped; 95/95 updates exceeded the clipping threshold, with maximum preclip norm4.71e+09. All saved adapter/optimizer tensors were separately checked for finite values. Training loss endpoints came from different examples and are not an unbiased learning curve.

## Secondary replay and validity

After checkpoint selection was frozen, selected-model proposals were replayed across budgets8/10/12/14/16 using old validation evaluation conformers, with fixed and adaptive queries. These curves do not choose the checkpoint. Results are in `selected-validation/secondary-curves.json`. The numerical no-edit baseline is always available. The original external 300-molecule cohort remains sealed regardless of the validation result in this run.

Checkpoint 0: 0 parsing errors, 0 compiler rejections, mean local generation22.3s per validation molecule.

Checkpoint 48: 0 parsing errors, 0 compiler rejections, mean local generation25.2s per validation molecule.

Checkpoint 95: 0 parsing errors, 0 compiler rejections, mean local generation19.5s per validation molecule.

The final checkpoint abstains on all 12 validation molecules. Abstention retains the fitted parent H3 Bayes correction; it does not mean reverting to an unfitted raw Sage potential. The model therefore adds no representation-selection accuracy over Bayes on these validation cases.

### Secondary learning-curve values

| Method | Adaptive mean error AUC | Fixed-query mean error AUC |
|---|---:|---:|
| Bayes | 0.660176 | 0.660176 |
| Fine-tuned selected LLM | 0.660176 | 0.660176 |
| Untuned LLM | 0.638349 | 0.640150 |
| Fixed two | 0.605798 | 0.613247 |
| Matched learned utility predictor | 0.603595 | 0.607298 |
| Matched two-proposal model averaging | 0.599786 | 0.641908 |
| Training-global constant plan (posthoc) | 0.584967 | 0.585875 |
| Larger ten-representation CV reference | 0.565095 | 0.604413 |

These values use the other 8 conformers of the same12 old validation molecules; they are not the external 300-molecule cohort. Fine-tuning improves the initial-fit mean over the untuned LLM but worsens its learning-curve mean. The trained planner supplies no alternative models for later Bayesian selection because it abstains. Its teacher/selection objective rewarded the initial8-point repair score rather than complete-trajectory utility. This is a concrete objective mismatch to investigate, not a causal ablation or a new performance claim. No checkpoint was reselected using these secondary results.

## Reproducibility and remaining scope

The bundle includes the frozen protocol, exact train/validation states and summed-feature data, all checkpoint programs/scores, actual adapters and optimizer states, source code, model hashes, numerical controls and descriptive analysis. Base 7B weights are external and are not redistributed. The prior geometry/export bundle supplies source Cartesian coordinates and background systems if those preparation stages need to be rebuilt.

No new QM target calculations, paid cloud jobs or website deployment occurred. Native-engine accuracy checks are outside this training experiment; previous export-consistency tests do not establish QM or thermodynamic accuracy for newly selected repairs. Any external test, prospective QM campaign or public repair deployment is a separate step. Multiple seeds, independent off-grid energies/forces, background fitting-overlap audits and a genuinely untouched test are still required for a paper-level accuracy claim.
