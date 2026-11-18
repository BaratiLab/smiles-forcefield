# QM-supervised BF16 QLoRA acquisition: fresh test

**Primary result: no reliable improvement over pure Bayes was demonstrated.** The primary comparison is the fine-tuned model minus Bayes in molecule-mean normalized error-query AUC, with a paired 95% scaffold-cluster bootstrap interval. Lower is better.

## Frozen design and fresh data

Training: 95 molecular parents / 37 scaffold groups. Validation: 30 parents / 25 groups. Fresh test: 168 parents / 143 groups. No train/validation/test scaffold overlap. Fresh validation and test also exclude every scaffold from the previous experiment, and all 460 eligible molecules in the previous test scaffold groups are quarantined entirely. All selected QM records were independently verified complete at Psi4 B3LYP-D3BJ/DZVP before teacher generation; no QC exclusions were needed.

The same complete 24-point 1D torsion scans, 16-point acquisition pools, 8-point within-molecule evaluation sets, Fourier harmonics 1–3, UFF baseline, Gaussian prior and fixed noise as the previous experiment are used. Five matched seeds, four initial labels, batches of two, twelve labels total. No new quantum calculations were commissioned.

## Numerical repair before test scoring

The original FP16 attempt failed a nonfinite check at update 84. Its saved checkpoints and partial unlabeled fresh predictions were preserved as diagnostic artifacts. No fresh QM acquisition outcomes had been scored. The repaired trial used BF16 compute, checks on every loss/gradient before optimizer updates, and saved optimizer checkpoints. Splits, QM supervision, hyperparameters, model architecture, checkpoint selection and the primary comparison were unchanged. The complete 96-update schedule was then run from initialization; this report evaluates that stable retry, not the interrupted model.

## What was fine-tuned

Qwen2.5-7B-Instruct, using the same pinned 4-bit base weights. Only 180,224 LoRA parameters on query/value attention projections in the last two decoder blocks are updated (rank 8, scale 8, dropout 0). Floating base tensors are cast to BF16 before freezing; quantized integer weight tensors are unchanged. Adapter tensors are FP32. The matched untuned baseline uses the same BF16 representation. The optimizer is AdamW with learning rate 2e-4, weight decay 0.01, norm clipping 1.0, seed 20261118.

The supervision is a QM-derived acquisition preference, not raw energy text. For each training and validation molecule only, all five existing strategies are replayed over five seeds. Their mean normalized AUC values define a soft action target proportional to exp(-cost/0.05). Training minimizes soft cross entropy over five action-letter logits. Each update averages all five cyclic option permutations for one molecule, exactly balancing strategy-to-letter placement. There are 96 updates / 480 gradient microsteps.

The compact prompt contains the molecular graph with driven atoms mapped, bond-specific descriptors and unlabeled pool angles, plus the five score formulas. There are no energies, teacher costs, source record IDs or few-shot examples in the prompt. The matched untuned base model uses exactly the same prompt and decoding. This is supervised cost-sensitive QLoRA, not reinforcement learning and not direct force-field parameter generation.

Checkpoints at updates 24, 48, 72 and 96 were compared on validation acquisition AUC. Update 96 was selected at validation AUC 0.618098; base validation AUC was 0.702559. The selection rule and hyperparameters were frozen before training. Only after checkpoint selection were the two fresh model decisions generated, followed by fresh QM outcome scoring. No retraining followed fresh test inspection.

## Fresh test performance

AUC is the trapezoidal mean centered test residual RMSE across 4–12 queries, normalized by each molecule’s centered QM-minus-UFF baseline RMSE, then averaged over seeds and molecules. The normalizer is used only by the evaluator.

| Policy | Normalized AUC | AUC (kcal/mol) | Final RMSE (kcal/mol) |
|---|---:|---:|---:|
| bayes | 0.668255 | 1.433061 | 1.039256 |
| rule | 0.679213 | 1.447028 | 1.041716 |
| base_llm | 0.699884 | 1.530259 | 1.123126 |
| finetuned_llm | 0.691977 | 1.487912 | 1.098276 |
| training_global | 0.668255 | 1.433061 | 1.039256 |
| extra_trees | 0.668672 | 1.432003 | 1.042100 |
| random | 0.901185 | 1.894323 | 1.320640 |
| fixed_angular_diversity | 0.671423 | 1.442181 | 1.041165 |
| fixed_sector_balance | 0.670828 | 1.436126 | 1.038559 |
| fixed_orthogonal_focus | 0.704451 | 1.496105 | 1.048368 |
| fixed_cis_focus | 0.981449 | 2.059686 | 1.591435 |

The fine-tuned model has 3.55% higher mean normalized acquisition error than Bayes (0.691977 versus 0.668255). The primary paired difference is +0.023722, with 95% interval [+0.008205, +0.044015]. Because the entire interval is positive, this trial provides evidence of worse acquisition performance, rather than merely failing to establish a gain. The small point improvement over the matched base LLM is not reliable: its secondary interval includes zero. Bayes should remain the default for this benchmark. This finding is conditional on the small training set, single fine-tuning seed and fixed action menu; it is not a conclusion that every LLM-guided design must fail.

## Paired comparisons

Negative differences favor the fine-tuned LLM. Seeds are averaged before resampling whole scaffold clusters. Only the comparison against Bayes is the prespecified primary 95% interval; other intervals are descriptive secondary comparisons.

| Comparator | Fine-tuned minus control normalized AUC | 95% cluster interval | Wins / ties / losses |
|---|---:|---|---|
| bayes | +0.023722 | [+0.008205, +0.044015] | 52 / 43 / 73 |
| rule | +0.012764 | [-0.005864, +0.034257] | 66 / 30 / 72 |
| base_llm | -0.007907 | [-0.034798, +0.018482] | 45 / 79 / 44 |
| extra_trees | +0.023305 | [+0.008106, +0.042950] | 51 / 41 / 76 |
| training_global | +0.023722 | [+0.008316, +0.044253] | 52 / 43 / 73 |

The learned non-LLM control is an ExtraTrees multioutput regressor predicting each strategy’s regret relative to Bayes. It is trained on the same training molecules and QM supervision, using 256-bit Morgan radius-2 features and chemical/angular coverage descriptors. Its parameters are frozen (256 trees, min leaf 3, max features 0.5); validation/test labels are not used to fit it. All fixed strategies, the training-global best strategy and random acquisition are also included.

A retrospective per-molecule best-in-menu oracle reaches normalized AUC 0.637864. This oracle uses fresh outcomes after evaluation and is not a deployable baseline or a trained model; it only describes the finite strategy menu’s headroom.

## Training, behavior and latency

Measured training plus validation took 34.90 minutes, excluding model loading and preparation. The selected adapter SHA256 is `9d5d30fff4d7fdea0ac25d3ae65ad03f1b24b3ddde3facdfabf7f2e65343ad46`. Full training loss, gradient norms, checkpoint decisions and scores are retained.

base: strategy counts `{"angular_diversity": 139, "cis_focus": 9, "orthogonal_focus": 12, "sector_balance": 7, "uniform": 1}`; letter counts `{"A": 52, "B": 46, "C": 30, "D": 40}`. 168 decisions took 506.0 seconds; median 2.993 seconds per molecule.
finetuned: strategy counts `{"angular_diversity": 81, "cis_focus": 6, "orthogonal_focus": 5, "sector_balance": 34, "uniform": 42}`; letter counts `{"B": 60, "C": 67, "D": 30, "E": 11}`. 168 decisions took 505.9 seconds; median 2.992 seconds per molecule.

Option frequencies diagnose behavior but do not establish counterfactual order invariance. Balanced training covers all five positions per molecule; each fresh model is called once per molecule, not five times. Inference and training ran locally on the Apple M2 GPU, with no new AWS resources or paid model API.

## Audit and interpretation limits

All dataset, adapter and prompt hashes are checked. Teacher targets contain only training/validation molecules. Training history uses only training parents. Validation scores are independently reconstructed from validation teacher costs. Fresh prompts are checked against the label-free allowlist; base and tuned prompts are identical. Every policy has the same initial set, twelve unique pool-only acquisitions, and no evaluation geometry IDs. Both LLMs and the learned selector exactly reproduce their corresponding fixed-strategy controls.

This is a small fine-tuning pilot with only 95 training parents and 37 training scaffold groups, and a single fine-tuning seed. Intervals quantify scaffold/molecule variation conditional on this selected adapter, not variability across training seeds. Fresh scaffold separation does not guarantee low fingerprint similarity, and public molecules may have appeared in LLM pretraining. The complete-scan selection biases coverage. Tests interpolate within a constrained scan; they do not validate new conformer families. The geometry pool itself was QM-optimized, so replay does not establish prospective QM time savings. The correction fit remains a fixed linear Gaussian model: acquisition within the chosen plan uses geometry and covariance rather than observed energies. No MD accuracy, force-field export improvement or general-purpose force-field learning is established. Do not tune on this fresh set and call another run independent.

## Sources and reproduction

- [Pinned public OpenFF Sage 2.2.1 QM fitting data](https://github.com/openforcefield/sage-2.2.1).
- [MLX-LM LoRA/QLoRA implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).
- [Qwen2.5-7B-Instruct quantized model](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit).

The companion archive contains frozen protocol and splits, source-member hashes, selected QM/UFF data, teacher values, all model prompts and decisions, every adapter checkpoint, validation selection records, baseline predictions, per-seed metrics and candidate IDs, and source code. Base model weights are referenced by hash and are not redistributed.

Independent source verification checked 7032 QM frames and all source-file hashes. Nearest-training Morgan radius-2/2048-bit Tanimoto similarity on fresh test molecules has median 0.202, 95th percentile 0.448 and maximum 0.565.
