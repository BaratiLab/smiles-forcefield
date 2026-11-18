# Held-out QM acquisition benchmark

**Outcome: the prespecified LLM-improvement criterion did not pass.** The criterion requires lower mean normalized error-query AUC than both pure Bayes and the frozen chemistry rule, with each paired scaffold-cluster 97.5% confidence interval entirely below zero.

## Experiment

299 valid unique held-out molecular parents across 178 scaffold groups; 60 development molecules across 37 disjoint groups. Five matched initialization seeds per molecule; 4 initial labels, batches of 2, 12 labels total, 16-point acquisition pools and 8-point unseen conformer tests. The source consists of complete 24-point one-dimensional published QM torsion scans. All 360 selected records have independently verified Psi4 B3LYP-D3BJ/DZVP specifications (including the synonymous method spelling b3lyp-d3(bj)). Ten policies give 15,000 archived trajectories, of which 14,950 use the 299 valid QM references. The original split selected 300 held-out parents. Independent QCArchive metadata verification, before outcome scoring, found record 1762338 marked invalid; it is excluded from primary inference without replacement. This is an explicit quality-control amendment to the original 300-molecule protocol, with all raw results retained.

The target is a Fourier correction to UFF energies at the identical QM geometries. QM labels are from OpenFF Sage 2.2.1 ForceBalance targets, with energies converted from hartree to kcal/mol. UFF is evaluated without minimizing geometry. These are real QM labels, but this is a finite-pool acquisition replay, not an MD-validation study or an export-accuracy study.

## Planner changes

Qwen2.5-7B-Instruct in 4-bit MLX format replaces the 1.5B model. The prompt contains the mapped molecular graph, driven atom maps, bond-specific chemistry flags and unlabeled pool angles. Five feature-only development examples are selected by a frozen, per-strategy best-margin rule on the 60 development molecules. No held-out energies, record IDs, or evaluation statistics enter a planner request. One constrained A–E next-token decision chooses a Bayesian acquisition weighting, and that plan is reused for all five seeds. This is in-context learning, not weight fine-tuning. All option log probabilities, prompts, model-file hashes and decisions are saved.

The same Fourier harmonics 1–3, Gaussian prior and fixed observation noise are used in every policy. All strategies choose by information gain; extra actions multiply it by diversity, sector balance, orthogonal-angle emphasis, or cis-angle emphasis. The rule uses the same bond descriptors: amide-like or conjugated -> orthogonal focus; otherwise both ends aromatic -> cis focus; otherwise uniform. The best global development strategy and a chemistry-blind development-frequency draw are additional non-LLM controls.

## Held-out metrics

Molecule means after averaging seeds. Lower is better. Normalized AUC is the trapezoidal mean error across 4–12 queries divided by the centered baseline QM-minus-UFF test RMSE. This normalizer is used by the evaluator only.

| Policy | Normalized AUC | AUC (kcal/mol) | Final RMSE (kcal/mol) |
|---|---:|---:|---:|
| bayes | 0.60552 | 1.41209 | 1.02547 |
| rule | 0.61458 | 1.44211 | 1.02135 |
| llm_plan | 0.61402 | 1.43926 | 1.03717 |
| random | 0.84594 | 1.89669 | 1.30267 |
| development_global | 0.60552 | 1.41209 | 1.02547 |
| chemistry_blind | 0.63230 | 1.46605 | 1.05247 |
| fixed_angular_diversity | 0.61028 | 1.41905 | 1.02254 |
| fixed_sector_balance | 0.61481 | 1.42325 | 1.02763 |
| fixed_orthogonal_focus | 0.63992 | 1.49857 | 1.03199 |
| fixed_cis_focus | 0.88926 | 1.99807 | 1.45781 |

The LLM normalized AUC is 0.61402 versus 0.60552 for pure Bayes and 0.61458 for the rule. Its point estimate is 1.40% worse than Bayes and 0.09% better than the rule. Neither primary paired interval establishes improvement. Pure Bayes is also the best fixed strategy by mean held-out normalized AUC and was selected as the development-global baseline. A simple Bayes default remains justified on this benchmark.

## Paired inference

Effects below are LLM minus comparator. Negative favors the LLM. Seeds are averaged before inference; scaffolds, not trajectories, are the bootstrap clusters. After excluding the invalid reference, the two primary comparisons use 97.5% intervals to account for two comparisons. Additional-control intervals are descriptive.

| Comparator | Paired normalized AUC difference | 97.5% cluster interval | Wins / ties / losses |
|---|---:|---|---|
| bayes | +0.00850 | [-0.00289, +0.01945] | 129 / 5 / 165 |
| rule | -0.00056 | [-0.01050, +0.01060] | 147 / 17 / 135 |
| development_global | +0.00850 | [-0.00306, +0.01949] | 129 / 5 / 165 |
| chemistry_blind | -0.01828 | [-0.03967, -0.00167] | 126 / 83 / 90 |

## Behavior and latency

Across the original 300 planned molecules, strategies: `{"angular_diversity": 183, "cis_focus": 12, "orthogonal_focus": 32, "sector_balance": 69, "uniform": 4}`. Option letters: `{"A": 34, "B": 146, "C": 63, "D": 28, "E": 29}`. The latter is an order-bias diagnostic, not a counterfactual prompt-order experiment. Letter B was chosen for 146/300 molecules (48.7%) despite shuffled option order, which raises a position-preference concern; it does not by itself identify its cause. The chemistry-blind control uses development-optimal strategy frequencies, not the LLM strategy frequencies, so outperforming that control alone would not prove chemically meaningful molecule-to-strategy matching.

Exactly 300 decisions took 746.6 seconds of measured inference, median 2.29 seconds per molecule. There were 278,744 prompt tokens and 300 output tokens. One exact shared prompt-prefix prefill is included; model loading, model download and data preparation are excluded. Individual decision timings exclude that shared prefill. Logical prompt tokens include cached prefixes. Inference ran locally on Apple M2 via MLX, with no new AWS resources or paid inference API. These timings are not a controlled speed comparison with the earlier AWS run.

## Audits and limits

The audit checks all 360 input dataset hashes, all prompt hashes and allowlisted targets, parent uniqueness, scaffold disjointness, identical initial sets and query budgets, pool-only selections, and exact replay against the fixed control for every LLM plan. Distances are used to reject exact labeled-geometry duplicates at 1e-6 angstrom resolution after rigid-motion invariance; symmetry-equivalent conformers are not exhaustively deduplicated. Only one eligible scan per nonstereo molecular parent is retained, deterministically by record ID. No energy-window filtering is applied.

“Held out” means excluded from this project’s development examples and decision tuning. Absence from LLM pretraining cannot be guaranteed. The QM archive includes Sage fitting data; this benchmark does not claim held-out Sage force-field accuracy, and deliberately uses UFF as the baseline. Scaffold groups use ring Murcko scaffolds, with generic full-graph topology for acyclic molecules; they do not guarantee low pairwise molecular similarity. Across the original 300 planned molecules, nearest-development Morgan radius-2/2048-bit Tanimoto similarity has median 0.20, 95th percentile 0.371, and maximum 0.80. Pilot identities and pilot ring scaffolds were excluded.

The eight conformer tests come from the same constrained scan as the pool, so this tests interpolation across angles within a molecule, not generalization to new conformer families. The candidate geometries themselves were produced by QM constrained optimization; replay treats these geometries as available and therefore does not establish prospective QM wall-time savings. A prospective study must generate cheap candidate geometries and run QM only on acquired configurations. Complete one-dimensional scans bias chemical coverage. The finite menu, fixed Gaussian model and noise limit expressivity. In this linear-Gaussian design, acquisition depends on geometry and covariance rather than observed energy values. The LLM selects a chemical prior; it does not invent force-field coefficients or QM labels. A stronger prompt/model can still fail. Do not tune on these 300 test molecules and present a rerun as an independent held-out result.

## Sources and reproduction

- [Official OpenFF Sage 2.2.1 data and fitting repository](https://github.com/openforcefield/sage-2.2.1), pinned source commit `b70c449e3a98e40f0b6e4c6b2f27932fce47be42`.
- [Quantized model card and Apache 2.0 license](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit).
- [MLX-LM inference library](https://github.com/ml-explore/mlx-lm).

The frozen protocol, importer, development selector, planner, evaluator, tests and analysis are in `research/acquisition`. The reproduction README lists commands. Dataset and source hashes, full selected inputs, raw requests, model decisions, per-seed curves and selection IDs are retained in the companion archive.
