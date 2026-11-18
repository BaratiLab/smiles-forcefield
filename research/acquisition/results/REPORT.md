# First AWS acquisition comparison

Implemented and run: random sampling, Bayesian experimental design, and Qwen2.5 0.5B through private CPU-only Ollama on the existing AWS VM. All 15 molecule/method pairs finished. No additional server or paid LLM API was purchased.

**Main finding:** The LLM made valid selections, but every batch matched the first-available candidate order. A deterministic ordered-selection control reproduces all five LLM query sequences and learning curves (numerically checked at 1e-8 tolerance). This pilot demonstrates execution, not a useful LLM acquisition advantage.

## Matched pilot results

Held-out centered residual RMSE in kcal/mol; lower is better. One seed (0), four initial observations, total budget 12, batch size four, 32 pool and 32 test geometries per molecule. Same initial observations, fitter, prior, noise, labels and dataset hash for each method.

| Molecule | Random | Bayesian | Qwen 0.5B |
|---|---:|---:|---:|
| butane | 0.0149 | 0.0126 | 0.0156 |
| biphenyl | 0.8280 | 0.7738 | 0.5982 |
| anisole | 0.1100 | 0.0627 | 0.0764 |
| nma | 1.0126 | 0.8703 | 1.4348 |
| phenylethanol | 4.6122 | 1.9886 | 5.6721 |

Bayesian selection has lower final error on four of five molecules. The ordered/LLM sequence happens to have lower final error on biphenyl, but its integrated learning-curve error is worse than Bayesian selection. Bayesian selection beats the LLM on query-normalized error AUC for all five molecules. These are descriptive one-seed results, not significance or generalization claims.

## Oracle and model scope

The reference is RDKit MMFF94 and the baseline is UFF, evaluated at identical constrained-relaxed real geometries. These are **not quantum energies**, not Sage residuals, and not a trained deployable force field. Both pool/test sets originate from one embedded conformer per molecule. Independent conformer families and geometry-invariant duplicate checks remain needed. Test labels are excluded from controller state; this trusted-process replay is not an adversarial sandbox.

The fitter is an additive Fourier correction (harmonics 1–3, cosine and sine, plus intercept), Gaussian prior precision 1 and assumed noise 0.1 kcal/mol. Those choices are engineering defaults, not calibrated QM uncertainty. The two-dimensional case also tests model-capacity limits. Symmetry constraints, force fitting and engine export of learned corrections are not implemented.

## Runtime and audit

- Model: `qwen2.5:0.5b`, digest `a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67`.
- Prompt: compact-v2; row arrays, angles/energies rounded to four decimals, variance to six. Full precision is retained in fitting and saved original states.
- 10 real model calls, 10 attempts, no invalid final actions or repair retries.
- 13,039 prompt tokens and 613 output tokens; 9.18 minutes total inference time across calls.
- First-available order matched in 10/10 rounds. The ordered control reproduces the LLM results.
- Model server had one CPU and 2500-MB memory cap; benchmark runner had 0.5 CPU and 1-GB cap. The model server was stopped after the run; weights remain cached. The production API remained healthy.
- Initial Qwen 1.5B/full-JSON attempt was stopped for latency. It is a separate incomplete feasibility run, excluded from this comparison.
- Experiment code commit: ca65d17. Result exporter/docs: e481b37. Full model responses, prompts, selected IDs, hashes, posterior coefficients, metrics, and SDFs are in the raw AWS archive.

## What to test next

1. Shuffle candidate presentation and anonymize IDs; run counterfactual tests that change numeric uncertainties while holding chemical text fixed. Reject a controller that only follows list order before paying for QM.
2. Compare Qwen 1.5B with compact prompts and a hybrid: Bayesian scoring proposes a shortlist, while the LLM selects among chemically distinct experiment types. Include a score-only shortlist baseline.
3. Run at least several matched seeds and more candidate/test conformer families. Freeze prompt, basis and acquisition settings before a held-out chemistry evaluation.
4. Replace the proxy with a fixed QM protocol and matching Sage single points, then evaluate accuracy versus real QM cost. Only promote a model after independent physical and export validation.

## Artifacts

- [aws-pilot.zip](aws-pilot.zip): original AWS data, model logs, metrics and plot.
- `audit.json`: automated configuration/action audit.
- Ordered control: reproduce with the supplied `ordered` policy and the archive datasets. The local deliverable also contains the verified control runs.
- Separate local three-seed random/Bayesian development run; do not compare its averages directly with the one-seed LLM table.

Source: https://github.com/BaratiLab/smiles-forcefield/tree/main/research/acquisition
