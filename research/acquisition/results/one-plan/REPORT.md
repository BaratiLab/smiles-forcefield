# One LLM plan followed by Bayesian acquisition

The AWS experiment completed, but it does **not demonstrate an LLM advantage**. Qwen2.5:1.5B selected unweighted Bayesian information gain for every molecule. All 25 LLM-plan trajectories exactly match pure Bayesian selection, including candidate IDs and errors.

## Matched design

Five molecules × five policies × five initial seeds = 125 trajectories. Each run starts with the same four candidates for its seed and ends at 12 queries, in batches of four. One molecule-only LLM plan is reused across seeds; seeds measure acquisition sensitivity, not independent LLM decisions. No pool or test labels enter the planner. The same additive Fourier model, Gaussian prior, noise and candidate pools are used throughout.

The model chooses among uniform Bayesian information gain, information gain weighted by periodic angular diversity, and sector-balanced information gain. Each available strategy has a fixed control. The frozen chemistry rule selects diversity for multiple torsions, sector balance for a single amide torsion, and uniform otherwise. The rule and strategy menu were specified before this run, but these five development molecules had already been inspected in the earlier pilot.

## Final centered test residual RMSE

Mean ± sample SD across five seeds, kcal/mol. Lower is better.

| Molecule | Pure Bayes | LLM plan + Bayes | Chemistry rule |
|---|---:|---:|---:|
| butane | 0.0129 ± 0.0017 | 0.0129 ± 0.0017 | 0.0129 ± 0.0017 |
| biphenyl | 0.7255 ± 0.0605 | 0.7255 ± 0.0605 | 0.7255 ± 0.0605 |
| anisole | 0.0621 ± 0.0022 | 0.0621 ± 0.0022 | 0.0621 ± 0.0022 |
| nma | 1.0253 ± 0.2014 | 1.0253 ± 0.2014 | 0.9931 ± 0.1099 |
| phenylethanol | 2.5960 ± 1.0144 | 2.5960 ± 1.0144 | 2.6340 ± 0.9505 |

The LLM ties the rule on butane, biphenyl and anisole. It has higher mean final RMSE on N-methylacetamide and lower mean final RMSE on phenylethanol; the learning-curve AUC comparison reverses those two directions. Thus there is no consistent advantage over the rule. Fixed diversity improves mean final RMSE on biphenyl and N-methylacetamide, but those are retrospective observations, not a validated selection rule. Full results for all fixed strategies appear in `aws-plan-runs/summary.json` and `metrics.csv`.

## Latency and behavior

Exactly five model generation calls took 146.34 seconds total (29.27 seconds per molecule on average), with 1364 prompt and 281 output tokens. The previous per-batch pilot used ten calls and 551.10 seconds. That is a 73.4% observed reduction in total inference time, but the model, prompt and workload also changed; this is not a controlled model-speed comparison. At this budget, one plan per molecule halves calls relative to two calls per trajectory. Reuse across five seeds avoids 50 per-batch calls for 25 trajectories.

All five choices differ from the first presented option, so the earlier first-candidate behavior was not reproduced. However, rationales are generic, repeat wording from the prompt, and do not establish chemical reasoning. The prompt explicitly permitted uniform selection, which may bias this result. Several responses exceeded the requested 25-word rationale; no retry was made. Presentation order, complete requests and raw responses are retained for audit.

## Limits and next experiment

This is MMFF94-minus-UFF residual replay at the same molecular geometries, not quantum chemistry, force-field training, or validated MD improvement. Each molecule has only 32 pool and 32 test structures from a single starting conformer; conformer independence is not established. Selection under this fixed linear Gaussian model depends on geometry and covariance, not observed energy values. The LLM supplies a cold-start acquisition prior only. Five molecules and five initialization seeds do not support generalization claims.

Freeze a neutral prompt and the chemistry rule on a development set, then evaluate once on scaffold-separated held-out molecules with QM torsion labels. Compare pure Bayes, the chemistry rule, every fixed strategy, and the LLM selector under the same query and compute budgets. Predefine normalized learning-curve AUC as primary and final RMSE as secondary; report paired effects across molecules and latency. A shuffled-molecule-information control and prompt-order controls can test whether the planner uses chemistry. If the LLM does not beat the rule on held-out molecules, keep Bayesian selection and reject the LLM-benefit hypothesis for this design. Do not tune and retest on these five molecules to claim success.

## Reproduction and checks

Acquisition source at launch: Git commit `2d52271`. The manifest records the exact script and dataset hashes; model digest, prompts, decoding parameters, raw responses, candidate IDs and all curves are archived. Original input datasets are in the earlier `results/aws-pilot.zip` archive in the repository. Twelve acquisition tests pass, including one-call/label-free request construction, exact Bayes equivalence, matched budgets and test-label leakage checks. The result audit verifies all 125 budgets/initial sets, input hashes and all 25 exact LLM/Bayes trajectory matches.
