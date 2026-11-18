# LLM-guided torsion acquisition

Research experiment, separate from the production website. It does not alter downloadable Sage parameters.

## First real-model experiment

Five molecules: butane, biphenyl, anisole, N-methylacetamide, 2-phenylethanol. All methods fit the same Gaussian linear model with an intercept and harmonics 1–3 (cosine and sine) for each mapped torsion. Random samples without replacement. Bayesian experimental design greedily maximizes conditional information gain with within-batch covariance updates. The LLM sees chemical identity, queried energy residuals, predictions and candidate variances, and returns exact candidate IDs. It sees no held-out energies or unqueried labels.

The inexpensive engineering oracle is MMFF94 minus UFF at identical real coordinates, in kcal/mol. Pool and test target angles use independent RNGs; constrained MMFF minimization relaxes nuisance coordinates, then **unconstrained single-point** MMFF and UFF energies are evaluated. Phenylethanol also samples OH rotation. SDFs and geometry hashes are retained. This is not QM, not Sage improvement, and not evidence of physical accuracy. Legacy `qm_energy_kcal_mol` fields contain MMFF energies and are explicitly identified in dataset metadata and LLM prompts. Both splits derive from one embedded seed; independently embedded conformer families and invariant geometric deduplication are still needed for scientific evaluation. Rounded Cartesian hashes only catch exact coordinate duplicates.

```sh
python build_proxy.py --out data --size 32
python suite.py --data data --out runs/proxy --seeds 0 1 2 --budget 12
OLLAMA_URL=http://localhost:11434 OLLAMA_MODEL=qwen2.5:0.5b \
 python suite.py --data data --out runs/llm-proxy --seeds 0 1 2 --budget 12 --llm
python -m unittest -v test_poc.py test_acquisition.py
```

Defaults: 4 shared seed observations, 12 total observations including initial points, batches of 4; 32 pool and 32 test geometries per molecule. Every policy has identical initial observations per seed. A full-pool fit is a capacity reference, not a lower bound. Metrics are centered residual RMSE and query-normalized error AUC; seed SD measures acquisition variation only. Five development molecules cannot establish generalization or statistical superiority.

## Private AWS CPU deployment

Ollama serves Qwen2.5 0.5B for the compact CPU pilot; Qwen2.5 1.5B Q4_K_M (~986 MB weights) is also supported. Both use Apache 2.0. The first 1.5B full-precision-JSON prompt was too slow for a quick pilot and was stopped; do not mix that partial run with the compact 0.5B experiment. No vendor API keys or GPU are required. The model runs on the existing VM inside `deploy_default`; **no port is published**, and no website route reaches it. Use a 2500-MB memory limit, one CPU, one loaded model, one concurrent request. The adapter unloads weights after each call. Run benchmarks with `--cpus 0.5 --memory 1g` to leave capacity for the public API. This is a small-model feasibility benchmark, not a recommendation to run many concurrent LLM users on a 2-vCPU VM.

The compact-v2 adapter rounds input angles/energies to 4 decimals and variances to 6 decimals, and uses row arrays to reduce prompt cost. Full precision stays in the fitter and original saved state. The adapter pins temperature/seed/context, uses a JSON schema with an available-ID enum, validates distinct IDs, allows one repair attempt, and fails without silently switching policies. Request/response files record model digest, token counts, timing and decisions. Record the Ollama image digest in the experiment environment. Treat model tag alone as mutable. Invalid actions count as failed runs; never report the mock adapter as an LLM.

For real QM replay, substitute datasets with `kind: qm_replay`, protocol/basis/program, matching baseline energies, converged geometries and atom mapping. `poc.py` accepts the same energy fields. First establish full-pool model capacity and freeze hyperparameters on development data; then evaluate acquisition under a fixed QM budget. Fitted sine terms and independent molecular coefficients are research fits, not exportable physical force-field corrections; symmetry constraints and prospective QM validation remain required.

Sources: https://ollama.com/library/qwen2.5:1.5b ; https://docs.ollama.com/api/generate ; https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html

Results are kept outside Git by default (`data/`, `runs/`, `aws-data/`, `aws-runs*`). `export_results.py --runs aws-runs-compact` bundles only the fixed benchmark data, acquisition logs and reports after all 15 molecule/policy pairs finish. No production website submissions are used. The experiment records selected queries, posterior coefficients, baseline/full-pool reference errors, per-step test metrics, adapter time, original states, model outputs and model digests.

## Observed first pilot

[Results and limitations](results/REPORT.md): the 0.5B model returned valid IDs but selected the first available candidates in every round. The ordered-selection control reproduces its learning curves. Add `--ordered-control` to `suite.py` for this fourth ablation; it is never labeled an LLM. Run `python audit.py --runs PATH --out audit.json` to verify matched budgets and flag first-available choices. This is a negative controller-quality finding, not evidence that LLM reasoning improves acquisition.

## One-plan experiment

`plan_benchmark.py` calls the LLM exactly once per molecule, before any labels are observed. The valid choices are uniform conditional information gain, information gain multiplied by a periodic angular-diversity weight, and information gain divided by angular-sector occupancy. Bayesian covariance and coverage are updated within each batch. All methods retain the same additive Fourier fitter and parameter priors. No new QM geometries or functional forms are introduced by the plan.

The fixed rule is frozen: multiple fitted torsions → angular diversity; otherwise an amide SMARTS match → sector balancing; otherwise uniform. All three fixed strategies are represented as controls (uniform is pure Bayesian selection). The model sees molecular SMILES and mapped torsions, not candidate IDs, energies, test labels or prior experiment results. Strategy presentation is deterministically shuffled and logged. A single plan is reused across all five acquisition seeds. The candidate pool and label budget are unchanged.

```sh
OLLAMA_URL=http://localhost:11434 python plan_benchmark.py \
 --data aws-data --out aws-plan-runs --model qwen2.5:1.5b --seeds 0 1 2 3 4
```

Plans, full requests/responses, model digest, token counts and latency are saved. Invalid output fails without fallback or a second model call. The benchmark includes 5 molecules × 5 policies × 5 seeds = 125 trajectories, but only five LLM calls. Fewer calls are an architectural property; runtime must be measured because the planner model differs from the earlier small-model pilot. This remains a small, previously inspected MMFF/UFF development benchmark; improvements are exploratory, not held-out scientific validation.

### Completed one-plan comparison

The AWS pilot ran 125 matched trajectories with five Qwen2.5:1.5B planning calls. All plans chose uniform Bayesian acquisition, reproducing pure Bayes exactly. No LLM advantage was demonstrated. See [full report and limitations](results/one-plan/REPORT.md) and [raw requests, responses and curves](results/one-plan/aws-plan-results.zip).


### Scaffold-held-out QM benchmark (v1)

The new benchmark uses published QM torsion profiles from the official OpenFF
Sage 2.2.1 fitting archive, pinned to a source commit and SHA256. It learns
QM-minus-UFF torsion corrections; it does not claim held-out Sage performance.
`qm_protocol.json` freezes 60 development and 300 held-out parent molecules,
five matched seeds, 4–12 labels, and the primary paired comparison against
Bayes and the chemistry rule. Ring Murcko scaffolds and generic acyclic graph
scaffolds are disjoint between development and test. The five original pilot
identities and pilot ring scaffolds are excluded.

Run the chemistry steps with NumPy >=2 and RDKit. Run the MLX inference step
in a separate environment with `mlx-lm` on Apple Silicon; no model server or
public inference endpoint is required. The inference script loads local
safetensors with remote tokenizer code disabled. Download the model named
`mlx-community/Qwen2.5-7B-Instruct-4bit` using Hugging Face's client; preserve
its file hashes and revision metadata. Only use public model weights, never
publish access tokens.

```bash
curl -fL https://raw.githubusercontent.com/openforcefield/sage-2.2.1/b70c449e3a98e40f0b6e4c6b2f27932fce47be42/04_fit-forcefield/nor4/fb-fit/targets.tar.gz -o targets.tar.gz
python qm_data.py --archive targets.tar.gz --out qm-data
python qm_benchmark.py develop --data qm-data --out qm-development
python qm_planner_mlx.py --data qm-data --frozen qm-development/frozen_planner.json --model /path/to/local/model --out qm-plans
python qm_benchmark.py run --data qm-data --frozen qm-development/frozen_planner.json --plans qm-plans/plans.json --out qm-runs
python qm_analyze.py --data qm-data --frozen qm-development/frozen_planner.json --plans-dir qm-plans --runs qm-runs --out qm-report
```

Dataset creation and evaluation require new output directories. Inference may
resume only when model, source, split, frozen examples and prompt hashes match.
Each successful molecule decision is saved immediately. No malformed-response
fallback or retry is used: decoding is constrained to five one-token choices.
Planner examples are chosen exclusively from development labels. The planner
never sees test energies, candidate energies, source record IDs or prior-run
metrics. Energy centering uses observed training labels only. Every available
LLM weighting has a fixed non-LLM control; random, the development-global best,
and a development-frequency chemistry-blind draw are also evaluated.

The main outcome is a molecule-mean baseline-normalized error/query AUC.
Inference averages seeds first and bootstraps scaffold clusters. Claim benefit
only if the LLM beats both primary controls under the prespecified intervals.
Do not change the planner after examining these held-out results and call a
rerun independent. Absence of these public molecules from LLM pretraining is
not guaranteed. See the completed results report for experimental outcomes.
