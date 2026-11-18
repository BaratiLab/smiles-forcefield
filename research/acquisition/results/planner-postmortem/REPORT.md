# Why the acquisition planner lost and what to test next

Keep Bayesian numerical fitting and experimental design. Give the LLM a chemically constrained hypothesis-generation role: propose which force-field assumptions need revision, then let tools and QM evidence test those proposals. The current results do not justify replacing Bayes with a larger or more heavily fine-tuned angle-ranking model.

The original confirmatory result remains unchanged: fine-tuned normalized error-query AUC 0.69198 versus Bayes 0.66826, a 3.55% disadvantage with a paired 95% scaffold-cluster difference interval of [+0.00821, +0.04401]. The following analyses use that now-retired test cohort and are exploratory. They cannot be presented as a new held-out improvement.

## What the additional measurements show

**1. The LLM is unstable to an irrelevant prompt change.** A deterministic, outcome-independent hash selected 20 retired molecules for a five-order stress test. All 20 changed their chosen semantic action across cyclic option permutations; seven used two different actions, twelve used three, and one used four. Repeating the original ordering reproduced all 20 original decisions and every option logit exactly. Thus the choice changes are caused by ordering in this diagnostic, rather than inference noise. The fine-tuned model never selected letter A in the 100 calls. Position counts alone would not establish causation; the semantic-order intervention does. This does not establish the exact fraction of the full-cohort performance deficit caused by position bias.

**2. A few costly choices dominate the loss.** Six cis-focus selections—3.6% of the 168 molecules—contribute 77.8% of the net AUC disadvantage to Bayes. The five largest individual losses contribute 71.6%. The model is not failing by a small, uniform amount; it sometimes selects a harmful prior without a reliable keep-Bayes/abstention gate. Removing those actions after observing their test losses would be test-set tuning, not a confirmed fix.

**3. The action mix and the molecule-to-action matching are both unhelpful.** The mean AUC of a shuffled assignment preserving the LLM’s action frequencies is 0.68257, compared with the actual 0.69198 and Bayes 0.66826. The deficit decomposes exactly into +0.01431 from using this action mixture instead of Bayes, and another +0.00941 from the actual assignment of those actions to molecules. The assignment is worse than 90.6% of 10,000 shuffles, but remains inside their 95% range: this is descriptive evidence, not a significant anti-reasoning or causal chemistry result. Always using angular diversity scores 0.67142, better than this LLM selector.

**4. The current action space has little room to win.** Perfect hindsight selection of the best of the five available plans reaches 0.63786: only 4.55% below Bayes. Even this oracle knows the evaluation outcomes, so not all of that gap is necessarily predictable from the permitted inputs. The trained model was trying to extract a small, potentially noisy conditional gain using only 95 training parents in 37 scaffold groups.

**5. Model revision has more headroom than reweighting the same acquisitions.** We kept exactly the existing Bayes-acquired IDs and changed only the residual Fourier representation. A pool-label-only leave-one-out selector over maximum harmonic orders {1,2,3,4,6}, with the original prior/noise, achieved mean AUC 0.63618. That is a 4.80% exploratory reduction without any LLM or extra QM labels. Hindsight selection of one harmonic family per molecule reached 0.51837, a 22.43% gap. These are post hoc development diagnostics, not fresh performance claims. The oracle uses evaluation labels; the cross-validation selector does not. They establish a candidate baseline and a direction to investigate, not evidence that an LLM is needed.

The full-pool H=3 test error is about 0.50319 normalized RMSE. It is a finite-data reference, not an irreducible error floor: it includes approximation, regularization, conformer-branch and sampling effects. Higher harmonic fits can also fit nuisance variation. Off-grid conformers and forces must distinguish useful representation changes from scan-specific overfitting.

## Why these failures are plausible

Our Bayesian model has posterior covariance `(alpha I + X^T X / sigma^2)^-1`. With fixed features and noise, this covariance and its information-gain query selection depend on geometry, not observed energy values. Bayes already has a well-defined numerical problem. The LLM cannot change its seven-coefficient Fourier representation, parameter sharing, priors or target experiments; it can only multiply acquisition scores by a few weights.

The cold planner receives no acquired QM residuals, posterior-predictive failure patterns or uncertainty-calibration evidence. It therefore cannot diagnose the physical reason a force field is wrong. A reasonable chemical belief such as “an amide barrier deserves attention” does not imply that emphasizing that angular sector best reduces QM-minus-UFF fit error.

The optimization objective is another possible contributor. Cross entropy to softmax-transformed plan costs is a surrogate for downstream argmax regret. A small distributional loss does not guarantee that the selected action avoids a rare large error. Our checkpoint validation was nonmonotonic, and the order intervention shows that balanced training alone did not create a semantically invariant selector. Small data, one fine-tuning seed, restricted adapters and numerical sensitivity are credible limitations; we did not perform causal ablations that identify their individual contributions.

This evidence supports changing the task. It does not show that LLMs cannot help force-field development, or that increasing model size or training duration would fix this design.

## The next experiment

**Working title: Chemically constrained hypothesis generation for Bayesian force-field repair.**

**Hypothesis:** after seeing a small set of QM observations and verified diagnostics, an LLM can propose useful changes to torsion parameter sharing or representation more efficiently than rules or a learned non-LLM proposer. Numerical parameter estimation and query selection remain Bayesian. The contribution must be in the quality of the proposed model revisions, not in having extra labels, tools or fitting compute.

The first task is ligand-specific torsion repair using a pinned classical starting force field. Background bond, angle, charge and nonbonded terms stay fixed. Each fresh ligand task starts from the same baseline; learning is not carried between test tasks. The primary evaluation uses untouched conformers of that ligand. Transfer of the learned pattern to untouched analogues is a separate, family-clustered endpoint.

### The decision loop

1. **Collect eight common support configurations.** Generate cheap MM candidates rather than assuming future candidates have already been QM-optimized. Acquire consistent QM energies and, when available, forces. All methods receive the same initial support and candidate pool.
2. **Compute evidence with tools.** Supply verified parameter assignments, atom-mapped graph neighborhoods, residuals by torsion group and conformer branch, posterior covariance, identifiability, predictive checks and training-only retrieved QM analogues. Parameter uncertainty and model discrepancy are distinguished as hypotheses to investigate.
3. **Ask the LLM for at most two repair programs.** The constrained schema permits keep/no-edit, a local SMIRKS specialization, or a bounded harmonic/prior revision for one scoped torsion family. A proposed diagnostic configuration must come from the shared cheap candidate pool. The LLM does not produce numerical force constants or energies. An illustrative question is whether an existing shared aryl-ether parameter needs a chemically defined local specialization; this is not a claim that we found such a defect in a released force field.
4. **Compile and validate the programs.** Check matching atoms, scope, parameter coverage, symmetry/permutation consistency, parameter count and MD-engine support. Record the impact on other assignments. Exact symmetry constraints require whole-system/conformer justification; a local methyl motif alone does not prove that an independently relaxed total-energy profile has exact threefold symmetry. Invalid proposals are rejected with their cost counted.
5. **Let numerical evidence choose.** Fit coefficients with the same priors and optimizer for every proposal generator. Use support-only nested validation or model evidence with complexity control, always retaining the no-edit hypothesis. The shared acquisition layer can query configurations that discriminate candidate models as well as reduce parameter uncertainty.
6. **Allow one further LLM call after sixteen observations.** Continue in batches of four to a total budget of 32 configurations. There are at most two LLM calls per task, never one per candidate. Cache graph, retrieval and diagnostic state. The generator may abstain when no repair is supported or when the portable representation is inadequate.

Initially permit only periodic torsion terms that our common force-field representation can export. Detecting a need for coupled or nonbonded physics is useful, but it must produce an explicit inadequacy flag rather than an unsupported downloadable force field. Changing protonation, molecular charge or spin is not a free repair action: that changes the physical target.

### Architecture and supervision

Use an open 7B model with LoRA to generate constrained JSON repair programs from the graph/tool/residual state. A shared utility critic scores each candidate independently of its displayed position. Canonical program identities determine tie-breaking. The same critic, compiler and Bayesian layer are used for rule, retrieval and learned graph proposal generators, isolating the LLM’s contribution.

Generate development-only repair episodes with nested QM holdouts. Label proposed edits by downstream out-of-sample cost-error reduction, including no-edit and rejected-edit examples. Use program supervision plus regret-weighted utility/preference learning, with checkpoint selection on task performance. Do not teach arbitrary A–E labels or rely on generated chemical explanations as correctness evidence. Split parents, scaffolds and target chemical environments before generating episodes; 5,000 episodes are not 5,000 independent molecules.

Target roughly 1,000 training parents, 200 validation parents and 300 fresh test parents in at least 150 scaffold groups, with three fine-tuning seeds and five acquisition seeds. These are planning targets, subject to inventory and a development-only power calculation. They are not obtained cohorts or a claim of sufficient power. All current cohorts are retired and may now be used explicitly for development. None of the inspected scaffold universe should be relabeled a fresh scaffold test.

Candidate sources are new, overlap-audited QCArchive/OpenFF records and a separate SPICE benchmark. SPICE supplies energies and forces at omegaB97M-D3BJ/def2-TZVPPD; the OpenFF scans used here are B3LYP-D3BJ/DZVP. Fit and evaluate protocol-specific models rather than pooling the levels of theory as interchangeable targets. Audit molecular/geometry duplication and the starting force field’s fitting records; absence from LLM pretraining cannot be guaranteed. [SPICE data description](https://github.com/openmm/spice-dataset)

### Baselines that make the result credible

| Proposal or model policy | Shared numerical machinery |
|---|---|
| Fixed force-field model | Bayesian fitting and acquisition |
| Verified chemistry rules | Same compiler, evidence scoring and acquisition |
| Support-only cross-validation | Same candidate representation library |
| Bayesian model averaging | Equally budgeted rule-generated hypotheses |
| Retrieval-only proposer | Same analogues, validator and fit budget |
| Learned non-LLM graph/descriptor proposer | Same QM supervision and proposal slots |
| LLM proposer | Same tools, critic, compiler and numerical budget |
| BespokeFit-style local refitting | Same counted QM budget; disclose workflow differences |

BespokeFit already selects torsions, constructs bespoke SMIRKS and fits them to QM torsion profiles with ForceBalance. Merely wrapping that workflow in an LLM is not the contribution. The test is whether the LLM proposes better targeted model revisions or diagnostic experiments at equal cost. Its default fragmentation and scan workflow must be accounted for explicitly in practical comparisons. [BespokeFit workflow](https://github.com/openforcefield/openff-bespokefit/blob/main/docs/users/bespoke-workflows.md)

Required ablations remove residual evidence from the LLM, replace it with the verified rule proposer, and compare model revision on fixed acquisition IDs with model revision plus updated Bayesian acquisition. This separates representation gains from query-selection gains. All methods share exact symmetry tools; their benefit cannot be credited to the LLM alone.

Espaloma, Grappa and Garnet already establish graph-based force-field parameter learning. The proposed role is complementary: validated structure/typing repair and experiment design around a numerical force-field learner. Whether that is a novel contribution requires a focused literature review and a positive comparison against the strongest non-LLM revision baseline. [Espaloma](https://pubmed.ncbi.nlm.nih.gov/36349096/), [Grappa](https://pmc.ncbi.nlm.nih.gov/articles/PMC11734696/), [Garnet](https://arxiv.org/abs/2603.16770)

### Primary endpoint and stopping rules

Use mean held-out centered relative-energy RMSE area under the error-versus-QM-evaluations curve at budgets 8, 16, 24 and 32. One QM energy/gradient calculation counts as one configuration; force components are not counted as separate acquired configurations. Prospective experiments also log every optimization evaluation and core-hour, including failed jobs, so a torsion scan is not falsely treated as one cheap query.

Compare the hybrid against pure Bayes and the strongest non-LLM revision method selected on validation before opening the new test. A proposed success gate is at least 10% lower error AUC against both, with both paired 97.5% scaffold-cluster intervals below zero. This is an effect-size target, not a power guarantee. Estimate sample size using development data first. Hierarchical uncertainty must respect molecules/scaffolds and training seeds rather than treating trajectories as independent samples.

Keep force error, barrier heights, conformer ranking, complexity, rejection/abstention calibration and total compute as separate secondary endpoints. Validate distinct conformer families and off-grid states to reject high-harmonic scan overfit. Check native exported energies/forces in available GROMACS, LAMMPS and NAMD engines using tolerances calibrated on numerical-precision controls before test. A short MD trajectory is a stability check, not proof of thermodynamic accuracy.

If rules, cross-validation or model averaging capture all gains, use those methods and do not claim an LLM advantage. Do not rescue a failed test by modifying prompts and reporting the same test again as fresh.

## First proof of concept and execution sequence

**Step 1 — development headroom and defect recovery.** Use retired data to reproduce the basis diagnostics and build controlled parameter-sharing defects in a portable torsion model. Known injected defects test recovery mechanics; they are labeled synthetic interventions, not newly discovered defects in a released force field. Include clean/no-defect tasks. Check whether a deterministic CV/BMA baseline already solves them.

**Step 2 — compiler and invariance gate.** Implement the bounded repair schema, atom-map/symmetry/coverage tests and independent candidate scoring. Require valid programs and identical semantic outcomes under candidate permutation and equivalent atom relabeling before fine-tuning. Numerical stability and finite-difference checks on small development models precede long runs.

**Step 3 — development-only teacher rollouts.** Fit all permitted candidate repairs on development support data, score on internal development holdouts and train the proposer/critic. Include observed-data-only acceptance and abstention. Freeze proposal slots, model priors, checkpoint selection and compute accounting.

**Step 4 — independent retrospective confirmation.** Inventory a new QM source, remove fitting-record and inspected-scaffold overlap, finalize power and preregister the test. Compare all generators with identical numerical machinery. Repeatedly generated episodes never cross a parent split.

**Step 5 — prospective and export confirmation.** Only after retrospective success, prepare a priced/core-hour-bounded campaign of about 20 blinded ligands. Generate cheap candidate geometries, run QM on acquired states only, validate off-grid forces/conformers and native exports, then check short MD stability. New QM/cloud jobs need their concrete compute specification and authorized budget before launch.

The immediate engineering milestone is therefore a tool-checked model-revision benchmark and compiler—not another expensive fine-tuning run on angle-weight labels. The machine-readable proposal is `next_experiment_proposal.json`; it is a draft design, not an executed or preregistered study.
