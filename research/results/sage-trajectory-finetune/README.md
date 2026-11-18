# Whole-trajectory Sage planner development experiment

This experiment changes the teacher objective from initial-fit error to full adaptive acquisition-curve error. The external 300-molecule cohort remains sealed; no new QCArchive outcome records are requested. The 95 training and 12 validation parents are existing retired development data.

For each of 285 training episodes, enumerate all 56 canonical plans (abstain, ten individual representations, 45 pairs). Start with eight observations, use the existing Bayesian numerical selector and adaptive two-point acquisition through budgets 10, 12, 14 and 16. Evaluate on the eight other conformers of that training molecule, outside the sixteen-point acquisition pool. Minimize centered RMSE area divided by the eight-point budget span. Teacher targets use the best plan unless its relative improvement over Bayes is at most 1%, in which case abstain. Future pool labels are revealed only after acquisition; evaluation labels never enter the controller.

A fresh Qwen2.5-7B-Instruct 4-bit adapter receives one shuffled pass of 95 distinct training parents, exactly one episode per parent. Its completion loss is weighted by training-trajectory benefit. This is supervised utility imitation, not reinforcement learning or differentiation through quantum calculations. Architecture, numerical priors, compiler, decoder, and two-proposal cap remain as in the prior initial-fit study. The prompt and schedule seed also change, so this is not an isolated objective ablation.

Checkpoint selection includes zero updates and chooses the lowest mean validation full-trajectory AUC at steps 0, 48 and 95, breaking ties toward the earlier checkpoint. No validation tuning of learning rate, number of epochs or teacher thresholds is planned. All 12 validation scaffolds have been inspected previously: results and paired bootstrap intervals are exploratory, not a fresh test.

Controls retrain on the exact same 95 parent/episode pairs: ExtraTrees regression of all 56 plan costs, a teacher-answer classifier, fingerprint retrieval and a training-global constant plan. Fixed two-proposal and two-proposal Bayesian model averaging are also matched in proposal count. Full-285 learners and ten-representation CV/BMA are stronger, explicitly larger-information or larger-proposal references. All policies use the same initial observed labels and acquisition budget. Numerical constants are fitted by the Bayesian model, never emitted by the LLM.

The decoder remains deterministic token-greedy constrained generation over the same canonical language. It is not global MAP over complete plan probabilities; decoder improvements require a separate experiment. No external held-out evaluation or website promotion is authorized by these development results.

## Reproduction

Use the repository root. Chemistry Python needs NumPy, SciPy, scikit-learn and RDKit plus the repository installed. Training uses a separate MLX-LM environment and an existing local Qwen model.

```sh
python research/torsion_repair/trajectory_utility.py --source ../sage-program-finetune/aligned-v1 --out ../sage-trajectory-finetune/aligned-v1
python research/torsion_repair/trajectory_controls.py --aligned ../sage-trajectory-finetune/aligned-v1 --out ../sage-trajectory-finetune/controls
python research/torsion_repair/train_trajectory_mlx.py --aligned ../sage-trajectory-finetune/aligned-v1 --model /path/to/qwen2.5-7b-4bit --chem-python /path/to/chemistry/python --out ../sage-trajectory-finetune/training-v1
```

The accompanying output archive supplies the aligned development data, teacher utilities, checkpoints, optimizer state and per-molecule acquisition traces. The base model is not redistributed.
