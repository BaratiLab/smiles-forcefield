# Sage repair-program adapter

Base: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [MLX4-bit conversion](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit). Upstream model cards identify Apache2.0 licensing. Base weights are not bundled.

This is a **new actual-Sage adapter**, trained on summed parent/local torsion features and QM-minus-Sage residual observations. It does not reuse or relabel the old UFF adapter.95 actual training-parent episodes,12 separate validation scaffolds,95 updates, one seed. Q/v LoRA rank8/scale8, last2 layers,180224 parameters. BF16 base floats and FP32 adapter/loss, LR5e-5, norm clip1.

Selection includes the unchanged model. Selected step=95; best nonzero step=95. `adapters.safetensors` contains the selected candidate, while `best-trained-adapters.safetensors` is the best nonzero adapter. Their exact SHA256 hashes are in `training-v2/COMPLETE.json`. A selected step0 means no fine-tuned upgrade was justified. No live deployment was performed.

Validation RMSE: Bayes 0.637374, untuned constrained LLM 0.731505, best trained0.637374. The same 56-answer grammar is applied to all LLM checkpoints. Refer to REPORT.md for stronger controls and descriptive intervals. No external held-out outcomes were used.

Secondary adaptive validation error-budget AUC: selected fine-tuned LLM 0.660176, Bayes 0.660176, untuned LLM 0.638349, matched two-proposal BMA 0.599786. The larger CV library reaches 0.565095. Lower is better. The planner abstains in all 12 cases and retains Bayes representation, so no planning advantage was demonstrated. The primary initial-fit target differs from full-trajectory utility.

The model proposes bounded hypotheses, never numerical force constants. Numerical fitting, observed-data acceptance, assignment checks and independent physical validation remain mandatory. It is not a universal force field, a direct SMILES-to-parameters model, or a validated MD accuracy upgrade. The12-group validation estimate, single seed, clipped gradients, restricted library and possible pretrained exposure limit conclusions.
