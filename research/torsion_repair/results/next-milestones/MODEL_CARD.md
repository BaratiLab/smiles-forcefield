# Repair-program adapter model card

Status: research diagnostic; validation selected the unchanged model.

Base: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), via the [MLX-community 4-bit conversion](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit). The upstream cards identify Apache2.0 licensing. Base weights are not bundled; preserve their licensing and attribution when using them.

Training:64 completion-token SFT updates, one seed,180224 q/v LoRA parameters in the last two layers. Rank8, scale8, dropout0; BF16 base floats, FP32 adapter/loss. AdamW learning rate1e-4, weight decay0.01, norm clip1. Available pool95 parents/285 aligned episodes; actual updates use64 distinct parent/seed episodes. Validation12 separate scaffold groups, q8support+8internalpoolholdout.

Selection: all checkpoints0/32/64 score0.9586271519. Earliest tie selects0. `training-v2/adapters.safetensors` is therefore the **zero-update deployment candidate**, not an improved fine-tuned model. `best-trained-adapters.safetensors` contains checkpoint32, retained for the diagnostic comparison; SHA256 `5f5ed9816de0b9a1f01732211ea630be3a748daffcdaafe6cc3388a4ccadc417`. All saved adapter/optimizer tensors were finite. No live website deployment occurred.

Evaluation:24 retired molecules, same prompts as cached base with an exact zero-adapter repeat check. Mean error-query AUC1.6188741689, exactly Bayes. All retain no-edit;26/48 raw programs fail strict validation. Mean local generation13.85seconds. No fresh QM outcomes were scored. The earlier angle-classification adapter was not reused.

Scope: trained only on QM-minus-UFF driven-instance residual repair programs. The later geometry-complete Sage corpus is separate and has not been used to train this adapter. It predicts hypotheses, never numerical parameters. Its outputs require compilation and observed-data validation and must not be installed directly into MD packages.

Limitations: small scaffold coverage, incomplete single-epoch exposure, abstention-heavy labels, large clipped gradients, unconstrained JSON generation, one training/acquisition seed, uncertain pretrained exposure and no off-grid QM force or MD accuracy evidence. This adapter is not recommended as a public accuracy upgrade.
