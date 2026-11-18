# QM acquisition-plan BF16 QLoRA adapter

This research adapter selects one of five Bayesian acquisition weightings for a Fourier QM-minus-UFF correction. It does not generate force-field parameters directly.

Base: mlx-community/Qwen2.5-7B-Instruct-4bit, Apache 2.0. Base-file hashes are recorded; no base weights are redistributed. Quantized integer weights are unchanged, while floating base tensors are cast to BF16 and then frozen. Only 180,224 FP32 LoRA parameters in the final two query/value attention blocks are optimized.

The primary trial uses the full frozen 96-update schedule with validation-only checkpoint selection. The BF16 repair was recorded before any fresh QM acquisition outcomes were scored. The interrupted FP16 attempt and its partial unlabeled predictions are preserved separately in diagnostic-fp16. The actual selected checkpoint and held-out performance are recorded in REPORT.md after completion.

Training pool: 95 parents/37 scaffold groups. Validation: 30/25. Fresh test: 168/143. All prior test groups are quarantined, and fresh validation/test exclude all previous groups. There is one fine-tuning seed. This is an acquisition replay on QM-optimized geometries, not MD accuracy or prospective QM-cost validation.

Selected adapter: `training-bf16-v2/adapters.safetensors`, update 96. SHA256: `9d5d30fff4d7fdea0ac25d3ae65ad03f1b24b3ddde3facdfabf7f2e65343ad46`. The full primary run completed 96 updates / 480 gradient microsteps; validation-only selection was used.

Use `finetune_mlx_stable.setup` to cast floating base tensors to BF16 before attaching the FP32 LoRA modules, then load the selected adapter. `finetune_infer_stable.py` verifies exact shapes and loaded values.

Fresh primary criterion passed: **False**. See REPORT.md for all controls, paired confidence intervals and limits; no MD force-field improvement is claimed.
