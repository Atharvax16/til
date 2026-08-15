# Claude Code task: controlled comparison of self-supervised objectives on medical images

## Goal (read this first, it constrains everything below)

Build a small, reproducible research codebase that answers one question:

> **Holding the encoder backbone and compute budget fixed, how do different self-supervised pretraining objectives compare as medical-image representation learners, and at what "abstraction level" does the denoising objective peak?**

Arms to compare (all sharing ONE encoder):
1. **MAE / masked reconstruction** (SimMIM-style, since the shared encoder is convolutional — see below).
2. **Diffusion / denoising** (a small DDPM whose denoiser reuses the shared encoder as its trunk). This is the OrthoDiffusion mechanism.
3. *(optional, phase 2)* **JEPA / latent prediction** (I-JEPA-lite). Leave a clean extension point; do not build it until arms 1–2 are done.

Downstream evaluation for every arm is identical: freeze the encoder → linear probe, then fine-tune. Plus a diffusion **timestep ablation** that should reproduce the rise-then-fall AUROC-vs-t curve.

Target hardware: **Apple Silicon (M-series), PyTorch MPS.** Everything must run on a laptop in 2D at ≤64×64. No 3D, no multi-GPU assumptions.

---

## Hard invariants (the scientific point — do NOT violate any of these)

These are the whole reason the project exists. If you find yourself weakening one for convenience, stop and flag it instead.

- **I1 — One shared encoder.** Define a single `Encoder` class in one place. Every arm instantiates the *same class with the same config*. Assert at runtime that the encoder parameter count is identical across arms. Do NOT give MAE a ViT and diffusion a U-Net — that confound is exactly what we're eliminating.
- **I2 — One feature-extraction site.** All arms extract the downstream feature from the *same named module* (the encoder bottleneck). Provide a forward hook or a `return_features` path that is byte-for-byte identical across arms. The only thing that varies at extraction time is the diffusion timestep `t` (MAE/JEPA use the clean input, i.e. the `t=0` analog).
- **I3 — Matched pretraining data.** Pretraining uses ONLY the official train split, treated as unlabeled (labels must not be read during any pretraining loop — enforce this in code, not just by convention). Same images, same resolution, same base augmentation pipeline for every arm. Each objective adds its own corruption (masking / noise) on top of that shared pipeline.
- **I4 — Matched compute budget.** Equal number of optimizer steps (or epochs at equal effective batch size) across arms. Same optimizer family and LR schedule unless an arm genuinely cannot train with it — if you must deviate, log it loudly. Record wall-clock and step count per arm.
- **I5 — Matched downstream protocol.** The linear-probe head and its training recipe (epochs, LR, optimizer, weight decay) are identical across arms. Same for the fine-tune recipe. Never hand-tune probe hyperparameters per arm to flatter one method.
- **I6 — Honest test discipline.** Fixed seed list (default 3 seeds); report mean ± std. Metric code lives in one module used everywhere. The test split is evaluated once per final config, never used for model selection — use the official val split for that.

---

## Dataset

- **PneumoniaMNIST** via the `medmnist` package. Binary classification (pneumonia vs normal). Use the official train/val/test splits — they are predefined; do not re-split.
- Resolution: use the **64×64** variant (`size=64`) if available in the installed `medmnist`; else load 28×28 and upsample to 64. Make resolution a config value.
- Note and handle class imbalance (report class counts on load). Because of imbalance, **AUROC and AUPRC are the primary metrics**, not accuracy.
- Grayscale, single channel. Keep the input pipeline identical across arms; normalize to `[-1, 1]` for the diffusion arm and expose normalization as config so the same transform object is reused.

---

## Shared encoder (decide once, reuse everywhere)

Use a **small convolutional U-Net-style trunk** as the shared encoder, mirroring OrthoDiffusion's "extract from the bottleneck / mid block" design:

- Encoder path: 3–4 downsampling stages, GroupNorm, SiLU, modest channels (e.g. base 32 → 64 → 128 → 256). Keep it small enough to pretrain in a few hours on MPS.
- **Bottleneck** is the canonical feature site (I2). If you include more than one mid block, expose each as a selectable extraction point so we can do a small block sweep like the paper.
- The trunk must accept an optional **timestep embedding** (sinusoidal → MLP, injected via FiLM/adaLN or additive conditioning). For non-diffusion arms, pass `t=0` (or skip conditioning) so the *same* trunk serves everyone.
- Provide `encoder.extract(x, t=0, block="bottleneck") -> Tensor[B, C]` that runs a forward pass and returns a pooled feature (global average pool over spatial dims by default; also allow an attention-pool option, since the paper found self-attention pooling best — keep GAP as the default for simplicity).

Arms wrap this trunk:
- **Diffusion arm**: full DDPM. The denoiser = shared trunk (as U-Net encoder+bottleneck) + a lightweight decoder to predict noise ε. Standard DDPM schedule (cosine or linear), `T=1000`, ε-prediction, ℓ2 loss. After training, sample a handful of images to sanity-check the model actually learned the data manifold.
- **MAE / SimMIM arm**: mask a fraction of input patches (e.g. 50–75%), run the shared trunk on the masked image, attach a lightweight conv decoder, reconstruct masked regions, ℓ2 on masked pixels only. Visualize a few reconstructions.
- **JEPA arm (optional/phase 2)**: context + target encoders (both the shared trunk), predict target latent from context in representation space. Stop after leaving the interface; don't implement yet.

---

## Evaluation protocol

For each arm and each seed:
1. **Linear probe**: freeze encoder, extract bottleneck feature, train a single `nn.Linear` head. Report AUROC, AUPRC, accuracy, F1 on val (for selection) then test (once).
2. **Fine-tune**: unfreeze encoder + head, train end-to-end with a much smaller LR. Report the same metrics. (Expected: MAE benefits more from fine-tuning than from linear probing — surface this, don't hide it.)
3. Include two **supervised baselines** trained from scratch on the same encoder: (a) full labeled train set, (b) 10% labels — to reproduce the label-efficiency story.

Diffusion **timestep ablation** (`ablate_timestep.py`):
- Sweep `t ∈ {10, 30, 50, 100, 150, 200, 300, 500}` for the diffusion arm's linear probe.
- If multiple mid blocks exist, also sweep block.
- Plot **AUROC vs t** and save to `results/`. The expected qualitative shape is rise-then-fall (low t = fine detail, high t = prior-dominated / signal gone). Call it out in the README as the empirical echo of the prior-vs-likelihood tradeoff.

---

## Engineering / reproducibility requirements

- **Environment**: use `uv` (`uv init`, `uv add ...`). Dependencies: `torch`, `torchvision`, `medmnist`, `numpy`, `scikit-learn` (for AUROC/AUPRC), `matplotlib`, `pyyaml`, `tqdm`. No Weights & Biases required; CSV + matplotlib logging is fine.
- **MPS handling**: single `get_device()` util → prefer `mps`, else `cpu`. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` guidance in the README for unsupported ops. Use float32 (don't rely on MPS half precision). Keep batch sizes modest and configurable.
- **Determinism**: one `set_seed(seed)` util seeding python/numpy/torch; document that full MPS determinism isn't guaranteed and that's why we report mean ± std over seeds.
- **Config-driven**: a single `configs/default.yaml` holds resolution, channels, epochs/steps, batch size, LR, mask ratio, diffusion T/schedule, seed list, timestep sweep list. No magic numbers buried in code.
- **Logging**: every run writes a row to a results CSV (arm, protocol, seed, t, block, AUROC, AUPRC, acc, F1, steps, wall-clock). Plots read from that CSV.
- **Sanity checks as code**: assert encoder param-count parity across arms (I1); assert labels are never accessed during pretraining (I3); a fast `--smoke` mode that runs 1 epoch on a subset to prove the whole pipeline end-to-end before any real run.

---

## Suggested repo structure

```
ssl-medical-compare/
  pyproject.toml
  README.md
  configs/default.yaml
  src/
    data.py            # PneumoniaMNIST loaders + shared transforms
    backbone.py        # shared U-Net trunk + timestep embedding + extract()
    arms/
      diffusion.py     # DDPM schedule, denoiser (uses trunk), train loop, sampling
      mae.py           # SimMIM masking + conv decoder + train loop
      jepa.py          # (stub / phase 2)
    features.py        # bottleneck feature extraction, timestep-aware
    probe.py           # linear probe + fine-tune train/eval
    metrics.py         # AUROC, AUPRC, acc, F1 (single source of truth)
    utils.py           # seeds, device, logging, param-count asserts
  scripts/
    train_pretrain.py  # pretrain one arm from config
    run_eval.py        # probe + finetune + baselines, write CSV
    ablate_timestep.py # diffusion t-sweep + plot
  results/             # CSVs + plots
```

---

## Build order (implement in these milestones, pause after each for me to verify)

- **M0** Scaffold, env, `data.py`. Print shapes, class balance, a grid of sample images. Add `--smoke`.
- **M1** `backbone.py`: shared trunk + timestep embedding + `extract()`. Assert param parity.
- **M2** Diffusion arm: train a small DDPM, sample images to confirm it learned the manifold.
- **M3** MAE arm: train, visualize reconstructions.
- **M4** `features.py` + `probe.py`: linear-probe harness. Run both arms + supervised baselines → first results table.
- **M5** `ablate_timestep.py`: t-sweep + AUROC-vs-t plot.
- **M6** Fine-tune both arms; final arm × {probe, finetune} table with mean ± std over seeds.
- **M7** *(optional)* JEPA arm.

---

## Explicit non-goals / don'ts

- Do NOT reproduce full OrthoDiffusion (3D, three orientation models, 16k volumes) — out of scope and out of hardware budget.
- Do NOT compare arms across different backbones (violates I1).
- Do NOT extract diffusion features at full noise (`t → T`); the ablation should *show* why, not assume it.
- Do NOT touch the test split for model selection.
- Do NOT silently change the probe recipe per arm.

## Acceptance criteria

1. All three (or two, if JEPA deferred) arms train to completion on MPS from `configs/default.yaml`.
2. A results table: rows = {diffusion, MAE, supervised-100%, supervised-10%}, columns = {linear-probe AUROC/AUPRC, fine-tune AUROC/AUPRC}, mean ± std over seeds.
3. An AUROC-vs-timestep plot for the diffusion arm showing the rise-then-fall shape.
4. Encoder param-count parity assertion passes; label-leakage assertion passes.
5. README explains the design, the invariants, how to run each milestone, and how to read the two headline outputs (the table + the timestep plot).

Start with M0 and stop for review before M1.
