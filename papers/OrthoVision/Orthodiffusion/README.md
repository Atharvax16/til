# ssl-medical-compare — controlled comparison of SSL objectives on medical images

One question: **holding the encoder and compute budget fixed, how do self-supervised
objectives compare as medical-image representation learners, and at what abstraction level
(diffusion timestep) does the denoising objective peak?**

Everything lives in `ssl_medical_compare.ipynb`; run it top to bottom. Outputs land in
`results/` (CSV + `tables.md` + `figures/`) and `checkpoints/`.

## Design

* **Dataset** — PneumoniaMNIST (`medmnist`), official train/val/test splits, 64x64,
  grayscale, normalized to [-1, 1]. The dataset is imbalanced, so **AUROC and AUPRC are the
  primary metrics**; the BCE loss is class-weighted identically for every arm.
* **Shared encoder** — a small conv U-Net trunk (base 32, mults [1, 2, 4, 8],
  GroupNorm + SiLU, 2 mid blocks), 4,058,656 parameters, always carrying a
  sinusoidal timestep embedding injected as FiLM. Non-diffusion arms pass `t=0`.
* **Arms** — diffusion (DDPM, T=1000, cosine schedule, eps-prediction),
  MAE/SimMIM (patch 8, mask ratio 0.6), JEPA (implemented, disabled by default).
* **Downstream** — frozen linear probe, then full fine-tune, identical recipe for every arm,
  plus supervised-from-scratch baselines at 100% and 10% labels and a random-init probe control.

## Invariants (the scientific point)

| # | Invariant | Enforcement |
|---|---|---|
| I1 | one shared encoder class + config | `build_encoder()`; `assert_param_parity()` compares count *and* (name, shape) signature |
| I2 | one feature-extraction site | `Encoder.extract()` / `extract_features()`; only `t` varies |
| I3 | pretraining is label-blind, train split only | `UnlabeledImageDataset` holds no labels; `label_blind()` makes labeled reads raise |
| I4 | matched compute budget | one `pretrain:` config block: 4000 steps, AdamW, warmup+cosine, for every arm |
| I5 | matched downstream protocol | `probe:` / `finetune:` config blocks; the functions take no per-arm hyperparameters |
| I6 | honest test discipline | seeds [0, 1, 2], mean ± std, selection on val, test evaluated once per config |

## Milestones

| cell | does |
|---|---|
| M0 | data shapes, class balance, sample grid, label-leakage assertion |
| M1 | encoder construction + parameter-parity assertion |
| M2 | train the DDPM, sample images, show the forward process at the probed timesteps |
| M3 | train SimMIM, visualize reconstructions |
| M4 | linear probes for every arm + random control + supervised baselines -> first table |
| M5 | timestep/block ablation -> `figures/m5_auroc_vs_timestep.png`, val-selected `t` |
| M6 | fine-tune every arm -> final arm x {probe, finetune} table |
| M7 | JEPA (optional) |

## Reading the two headline outputs

**The table** (`results/tables.md`). Compare *within a column*: the probe column
answers "how linearly separable is the frozen bottleneck", the finetune column answers "how good
an initialization is this". A pretrained arm only earns a claim if it beats **both** the
random-init probe control and supervised-10%.

**The AUROC-vs-t plot** (`figures/m5_auroc_vs_timestep.png`). Expect rise-then-fall. Small `t`:
the denoiser only has to fix fine detail, so features are texture-level. Mid `t`: the task
requires reconstructing structure that is genuinely gone, which forces semantic features —
this is where the curve should peak. Large `t`: the input is mostly noise, the model falls back
on the prior, and the features stop describing *this* image. That peak is why you never extract
at `t -> T`, and the ablation is meant to *show* it rather than assume it.

## Running

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1     # some ops have no MPS kernel
jupyter lab                              # then: Restart & Run All
```

Leave `SMOKE = True` for a full end-to-end dry run on a subset (~2 min), then set it to
`False` for the real run. Measured on an M-series laptop (MPS, batch 128 @ 64px):
~0.43 s/step for diffusion and ~0.37 s/step for MAE, i.e. ~30 min per arm per seed at
4000 steps (~3 h for the full grid) plus ~30 min for fine-tunes, baselines and the
sweep. To shorten it, cut `eval.seeds` to `[0]` or lower `pretrain.steps` **for every arm at
once** — a per-arm budget would break I4. Checkpoints are reused across runs; pass `force=True` to
`pretrain_arm` to retrain. Determinism is not guaranteed on MPS, which is why every number is
reported as mean ± std over 3 seed(s).

## Known asymmetry (flagged, not hidden)

The diffusion decoder keeps U-Net skip connections while the MAE decoder does not
(`diffusion.decoder_skips` / `mae.decoder_skips`). Skips let a denoiser route detail *around*
the bottleneck we probe; removing them from DDPM hurts sample quality. Both are config flags,
so the effect is measurable rather than assumed.

## Non-goals

Not a reproduction of full OrthoDiffusion (3D, three orientation models, 16k volumes). No
cross-backbone comparisons (that would violate I1). No test-split model selection.
