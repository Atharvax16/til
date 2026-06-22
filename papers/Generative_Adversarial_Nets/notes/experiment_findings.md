# Saturating vs. Non-Saturating GAN Loss — Experiment Findings

My research notes on the experiments in `../experiments/`. The study of the *paper* is in
`GAN_paper_study_notes.md`; this file is about **what I actually ran and observed**, and the
questions that drove it.

Reproduce everything with:
- `../experiments/gan_sat_vs_nonsat_8blobs.ipynb` (2D 8-mode ring, runs in seconds)
- `../experiments/gan_sat_vs_nonsat_mnist.ipynb` (MNIST, GPU)

---

## The questions I set out to answer

1. **Does the saturating loss really starve the generator of gradient early on**, the way
   Goodfellow et al. (2014) claim — and if so, *where exactly* does the gradient vanish?
2. **Why did my first attempt show nothing?** My earlier run (`gan_observation.ipynb`) couldn't
   see any difference between the two losses. Was the theory wrong, or was my setup hiding it?
3. **Does the optimizer change what you observe?** Specifically, does **Adam vs SGD** change the
   *conclusion*, or just the *appearance*?
4. **Does the same story hold on real images (MNIST)**, not just a toy 2D distribution?

The design is a 2×2(×2) sweep — `{saturating, non-saturating} × {Adam, SGD}`, repeated on the
8-blob ring and on MNIST — with **per-step logging** of `lossG`, `lossD`, `‖∇θ_G‖`,
`D(real)`, and `D(G(z))`, so nothing is averaged away.

Definitions used throughout (`d` = discriminator logit, `p = D(G(z)) = σ(d)`):
- **saturating**: `G` minimizes `log(1 − D(G(z)))` → grad factor `∂L/∂d = −p`
- **non-saturating**: `G` minimizes `−log D(G(z))` → grad factor `∂L/∂d = −(1 − p)`

---

## Finding 1 — The gradient *does* vanish, exactly where the theory predicts

Measured at initialization, after giving a fresh `D` a short head start so it confidently
rejects the generator's fakes — i.e. the realistic early-training regime. Same generator, same
batch, only the loss differs:

| quantity | value |
|---|---|
| mean `D(G(z))` on fakes | **0.0003** (D is confident — the regime the effect needs) |
| `‖∇θ_G‖` saturating | **0.0021** |
| `‖∇θ_G‖` non-saturating | **5.37** |
| ratio (non-sat / sat) | **~2,600× stronger** |

**Where the gradient vanishes:** the *saturating* loss, when `D` is confident (`D(G(z)) ≈ 0`),
i.e. early in training. Its gradient factor is `−p → 0`, so it multiplies the whole
backpropagated signal by ~0. The non-saturating factor is `−(1−p) → −1`, so it stays strong.
This is the paper's claim, confirmed through a real MLP — not just the analytical curve.

---

## Finding 2 — The Adam mystery, solved: the optimizer decides whether you can *see* it

This is the most important practical finding, and the answer to question 2. Feeding those same
gradients through **one optimizer step** and measuring the actual parameter change `‖Δθ_G‖`:

| optimizer | sat step | non-sat step | non-sat / sat |
|---|---|---|---|
| **SGD**  | 0.00002 | 0.049 | **~2,435×** (gap preserved) |
| **Adam** | 1.168   | 1.209 | **~1.0×** (gap erased) |

**Why:** Adam divides each gradient by its own running RMS (`m̂ / √v̂`), which makes the update
**scale-invariant** — a tiny gradient gets divided by a tiny `v̂` and is restored to a ~`lr`-sized
step. So Adam *silently rescues the saturating generator*, normalizing the 2,600× gradient gap
down to ~1×. Plain SGD has no such normalization, so the gap survives into the parameter update.

**This is why my first run saw nothing:** it used Adam only. The saturating-loss weakness was
real the whole time — Adam was just hiding it. The vanishing-gradient problem is a
*gradient-magnitude* problem, and Adam is precisely the tool that papers over gradient magnitude.

---

## Finding 3 — What the per-step training curves show

Running the full 2×2 sweep with per-step logging, the curves split cleanly by optimizer:

- **SGD + saturating** — `‖∇θ_G‖` stays near zero; the generator is **frozen** and its samples
  barely leave the initial blob. `lossG = log(1 − D(G(z)))` sits pinned near 0 (that flat line
  *is* the saturation).
- **SGD + non-saturating** — strong gradient from step 1; the generator moves out to cover the
  ring. `lossG = −log D(G(z))` sits high while `D` is winning.
- **Adam + (either loss)** — both train comparably well. The two losses look almost identical,
  which is exactly the masking from Finding 2.

I also confirmed the **precondition** that my earlier run failed: with tight modes (`STD = 0.05`)
and enough D capacity, `D(G(z))` actually drops toward 0 early on. My first attempt was stuck
near `D ≈ 0.4–0.5` (D never confident), so there was no saturation regime to observe in the
first place. Two separate bugs were stacked: a never-confident `D`, and Adam masking.

---

## Finding 4 — The other tail: where gradients get *large*

The non-saturating loss is the opposite failure mode of the saturating one. When `D` is
confident, `−log D(G(z))` blows up (→ ∞ as `D(G(z)) → 0`), so its gradient is **large** exactly
when the saturating one vanishes — that is the 5.37 vs 0.0021 split in Finding 1. This is the
*intended* strong early signal, but it is also why the non-saturating generator can produce
**large, spiky gradient norms** early in training (the reason DCGAN-era recipes pair it with a
low Adam learning rate and `β₁ = 0.5`). So:

- **saturating loss → vanishing** generator gradient when `D` wins (no learning),
- **non-saturating loss → large** generator gradient when `D` wins (useful, but watch the spikes).

Same fixed point, opposite gradient behavior on the way there.

---

## Finding 5 — MNIST: the same story in image space

The MNIST notebook repeats the 2×2 sweep on real images (MLP G/D, GPU). The expected and
observed pattern matches the toy result: **SGD + saturating** stays noisy/blurry (starved G),
while the other three produce recognizable digits, and the **Adam** panels look similar across
both losses (masking again). With images you additionally *see* the gradient story as sample
sharpness over the first few epochs. (Run on GPU to capture the exact loss curves and sample
grids — the per-step logs and fixed-noise sample grids are built into the notebook.)

---

## Bottom line

- **Yes**, the saturating loss starves the generator — its gradient vanishes precisely when `D`
  is confident (early training), measured at ~2,600× weaker than non-saturating.
- **The optimizer decides visibility, not the verdict.** SGD shows the starvation plainly; Adam's
  per-parameter normalization hides it (~2,435× → ~1×). My first null result was an Adam artifact
  plus a never-confident discriminator, not evidence against the theory.
- **Prefer the non-saturating loss** for the usable early gradient it gives `G`. If you want to
  *see* the textbook effect, use **SGD** and make sure `D` actually reaches confidence first.
