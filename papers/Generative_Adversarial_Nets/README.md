# Generative Adversarial Nets — the original GAN, and the non-saturating trick

Notes and experiments on **Generative Adversarial Nets** (Goodfellow et al., 2014) — the paper
that introduced the adversarial game between a generator `G` and a discriminator `D`.

The thread running through this folder is the paper's **non-saturating loss trick**: why the
literal minimax objective gives the generator almost no gradient early in training, and what to
do about it.

## Contents

| Path | What it is |
|------|------------|
| `notes/GAN_paper_study_notes.md` | Intuition-first walkthrough of the paper — the value function term by term, the min/max confusion, the non-saturating trick, optimal `D` → JSD, and how it's trained. |
| `notes/experiment_findings.md` | Research write-up of the experiments: the questions asked, where gradients vanished vs. exploded, and why Adam hid the effect while SGD revealed it. |
| `experiments/gan_sat_vs_nonsat_8blobs.ipynb` | Experiment on a 2D 8-mode Gaussian ring. Makes the saturating-vs-non-saturating gradient gap **visible**, and shows why **Adam hides it while SGD reveals it**. Runs in seconds. |
| `experiments/gan_sat_vs_nonsat_mnist.ipynb` | The same experiment on MNIST images (GPU-ready). The gradient story shows up as sample sharpness early in training. |
| `experiments/gan_observation.ipynb` | Earlier exploratory notebook (kept for reference; superseded by the two above). |
| `paper/generative-adversarial-nets-Paper.pdf` | The source paper. |

> `experiments/data/` (MNIST) is **not** committed — the MNIST notebook downloads it
> automatically via `torchvision` on first run.

## The key result from the experiments

The non-saturating trick is fundamentally a **gradient-magnitude** fix, so whether you can *see*
it depends on the optimizer:

- When `D` is confident (`D(G(z)) ≈ 0`, the realistic early-training regime), the saturating
  loss `log(1 − D(G(z)))` delivers a raw generator gradient **~1000× weaker** than the
  non-saturating `−log D(G(z))`.
- **SGD** preserves that gap → the saturating generator is starved and barely learns.
- **Adam** divides each gradient by its running RMS (`m̂/√v̂`), renormalizing the ~1000× gap to
  ~1× → it **masks** the saturating loss's weakness. This is why a naive Adam run shows little
  difference between the two losses.

## One thing to remember

Prefer the non-saturating loss for the *early gradient* it gives `G`. The textbook saturation
effect is clearest under plain **SGD**; Adam's per-parameter normalization papers over it.
