# RINE — Synthetic Image Detection Notes

**Paper:** *Leveraging Representations from Intermediate Encoder-Blocks for Synthetic Image Detection (RINE)*
**Authors:** Christos Koutlis, Symeon Papadopoulos (CERTH-ITI)
**Venue:** ECCV 2024 · [arXiv:2402.19091](https://arxiv.org/abs/2402.19091) · [Code](https://github.com/mever-team/rine)

---

## The core problem

Synthetic Image Detection (SID): telling AI-generated images apart from real ones, and crucially doing so in a way that **generalizes** to generators the model never trained on.

The field's recurring pain point: detectors trained on one type of fake-image machine (e.g. GANs) fall apart when shown fakes from a different machine (e.g. diffusion models).

### Simpler version

- Train a detector on fakes from **only one** fake-image machine (Machine A) → it gets great at catching Machine A's fakes because it learns Machine A's specific "fingerprints."
- But there are **many** machines (GANs, diffusion models like DALL-E / Midjourney), and each leaves **different** fingerprints.
- So a Machine-A-trained detector meets a fake from **Machine B** and fails — it was looking for the wrong fingerprints.
- Catching fakes from machines you **never trained on** = *generalization*, and it's the hard part. New generators appear constantly, so a detector that only catches what it trained on isn't useful in the real world.
- **Goal:** learn something *general* about what makes an image fake, not just one generator's quirks.

RINE does exactly this: trains only on one old generator (ProGAN) but still catches fakes from 20 different newer ones.

---

## The key insight

Prior SOTA (UFD, "Universal Fake Detector") found that taking features from CLIP's image encoder + simple linear probing generalizes surprisingly well across generator families.

**The catch:** UFD only uses CLIP's **final layer**, which encodes high-level *semantic* content ("this is a dog"). But the actual fingerprints of synthetic generation — subtle texture, frequency, and pixel-level artifacts — live in **low-level features**, better preserved in the **intermediate/shallow layers**.

---

## The RINE method

Instead of using only CLIP's last layer, RINE harvests the CLS token from **every intermediate Transformer block** of CLIP's (frozen) ViT encoder, then combines them. Three small trainable pieces sit on top of frozen CLIP:

1. **Projection network** — maps the concatenated CLS tokens into a "forgery-aware" feature space.
2. **Trainable Importance Estimator (TIE)** — a learned weighting that decides how much each block contributes, since different layers matter differently.
3. **Classification head** — trained with two losses jointly:
   - Binary cross-entropy (BCE) → directly optimizes the real-vs-fake decision.
   - Supervised contrastive loss → tightens the real-vs-fake clustering in feature space.

CLIP stays **frozen** throughout — only these small modules (~6.3M params for the best model) are trained.

---

## Key findings

- **Headline:** trained only on ProGAN, evaluated on **20 test datasets** (GANs, diffusion, deepfakes, low-level vision, DALL-E) → beats prior SOTA by **+10.6% accuracy** (+4.5% mAP) with the 4-class model. The 1-class model still beats every baseline.
- **Tiny training budget:** best models train in a **single epoch (~8 min)** on one RTX 3090 Ti. More epochs didn't help (sometimes hurt).
- **Data-efficient:** performance barely degrades even at 20% of training data (AP stays nearly flat).
- **Ablations confirm the thesis:** removing intermediate representations causes the biggest drop (collapses back to ~UFD level). TIE and contrastive loss add smaller but real gains.
- **Importance analysis (nicest evidence):** inspecting TIE's learned weights shows many *intermediate* blocks get higher max-importance than the final block — directly supports the motivation.
- **Robustness:** holds up well under cropping and JPEG compression (social-media conditions); moderate degradation under blur/noise.

---

## Limitations & future work

- **Compound perturbations hurt:** blur + crop + compress + noise applied together drops performance to roughly current-SOTA-without-perturbation levels.
- **Training-data origin still bounds generalization:** on Synthbuster (commercial diffusion tools), a ProGAN-trained model does poorly on some modern generators — notably **DALL-E 3 collapses to ~21% accuracy**. Training on diffusion data fixes most of it.
- **Inherits CLIP's ceiling:** swapping CLIP for a plain ImageNet ViT or Wang's detector as backbone tanks performance → method is tightly coupled to the foundation model's feature quality, not backbone-agnostic.
- **Simple fusion:** the fusion mechanism is deliberately simple (FPN-style fusion doesn't beat it); more sophisticated cross-layer fusion is open.
- **Implied future work:** better fusion, training on more diverse (esp. diffusion/commercial) generators, hardening against compound real-world perturbations.

---

## Relevance to my work

- Sits squarely in the same domain as the **Etsy AI-image-detection project** (CLIP ensemble, AI-generated image detection).
- RINE's pitch: a single well-designed CLIP-feature extractor with learned layer weighting can outperform heavier approaches at a fraction of the training cost — a useful comparison point against a multi-model Optuna ensemble.
- The **TIE idea** (learning per-layer importance) is a portable trick worth borrowing.

---

## Open threads to discuss next

- [ ] Architecture diagram
- [ ] How the contrastive loss / TIE module is implemented (for a possible reimplementation)
