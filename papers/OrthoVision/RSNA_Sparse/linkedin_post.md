# LinkedIn post

Structure follows §8 of the project outline. Every number below comes from `reports/results.csv`.

**Attach:** `reports/figures/06_summary.png` (one-slide summary) and
`reports/figures/02_top_activating.png` (top-activating images per candidate feature).

---

## Main post

While reading about knee abnormality detection, I kept circling the same question from different
directions: how can a model keep the evidence that matters while making its own internal
representation easier to inspect?

Biomedical image understanding is not only classification. For anatomy-related question answering, a
system has to connect visual evidence to concepts a radiologist would name — meniscus, ACL,
cartilage, effusion, bone marrow abnormality. BioMedCLIP gives you a shared image–text space for
that. What it does not give you is a readable one: its embeddings are 512 dense dimensions, and no
single dimension means anything on its own.

So I tested a hypothesis: a sparse autoencoder trained on top of a frozen BioMedCLIP encoder should
decompose those dense embeddings into a larger dictionary of sparse **candidate** concepts, and a
question-aware memory should then be able to keep only the few features a given question needs.

The experiment, small enough to run on a MacBook: freeze BioMedCLIP, extract one embedding per
image, train only a sparse autoencoder (512 → 1024, ~33 active features per image), rank the
top-activating images for each feature, align features to a controlled anatomical vocabulary through
the text encoder, and compare dense versus sparse evidence for four anatomical questions.

**Important:** this run used a *synthetic* knee-MRI-like corpus I generated with known ground truth —
6,000 images, 1,000 simulated patients, patient-level splits. Not RSNA data, not real patients, and
nothing here transfers to real knee MRI. I used synthetic data deliberately, because it is the only
way to know what the "right answer" is when you are testing an interpretability method rather than a
diagnostic one.

Three results, including the ones that did not go my way.

**1. The sparse features are more selective — but they are not better detectors.** Each informative
sparse feature responds to 2.0 attributes on average versus 4.7 for a dense dimension, and 32 % of
them respond to exactly one attribute versus 0.6 % of dense dimensions. But dense dimensions
separate pathology more cleanly (precision 0.89 vs 0.84 at matched coverage). The sparse code buys
readability, not accuracy. Both halves belong in the same sentence.

**2. Compression mostly holds.** Keeping 16 of 1024 features — 32 numbers instead of 512 — answers
the four questions within 0.015 AUROC of the full dense embedding (0.915 vs 0.930). A random
selection of the same size collapses to 0.55.

**3. The routing idea I was most attached to did not work.** Selecting features by their similarity
to the question's text embedding performed *worse* than simply keeping the loudest features
(0.869 vs 0.892 at small memory sizes). Question-aware retrieval sounds obviously right. It wasn't,
here.

And the finding I would want any reader to take away: **57 % of the usable features were at least as
enriched for an acquisition attribute — imaging plane, pulse sequence, scanner — as for any
pathology.** A feature whose top images all show a torn ligament may be encoding the fact that those
images are T2-weighted. Automated feature labels are hypotheses, not explanations, which is why
every feature in this project is called a *candidate* concept until someone qualified reviews it.

This is an interpretability and representation-learning study, not a diagnostic tool and not a
clinical decision-support system.

Can sparse memory become a useful interface between biomedical vision-language models and anatomical
reasoning — or does compression remove the very evidence needed for a faithful answer?

I would genuinely like to be argued with here, particularly by people working on medical imaging,
mechanistic interpretability, JEPA-style representation learning, or memory-augmented models. Two
things I am unsure about: whether pooled slice embeddings throw away too much to make this
meaningful on real MRI, and whether text-derived routing can be made to beat magnitude once the
concepts are less correlated than mine were.

---

## First comment (post links here, not in the post body)

Notebook, synthetic data generator and full evaluation: [repo link]

Built on:
- BioMedCLIP — https://arxiv.org/abs/2303.00915
- I-JEPA — https://arxiv.org/abs/2301.08243
- MovieChat (dense token → sparse memory) — https://github.com/wenhaochai/MovieChat
- MedSAE — https://arxiv.org/abs/2510.26411

Method notes for anyone reproducing it: the sparsity coefficient is adapted during training to hold
L0 at a target, because a fixed λ does not transfer between encoders. Selectivity is measured with
precision@k rather than AUROC — AUROC is structurally capped for sparse units (a feature firing on
5 % of images cannot exceed 0.625 against a 20 %-prevalence label), so an AUROC comparison against
dense dimensions is decided before it starts. Controls: a width- and L0-matched random feature bank,
a shuffled-label null, and acquisition attributes scored alongside the pathology labels.

---

## Shorter variant (~150 words)

I trained a sparse autoencoder on frozen BioMedCLIP embeddings of knee images to test whether dense
biomedical representations can be turned into sparse, inspectable candidate concepts — and whether a
question-aware memory can keep only what a question needs.

On a synthetic knee-MRI corpus with known ground truth (6,000 images, 1,000 simulated patients — not
RSNA data, not real patients):

• Sparse features are more selective than dense dimensions (2.0 vs 4.7 attributes per unit) but
  worse detectors (precision 0.84 vs 0.89).
• 16 of 1024 features — 32 numbers instead of 512 — answer four anatomical questions within 0.015
  AUROC of the full embedding.
• Routing features by question text lost to simply keeping the loudest ones.
• 57 % of features tracked imaging plane, pulse sequence or scanner at least as strongly as any
  pathology.

That last one is the point. A feature that looks like anatomy may be encoding the protocol.

Interpretability study, not a diagnostic system.
