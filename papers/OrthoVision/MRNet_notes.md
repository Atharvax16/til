# MRNet — Study Notes

**Paper:** Bien, Rajpurkar, Ball, et al. (2018). *Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet.* PLOS Medicine 15(11): e1002699.

**Why this paper matters for OrthoVision:** It is the direct ancestor of the 2026 RSNA Knee MRI competition — same task shape (read a knee MRI, output findings), same "labels came from radiology reports" setup. It was single-institution; the competition is 16 institutions. The two hardest problems in *your* version (multilingual report→label extraction, and surviving the jump across sites) are exactly the two things this paper didn't have to solve. That's what makes your setup publishable rather than a re-run.

---

## 1. The problem they solved

Reading a knee MRI is slow and inconsistent — even expert radiologists disagree with each other. MRNet reads a knee exam and answers three yes/no questions:

- Is anything **abnormal**?
- Is there an **ACL tear**?
- Is there a **meniscus tear**?

Three binary labels per exam. (Yours is bigger — 12 findings — but same shape.)

---

## 2. The data

- **1,370 knee exams**, all from **one hospital** (Stanford), all on **GE scanners**. This single-site detail is the whole story of the paper — hold onto it.
- Each exam has several series; they used three: **sagittal T2, coronal T1, axial PD**.
- **Labels came from the radiology reports** — they just extracted them *by hand*. You're doing the same extraction, except automated and across 9 languages, which is the part they got to skip.
- Class balance: 80.6% abnormal, 23.3% ACL tears, 37.1% meniscal tears. Imbalanced — they handled it with prevalence-weighted loss.

---

## 3. The architecture — the clever core

MRNet climbs the **slice → series → study** ladder in two moves. Both moves are the *same verb*: **collapse a group into one summary.**

### Move 1 — slices into a series verdict

Take one series (e.g. the sagittal stack, ~30 slices).

1. Run **every slice** through an ImageNet-pretrained **AlexNet** feature extractor → one feature vector per slice.
2. Collapse the ~30 slice-vectors into one using **max-pooling across slices**.
3. That one vector → a probability for that series.

**The intuition for max-pooling:** a tear only shows up on a couple of slices, so "let the most suspicious slice win." A focal finding on 2 slices survives; it isn't diluted by 28 normal ones.

### Move 2 — series into an exam verdict

They train a **separate small CNN for each series and each task** (3 series × 3 tasks = 9 CNNs), then use a plain **logistic regression** to weight the three series' opinions into one final answer per task.

- For **one knee**, each of the 3 series gives its own probability for (say) ACL.
- Logistic regression = a **learned weighted average** of those 3 probabilities.
- When they inspected the learned weights: **axial mattered most for meniscus, coronal for ACL** — matching radiologist intuition. Nice interpretability sanity check.

**Why not one big fusion network?** They chose the dead-simple weighted vote because it's tiny, trains in seconds, and the weights are *readable* ("coronal matters most for ACL"). A big network would work but tell you nothing about *why*.

### Performance (single-site, internal validation)

| Task | AUC |
|---|---|
| Abnormality | 0.937 |
| ACL tear | 0.965 |
| Meniscus tear | 0.847 |

Meniscus is the weakest — meniscus tears are genuinely the hardest of the three. **Expect that to hold for you too.**

---

## 4. THE finding that should shape your whole strategy

They tested the Stanford-trained model on a **different hospital** (Croatia; Siemens scanner, T1 instead of T2, different labeling rules):

- ACL AUC fell **0.96 → 0.824** with no retraining.
- It only recovered to **0.911** *after retraining on the new hospital's data.*

**Sit with this.** A model that looked near-perfect at home lost a big chunk of skill the moment it saw a new machine. This is **domain shift**, and your competition is built to punish it: MRNet was 1 hospital; your test set spans **16 institutions across 5 continents.** Whatever leaderboard number you get on data resembling your training set will *overstate* how you do on unseen sites.

The 0.96 → 0.824 drop is the number to quote when you frame *why* cross-site generalization matters. It's a ready-made motivation for a research question.

---

## 5. Secondary result (less central for you)

Giving the model's predictions to radiologists made them slightly better (mainly more specific on ACL) and made them agree with each other more (higher Fleiss kappa). This is the "AI as assistant" / clinical-workflow angle — interesting, but a workflow claim, not a modeling one. Probably not your lane.

---

## 6. What to steal for OrthoVision

- **Max-pool-across-slices is your cheap, strong baseline** for turning a slice-stack into a prediction. Steal it on day one. Modernize the backbone later (AlexNet → ResNet/EfficientNet, or 3D/attention).
- **Per-series models + a simple learned combiner is a proven fusion recipe.** Don't over-engineer multi-series fusion before you have a baseline.
- **Expect meniscus-type findings to lag** the ligament findings.
- **Their AUCs are a ceiling to be suspicious of** — single-site. Your honest target is "does it hold across institutions."
- **They extracted labels from reports manually.** Your automated multilingual extractor is the harder, more novel version.

**One wrinkle:** MRNet had exactly 3 clean series per knee, so a fixed "3 weights" logistic regression fit perfectly. Your 16-institution data will be ragged — some knees with 4 series, some 6, some missing a plane. A fixed-3 combiner won't fit; you'll want a fusion step that tolerates a variable, ragged set of series (an argument for attention-style pooling at the *series* level too, not just the slice level).

---

## 7. Mechanical clarifications worked through

### 7a. The imaging hierarchy (get the vocabulary airtight)

- A **slice** = one photo, one thin cross-section. All slices *inside a series* share the **same angle and setting**; they differ only by *where* along the knee they cut.
- A **series** = one full stack of slices — **one angle, one setting**, a 3D volume. "Sagittal T2" is one series; "Coronal T1" is a *different* series.
- A **study** = one knee exam = **several series** (usually 5–6) bundled together. Angle (sagittal/coronal/axial) and setting (T1/T2/PD) vary **between** series, not inside one.

**The correction that keeps coming up:** "various angles" belongs at the **study** level, not the slice level. Within any single series, both angle and setting are locked.

`train_series.csv` proves the multiplicity: ~24,000 series ÷ ~4,400 studies ≈ **5.5 series per knee.**

### 7b. "3 series" means 3 VIEWS of ONE knee — not 3 knees

Everything in Move 2 happens inside **one** knee. The three series (sagittal, coronal, axial) are three camera angles on the **same** joint. Three mini-models each judge that one knee from their angle; the weighted vote combines the three judgments into that knee's final score. Three angles collapsing to one verdict — same "many → one" move as slices collapsing to a series vector, one rung higher.

### 7c. How the slice-stack collapses to a single feature vector

Max-pooling **never keeps a slice.** It keeps **values, feature by feature**, and the winning value for different features can come from different slices.

Picture the post-CNN grid for one series: `30 slices × 256 features`.

```
              feat_0  feat_1  ...  feat_255
slice_0        0.1     0.0    ...   0.2
slice_14       0.1     0.9    ...   0.1     ← tear visible here
slice_15       0.1     0.8    ...   0.2     ← and here
slice_29       0.0     0.1    ...   0.1
```

Max-pooling operates **down each column** (the slice axis): for every feature, keep the single highest value over all 30 slices.

- feat_1 → 0.9 (peaked on slice_14, where the tear is)
- every other feature → its own column max

Result: one vector of 256 numbers. `30 × 256 → 256`. **The slice axis is gone; the feature axis survives.**

**Key subtlety:** different features can win on different slices, and that's *desirable*. If an effusion is brightest on slices 2–3 and a tear on slices 14–15, the final vector holds "strong fluid evidence *somewhere*" AND "strong edge-disruption *somewhere*" — even though those somewheres were different slices. The vector summarizes *"what strong evidence appeared anywhere in this series,"* feature by feature. That's why it can carry multiple findings that never co-occur on a single slice.

**One-line takeaway:** max-pool-across-slices = "for each feature, keep its peak response over the whole stack, forget where it occurred."

### 7d. The model doesn't know slices belong to one knee — the dataloader does

The grouping lives in the **IDs, not the pixels.** Every slice carries StudyInstanceUID, SeriesInstanceUID, and its position. In preprocessing *you* group slices → series → study by those IDs. By the time a training example is assembled, it's already "all of knee #4407's images, grouped."

**The model never sees loose slices.** One forward pass = one pre-grouped knee. Grouping is a data-engineering fact you *impose*, not a perceptual fact the model *infers*.

**Why this is load-bearing:** labels attach at the **study** level, so you *can't* train slice-by-slice — a single slice has no label of its own. Group wrong and you get thousands of images all wearing the same study-level label, most showing normal anatomy → silent label-noise disaster.

**Two traps that follow:**

- **Leakage:** split at the **StudyInstanceUID** level (or patient level if available), never at the series level. Otherwise a knee's series land on both sides of the split and the model "recognizes" the knee — inflated validation. MRNet did this ("all exams from each patient in the same split").
- **Ragged bundles:** different knees have different slice counts and series counts. You'll need pad/truncate strategies and a rule for missing series.

---

## 8. Pooling: max is opinionated, not universally right

Max-pooling bakes in an assumption: **"the finding is the peak slice."** True for focal findings (tears), false for diffuse ones.

- **Great for focal/sharp** (meniscus tear, ACL tear, fracture) — a 2-slice signal survives.
- **Blind to diffuse/cumulative** (effusion, synovitis, osteoarthritis) — a feature at 0.4 on twenty slices, never spiking, is represented by that single 0.4. Max has no notion of *how many* slices lit up, only *how bright the brightest was.* The "present on 20 slices" fact evaporates.

**Averaging is the opposite failure:** it dilutes focal findings (2 loud slices drowned by 28 quiet ones).

**Attention pooling** is the modern fix: learn a weight per slice and take a weighted sum, so it can concentrate weight on the 2 tear-slices for a focal finding *or* spread weight across many slices for a diffuse one. Same collapse (`30 × 256 → 256`), smarter rule for combining the column. Other options: log-sum-exp (a tunable dial between mean and max), or concat max+mean so the net sees both peak and accumulation.

### The hidden research question

> Does the slice-aggregation rule (max vs mean vs attention) systematically change *which kinds* of findings a knee-MRI model detects — focal vs diffuse — and does the wrong choice quietly hurt the rare, high-stakes ones?

Testable, cheap (swap one layer, hold everything else fixed), and dead-center in your research thread (when a model can be trusted; whether it's silently failing on rare cases). The right way to test it: a **synthetic sandbox first** (controlled focal vs diffuse volumes, energy-matched so the only difference is spatial distribution), then confirm on real knee MRI. Measure the **interaction** (pooling type × finding type), not just overall accuracy — a "max beats mean by 0.09 on focal but 0.01 on diffuse" result is the publishable shape.

---

## 9. Quick-reference summary

| Level | Unit | How it collapses to the next level |
|---|---|---|
| Slice | one cross-section photo | — |
| Series | stack of same-angle slices (3D volume) | **max-pool across slices** → one 256-vector → one probability |
| Study | 3–6 series (one knee) | **logistic regression** (learned weighted vote) → one final score per task |

**The recurring motif:** every rung does the same thing — collapse a group into a single summary. Once that's fixed in your head, "3 series" can never again read as "3 knees."

**The one number to remember:** external-validation drop **0.96 → 0.824**. That's the evidence that domain shift is real, and the hook for your generalization research angle.
