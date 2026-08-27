# Attention-Based Deep Multiple Instance Learning — Study Notes

Notes built from a walkthrough of **Ilse, Tomczak & Welling, "Attention-based Deep Multiple Instance Learning" (ICML 2018)**, arXiv:1802.04712 — plus how it maps onto a 3D knee-MRI problem (RSNA-style) and the open research question of spatial / cross-plane structure.

> Verify specific numbers and citation names against the actual PDF — these notes are conceptual and written to be re-read, not quoted.

---

## 1. The core problem: Multiple Instance Learning (MIL)

**Normal supervised learning:** one item → one label. "This picture is a cat." The model learns from labels attached to every individual example.

**MIL:** you get a label for a whole *group* (a **bag**) of items, and you are **not** told which item inside earned that label.

The classic bag rule:

- A bag is **positive** if *at least one* instance inside is positive.
- A bag is **negative** only if *every* instance is negative.

So a positive bag is deliberately messy: it holds some positive instances **mixed together with** many negative ones, and the model has to figure out which is which on its own.

### Key correction to a common misconception
You do **not** group all the "1" outputs together and all the "0" outputs together — that would just be ordinary two-class training. MIL *cannot* do that, because the whole premise is that the per-instance labels are **unknown**. Each bag stays as a single sample's mixed collection of pieces.

### When is it actually MIL? (the trigger)
> Your label is attached to a **group**, but you feed the model the **individual members**, and nobody told you which member earned the label.

- **One label per whole image → NOT MIL.** That's ordinary supervised classification.
- **Label coarser than the input you feed in → MIL.** e.g. "this patient has the condition *somewhere*" but not *which slice*.

### The motivating example — computational pathology
- Whole-slide image = **bag**.
- Patches cropped from it = **instances**.
- You know the patient has cancer (bag label) but nobody marked *which patches* contain tumor.
- MIL lets you train on the cheap, already-available bag label and still recover a rough instance-level signal.

---

## 2. What MIL buys you (and what it does NOT)

**MIL's advantage is LABEL efficiency, not COMPUTATIONAL efficiency.**

- It is **not** faster or lighter to compute — often *heavier*, since every instance is pushed through the network.
- It does **not** "beat" fully-supervised training on accuracy. If you had a label on every instance, plain supervised training would almost always win.
- What it saves is **expensive human annotation.** Getting a specialist to mark every slice/tile is brutally slow and costly. MIL learns from the cheap label you already have (e.g. the patient-level diagnosis in the record).

This is why MIL is called **weakly supervised** learning: you learn from a weaker, cheaper signal than you'd ideally want. You reach for it when fully-labeled training simply *isn't possible* because the fine-grained labels don't exist and paying humans to make them is off the table.

---

## 3. The framework: three steps + one hard constraint

The paper formalizes bag prediction as three steps:

1. **Transform** — push each instance `x_k` through a shared neural net `f` → an embedding (vector) `h_k`. 40 slices → 40 vectors. Nothing combined yet.
2. **Aggregate** — squash the set `{h_1, …, h_K}` into a single bag vector `z` using a **permutation-invariant (order-blind)** pooling operator.
3. **Classify** — feed `z` through a final net `g` → the bag label.

### The constraint: permutation invariance
Shuffling the items in a bag must **not** change the answer. A bag of knee slices is just a *pile*; reordering how you stacked them can't change whether the knee has a tear. So the aggregation step must be order-blind (sum, mean, max are order-blind; anything that reads the items as an ordered sequence is not).

### Deep Sets (the theoretical permission slip)
Any permutation-invariant function can be written in the form `g( Σ_k f(x_k) )` — transform each item, sum, then process the total. This is why the transform → pool → classify recipe is a *complete and legitimate* way to build any order-blind operation. It's not missing anything.

### Embedding-level beats instance-level (squash first, decide once)
Two possible orderings:

- **Way A — instance-level ("decide per slice, then vote"):** classify each instance to a yes/no first, then combine the yes/no answers. **Problem:** the moment each instance is forced to commit to a single number, all its nuance ("40% suspicious, faint shadow in the corner") is crushed → **information thrown away too early.**
- **Way B — embedding-level ("blend first, decide once"):** keep each instance as its full rich vector, blend the vectors into one bag vector, then make **one** decision at the end. Subtle "maybe" signals from several instances can *add up* and tip the final call.

**Analogy:** Way A = each doctor shouts a bare "sick!" / "fine!" and you count votes. Way B = each doctor writes full notes, all notes are pooled, then one senior doctor reads the combined picture and decides. Way B keeps the detail alive long enough to use it. **The paper argues Way B is better.**

---

## 4. The paper's actual contribution: attention pooling

Prior work used **max** or **mean** pooling for the aggregate step. Both are fixed and non-trainable:
- **max** is rigid.
- **mean** dilutes a rare positive instance among many negatives — a real problem when one tiny tumor patch (or one bad slice) sits in a sea of healthy tissue.

**The fix:** a **trainable weighted average**, where the weights come from a small attention network:

```
z = Σ_k a_k · h_k

           exp{ wᵀ · tanh(V h_kᵀ) }
a_k = ─────────────────────────────────
        Σ_j exp{ wᵀ · tanh(V h_jᵀ) }
```

- The softmax keeps the weights positive and summing to 1.
- It stays **permutation-invariant**, so it's still a valid MIL pooling operator.
- But now the network **learns** how much each instance matters instead of using a fixed rule.

### Gated attention variant
Adds a sigmoid gating branch for extra nonlinearity, letting attention model more complex instance interactions than plain `tanh` (which is roughly linear near zero):

```
a_k ∝ exp{ wᵀ ( tanh(V h_kᵀ) ⊙ sigm(U h_kᵀ) ) }
```

### Why this matters beyond accuracy
The attention weights `a_k` are **directly interpretable**. A high `a_k` = "this instance drove the positive prediction." So the same mechanism that improves pooling hands you a **free, weakly-supervised localization / saliency map** — without ever having instance-level labels.

---

## 5. Experiments (from the paper)

- **Classic MIL benchmarks:** MUSK1/2, FOX, TIGER, ELEPHANT — competitive with or better than prior MIL methods.
- **MNIST-bags:** bags of MNIST digits labeled positive if they contain a target digit (e.g. a '9'); bag size and number of positives are varied. A controlled test of whether the model finds the key instance.
- **Histopathology (breast + colon cancer):** the headline use case — competitive classification plus attention maps highlighting cancerous regions.

---

## 6. Worked mapping: 3D knee MRI (RSNA-style)

**Setup:**
- One knee = a 3D scan = a stack of many 2D slices → **bag**.
- Each slice (or patch of a slice) → **instance**.
- Label at the knee/exam level (e.g. "ACL tear: yes/no", or a severity grade) → **bag label**.
- No per-slice label saying "the tear is visible on *this* slice" → the missing instance-level signal.

This is textbook MIL: label lives at the whole-scan level, you feed in individual slices, nobody told you which slice earned the label.

### Full journey of one bag through the model
1. **Encode each slice** — shared CNN turns each of the 40 slices into a vector.
2. **Score importance (attention)** — a small side-network gives each slice a weight `a_k` via softmax; tear-like slices get high weight, boring slices near zero.
3. **Merge into one bag vector** — weighted average: important slices dominate the sum, 40 vectors collapse to 1.
4. **Predict at knee level** — that bag vector → final classifier → yes/no (or grade). This is what's compared to the label.
5. **Learn** — measure error vs. the true bag label, adjust *everything at once* (slice encoder, attention net, classifier). The error signal flows **backward through the attention weights**, so the model learns *which slices it should have attended to* — purely from getting the bag label right or wrong, never from a per-slice label.

### After training — two outputs
- **Prediction:** feed a new knee → yes/no or grade. (What the competition scores.)
- **Attention map (free bonus):** read out the `a_k` weights → highest-weighted slices are the ones the model leaned on → "the model thinks the tear is on slices 18–21." You come out with an approximate *location* you were never given.

> **Caveat:** attention maps are a *hint*, not ground truth. They show what the model relied on, which usually overlaps with the real pathology but isn't guaranteed to. Treat as interpretability, not verified segmentation.

### Design fork to decide early
- **MIL route** (bag of slices + attention): gives interpretability, lighter on memory.
- **Straight 3D CNN**: eats the whole volume, respects spatial structure natively.

They're two different bets — pick deliberately.

---

## 7. Open thread: does SPATIAL / CROSS-PLANE structure matter?

Plain attention-MIL treats instances as an **unordered, interchangeable set** — it throws away (a) that slice 19 is next to 18 and 20, and (b) that sagittal / coronal / axial planes describe the same physical knee.

### Reframe on the word "proved"
In ML there's rarely a *theorem* proving "spatial structure helps." The available standard of proof is **empirical**: architecture that models the structure beats one that ignores it, on the same data, held out fairly. So the honest phrasing is "strong empirical support," not "mathematically proven."

### Within-plane (slice adjacency in one stack) — WELL SUPPORTED
- 3D CNNs (see slice-to-slice context by construction) generally beat unordered-bag treatments *when the pathology has 3D extent* — a tear/lesion/nodule spans adjacent slices, and continuity helps.
- 2.5D (stack neighboring slices as channels) reliably beats single-slice → indirect evidence adjacency carries signal.
- **Counterpoint:** task-dependent. If the finding is obvious on a single slice, the spatial-context gain shrinks. Benefit scales with how much the pathology's 3D *shape* is diagnostic (for knee ligament/meniscus work, it usually is).

### Cross-plane (sagittal ↔ coronal ↔ axial) — WEAKER, MORE OPEN
- Radiologists use multiple planes because a structure ambiguous in one view is clear in another (ACL best on sagittal, meniscal tears often clearer on coronal). Strong *clinical* prior that planes carry complementary info.
- Multi-view / multi-plane deep models that fuse planes have shown gains — **but most fuse late and loosely** (separate net per plane, then average/concatenate final features).
- **The gap:** they usually do **not** exploit the known *coordinate correspondence* across planes. DICOM headers (`ImagePositionPatient`, `ImageOrientationPatient`) give the exact geometry to say "this voxel in the sagittal stack is the same physical point as that voxel in the coronal stack." Almost nobody threads that into the model.
- **Verdict:** coordinate-registered cross-plane fusion is **largely unexplored, not disproven** — an opening, not a dead end.

---

## 8. Research direction — coordinate-aware cross-plane MIL

Concrete directions, roughly increasing ambition:

1. **Register, then treat as one volume.** Use DICOM geometry to resample all planes into a common coordinate frame, run a single 3D model. Cleanest baseline for "does coordinate info help at all?"
2. **Positional encodings from real coordinates.** In transformer/attention-MIL, tag each slice/patch with its *actual physical position* from the DICOM header (not just its index). Attention can then learn to attend across planes at the same physical location. Most direct way to inject geometry without leaving the MIL framing.
3. **Cross-plane attention keyed on geometry.** Let a query patch in one plane attend to patches in the others, *biased* by physical distance (same physical point → attention prior). The coordinate-aware version of correlated / non-local MIL.

### The experiment that makes it publishable — an ablation ladder
On the same knee data:
- (a) unordered attention-MIL (no spatial info)
- (b) + within-plane ordering
- (c) + late multi-plane fusion
- (d) + coordinate-registered cross-plane fusion

If **(d) > (c) > (b) > (a)** holds on held-out data with proper stats → that's the evidence coordinate-aware spatial modeling matters.

**Do the difficulty-stratified analysis:** likely the cross-plane gain concentrates on hard/ambiguous cases (findings unclear in one plane) — exactly the rare-failure regime worth caring about. Reporting gain *stratified by difficulty* is more convincing than a single aggregate AUC.

### Reality check before the fancy model
Registering planes accurately is where this gets hard: patient motion between acquisitions, different slice thicknesses, resampling artifacts. Many "geometry-aware" ideas die on real data because registration isn't clean enough. **First experiment isn't the model — it's checking how well the planes actually register on the real data.**

---

## 9. Where the paper sits in the wider literature

Attention-MIL made attention the **default MIL pooling operator** — the right starting point. Its main limitation is the independence assumption (no explicit modeling of relationships between instances). Later work set out to fix exactly that:
- **TransMIL / non-local MIL** — transformer-style, models instance correlations.
- **Correlated-MIL variants** — relax the independence assumption.

The spatial / cross-plane thread above is a natural continuation of that line.

---

## TL;DR

- **MIL** = one label per bag, unknown which instance earned it. Weak supervision → saves annotation cost, does **not** beat full supervision or save compute.
- **Framework** = transform (`f`) → order-blind pool → classify (`g`); pool the rich embeddings *before* deciding, not after.
- **Contribution** = replace fixed max/mean pooling with a **trainable attention-weighted average**; weights double as a free interpretability / localization map.
- **Knee MRI** = stack of slices is a bag; attention tells you roughly which slices drove the call (hint, not ground truth).
- **Open gap** = within-plane spatial structure is empirically well-supported; **coordinate-registered cross-plane fusion is largely unexplored** — a real, testable research opening, best proven via a difficulty-stratified ablation ladder (and only after checking registration quality).
