# SteerViT — Full Discussion Recap

*Paper: "Steerable Visual Representations" (SteerViT), arXiv 2604.02327, Apr 2026.*
*Ruthardt, Gaur, Ramanan, Tapaswi, Asano — UTN / CMU / IIIT-Hyderabad.*

This doc recaps everything we talked through, each topic given twice: once in **baby
language** (plain, no jargon) and once **technical** (precise).

---

## 1. What the paper is

**Baby:** Imagine a super-smart friend who describes photos, but he only ever talks
about the *biggest, most obvious thing* — show him a cat on a couch and he yells
"cat!", ignoring the little TV remote you actually cared about. SteerViT is a trick
to whisper in his ear "look at the remote" — and now he does. You don't rebuild the
friend, you just add a tiny helper that lets words steer his attention.

**Technical:** Pretrained ViTs (DINOv2, MAE) produce rich but *query-agnostic*
features that collapse to the most salient object (photographer bias / object-centric
data). SteerViT makes a **frozen** ViT steerable with natural language by injecting
text **into** the encoder via early fusion, so the prompt shapes the visual encoding
process itself rather than being fused only at the output. It adds ~21M trainable
params and preserves the base ViT's representation quality (a Pareto improvement).

---

## 2. Who actually "makes the words"? (the clarification)

This was the confusion to untangle. In standard setups, **nobody generates words from
the image — YOU provide them.** Three flavors:

**Flavor 1 — plain frozen ViT (DINOv2):**
- *Baby:* The eyes look and produce a *number-summary* of the picture. No words at all,
  anywhere. The meaning is baked into numbers by the model itself.
- *Technical:* ViT maps image → patch/CLS embeddings. No text branch. Semantics are
  implicit in the feature vector; fully query-agnostic.

**Flavor 2 — CLIP / SigLIP (late fusion):**
- *Baby:* You type the words. A separate text-reader turns your words into numbers.
  Now there are two number-summaries — one from the image, one from your words — and
  at the very end they're just *compared*. The image side never saw your words while
  looking. That's the weakness.
- *Technical:* Two independent towers into a shared space. Visual features are
  extracted independently of the query; post-hoc combination (e.g. element-wise add)
  gives ~0.02% benefit → not steerable.

**Flavor 3 — MLLMs (vision-LLMs):**
- *Baby:* Here words really do get written at the end — the model turns the image into
  tokens, hands them to a big language model, and it writes sentences. But it's
  expensive and fine visual detail gets fuzzy.
- *Technical:* ViT tokens fed to an LLM; fusion inside the LLM's layers. Representations
  become language-dominant, lose visual fidelity, need ≥1B params.

**SteerViT's change:** slip the (human-provided) words into the ViT's internals *while
it's still encoding the image*, so the instruction bends what the eyes focus on instead
of arriving too late.

---

## 3. "But in real life the model works first, then the question comes… right?"

The key mental-model flip.

**Baby:** You pictured it as *look first, ask later.* But in most real jobs the
**question comes first** — it's the whole reason you're looking. Examples:
- Searching 50,000 warehouse photos for "fire extinguisher" → you type it *first*,
  then look through all images with that in mind.
- Factory line checking for "scratch on metal casing" → set up *before* any product
  rolls by.
- Retinal screening → you're looking *for* microaneurysms/hemorrhages; the clinical
  question exists *before* analysis.

And even in a chat where the question does come after, the model doesn't edit an old
look — **it just looks again.** Re-examining an image is cheap, so it re-runs the ViT
with the new word whispered in from the start.

**Technical:** Encoding is conditioned at inference on the query prompt. In retrieval/
detection/screening pipelines the prompt is fixed *before* the pass over the data, so
conditioning the encoder up-front is natural, not backwards. For post-hoc queries, a
fresh conditioned forward pass is cheap (frozen backbone), so statelessness is a
non-issue. "Analyze-then-question" → really "question-then-analyze."

---

## 4. Overall purpose of SteerViT

**One sentence:** Let you *tell a frozen vision model what to pay attention to*, using
words, without retraining it and without breaking what it was already good at.

**Baby — why it's worth building:** Before this, if your great vision model kept
staring at the wrong thing, every fix was bad: retrain it (slow, and you'd need a
separate model per task), switch to a giant expensive MLLM (fuzzy on detail), or just
live with it. SteerViT kills that trade-off: one frozen model + tiny add-on + words =
redirect its attention to anything, on the fly.

**What it buys you:**
1. **Flexibility without cost** — one model, many "what am I looking for" jobs via
   different prompts. (Beat 100 separately fine-tuned models with one prompted model.)
2. **Reach small/hidden things** — point it at the non-salient object, faint defect,
   background item. That's the core capability.
3. **Adapt to brand-new situations free** — words describe anything, so it transfers
   zero-shot to unseen domains (factory defects, medical anomalies).
4. **Keep original strengths** — you add a steering wheel, not a new engine; the ViT
   stays smart at classification/segmentation.

**Technical / bigger claim:** Rather than scaling MLLMs that fuse language *on top of*
vision, condition a strong frozen vision encoder on language *from the inside* —
cheap, flexible, reversible. SteerViT is the proof-of-concept for that paradigm.

---

## 5. Could this be the foundation for a visual "episodic memory" model?

(e.g. "where did I leave my keys," recalled through a camera/lens.)

**Baby:** Yes — for the *recall* part. The things you forget the location of are never
the big obvious object (you never lose the couch); they're small backgroundy things —
keys, scissors, passport. Normal encoders are blind to those; SteerViT is specifically
good at finding them by words. **But** SteerViT has *no memory at all* — it looks at one
image now and forgets instantly. To get real "where are my keys" memory you also need:
1. **Storage** — actually save what it saw over time.
2. **Time (when)** — "I saw X at this time."
3. **World location (not photo location)** — *where in the room*, a persistent map
   across many photos/angles — not just "which patch of one picture."
4. **Update-on-change** — keys move; overwrite "on the hook" → "on the desk."

So SteerViT is the **indexing/front-end box**, not the whole memory system.

**Technical:** SteerViT is a stateless, language-queryable perception front-end (its
CORE task = "given words, find the small non-salient object" = exactly the recall
muscle). Episodic memory needs machinery wrapped around it:
`camera stream → SteerViT-style encoder (index each frame by language) →
spatial+temporal memory store (where + when) → language query → return location`.
This is an active area (see Ego4D's Episodic Memory benchmark: "where did I last see
object X"). Framing: SteerViT is a *better language-conditioned retrieval front-end*
than what those systems currently use — a component upgrade, not a new paradigm. The
memory, spatial grounding, and temporal indexing are the open research.

**Your edge (rare-event / trustworthiness thread):** the sharp problem isn't building
the memory — it's the *failure mode*. When the system says "keys are on the desk" and
they're not, how do you know it's wrong? Query-conditioned encoders can hallucinate a
confident match. Reference-free "is this recalled location trustworthy" detection sits
right in the thread you've already been pulling on.

---

## 6. The plan we sketched

**Phase 1 (do first, alone): minimal from-scratch SteerViT repro.**
- ~4–5k images, small proof-of-mechanism. NOT the full paper, NO memory yet.
- Goal = prove the *mechanism* works, not match their 96% numbers.
- **Most important eval = the wrong-prompt sanity check:** condition on a random
  expression; localization must collapse. If it doesn't, the model memorized instead
  of steering → failed repro. That one test beats any accuracy number.
- A detailed Claude Code prompt for this was written out separately (DINOv2 ViT-B/14
  frozen + roberta-base frozen + 2-layer MLP adapter + 6 gated cross-attn layers +
  patch-level referential-segmentation loss on RefCOCOg).
- Watch out: the RefCOCOg/COCO mask → 24×24 patch-grid plumbing usually breaks these
  repros (resolution / off-by-one). Visualize the patchified target before trusting it.

**Phase 2 (only after Phase 1 validates): memory.**
- Storage, timestamp, spatial/world anchor, update-on-change.
- Build the trustworthiness/hallucination-detection angle in from the start.

---

## 7. The architectures in the paper

The paper (Fig. 3 taxonomy, Table 1 scorecard) contrasts four existing families +
its own, organized by *where* language meets vision.

| Family | Steerable | Features | Fusion | Trainable MM params |
|---|---|---|---|---|
| Unimodal ViT (DINOv2, MAE) | ✗ | ✓ | none | 0 |
| Cross-modal (CLIP, SigLIP) | ✗ | ✓ | late | ~200M |
| OV Localize (SAM3, GroundingDINO) | ✓ | ✗ | late | ~200M–1B |
| MLLM (InternVL3, Qwen3-VL) | ◐ | ◐ | late, in LLM | ≥1B |
| **SteerViT (theirs)** | ✓ | ✓ | **early, in ViT** | **21M** |

**Baby version of the four rivals:**
- **Unimodal ViT** — eyes only, no words, stares at the obvious thing.
- **CLIP** — eyes and ears work separately, only compared at the end.
- **OV localizers** — good at "find the named thing," but their inner features are too
  specialized to reuse for other tasks.
- **MLLMs** — bolt a huge language brain on top; smart but heavy and fuzzy on detail.
- **SteerViT** — whisper words *into* the eyes while they look; tiny add-on, nothing
  broken.

### SteerViT components (only the adapter trains)

**A. Visual encoder (FROZEN).** Pretrained ViT, mainly DINOv2 ViT-B/14 (also SigLIP,
MAE). Image `X_v ∈ ℝ^(H×W×3)` → N patch tokens `Z_v ∈ ℝ^(N×d_v)` (+ optional CLS).
All original params frozen; new capacity only from injected layers.

**B. Text encoder (FROZEN).** RoBERTa-Large → token embeddings `Z_t ∈ ℝ^(L×d_t)`.

**C. Multimodal adapter (TRAINABLE).** ℓ2-normalize each text token, then a 2-layer
MLP projects text into visual space → `H_t ∈ ℝ^(L×d_v)`.

**D. Gated cross-attention (TRAINABLE).** Inserted into **every other** ViT block
(6 layers for ViT-B's 12). Vision patches = **queries**; adapted text `H_t` = **keys/
values**:

```
Ẑ_v^(ℓ) = softmax(QKᵀ / √d_k) · V,   Q = Z_v^(ℓ) W_Q,  K = H_t W_K,  V = H_t W_V
```

Integrated via a **tanh gate** with per-layer learnable scalar `α_ℓ`, **init 0**:

```
Z_v^(ℓ+1) = Z_v^(ℓ) + tanh(α_ℓ) · Ẑ_v^(ℓ)
```

`tanh(0)=0` → identical to frozen ViT at init; `sech²(0)=1` → gate still gets gradient,
so `α_ℓ` opens the text pathway during training. This is Flamingo's gated cross-attn
**inverted** (Flamingo: language→vision; SteerViT: vision-attends-to-language).
Total ~21M trainable params.

**Training-time head (proxy task).** A linear segmentation head maps each patch token →
softmax over the patch grid; trained with **soft cross-entropy** on a **patch-level
referential-segmentation** target (fraction of foreground pixels per patch). This is
what forces the cross-attention to route text into the correct patches.

### Architectural choices that mattered (ablations)
- **Drop the Flamingo FFN** after cross-attn — barely helps quality, hurts
  steerability/OOD, and would inflate params 21.2M → 35.4M (+67%).
- **Early fusion matters for fine-grained** — late-fusion variant matches on coarse
  retrieval but collapses on PODS (36.6 vs 58.1).
- **MLP adapter > single linear projector** (better modality alignment).
- **tanh gate is load-bearing** — ungated cross-attn disrupts the frozen features and
  drops every metric.
- **Generalizes across backbones** — biggest gains on weaker ones (MAE steerability
  +33.9 from early fusion).

### The inference "control knob"
Scale all gates by `ω ∈ [0,1]` at inference to interpolate between vanilla ViT (`ω=0`,
identical to frozen backbone) and full steering. **`ω ≈ 0.6`** is the Pareto sweet spot
for DINOv2/SigLIP (slightly *exceeds* base ViT quality while unlocking steerability).

---

## Headline numbers (for reference)
- CORE conditional retrieval: DINOv2 44% → **SteerViT 96%** (FLAIR 81.3%).
- GeneCIS (real images, zero-shot): 25.4 R@1 vs 9.6 DINOv2, 18.7 specialized.
- MOSAIC targeted attention: PR-AUC 14.3% → **50.2%**.
- PODS personalization: 58.1% PR-AUC with detailed prompts, beats fine-tuned DINOv2
  (48.0%) — one model vs 100 per-object models.
- Zero-shot anomaly seg (MVTec AD): 82.1 PRO, ~matches dedicated FADE (84.5).
