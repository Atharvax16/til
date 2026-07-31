# MovieChat: From Dense Token to Sparse Memory for Long Video Understanding

*A working explainer — paper walkthrough, plain-language version, and the encoder/frames clarification.*

Paper: arXiv 2307.16449

---

## 1. The core problem

Video-LLMs at the time (VideoChat, Video-LLaMA, Video-ChatGPT) could only ingest a handful of frames — roughly **32 to 100** — before running out of VRAM. Each frame becomes a batch of visual tokens, and stuffing thousands of frames' worth of tokens into a transformer is **quadratically expensive** in attention and **linearly brutal** in memory. So a two-hour movie was simply out of reach.

Three challenges block long video:

- **Computation complexity** — attention cost blows up with token count.
- **Memory cost** — storing every frame's tokens exhausts VRAM.
- **Long-term temporal connection** — keeping coherence across events separated by thousands of frames.

**Headline efficiency claim:** previous methods cap out around 100 frames, while MovieChat handles **>10K frames on a 24GB GPU** — roughly a **10,000× improvement** in per-frame VRAM growth (from ~200MB/frame down to ~21KB/frame).

---

## 2. The memory mechanism (the actual contribution)

They borrow the **Atkinson-Shiffrin model** from cognitive psychology: short-term memory acts as a buffer that consolidates into long-term memory. MovieChat gives a video AI the same two-tier setup so it can "remember" a very long video without choking.

### Feature extraction

Rather than a video foundation model (ViViT, Video-Swin), they use a plain **image** encoder frame-by-frame — **ViT-G/14 from EVA-CLIP** plus the **BLIP-2 Q-former**. Their argument: image encoders align better with text, and their memory mechanism handles the temporal side. Frames are processed in a **sliding window of 16 frames per slide**.

### Short-term memory (the holding tray)

A fixed-length **FIFO buffer** holding dense, unprocessed frame tokens (**18 frames × 32 tokens each** in their config). It works like a queue: when a new frame arrives and the tray is full, the oldest frame is pushed out the other end for consolidation. Nothing is compressed here yet — these frames are stored in full detail.

### Long-term memory (the clever bit)

This is where dense tokens become **"sparse memory."** The frames pushed out of the short-term tray don't get thrown away — they get **squeezed down** before being filed into long-term storage.

The insight: in video, consecutive frames usually look almost identical (a person just standing there talking = 30 frames of basically the same picture). So why store all 30? The algorithm (borrowed from **ToMe / Token Merging**):

1. Compute average **cosine similarity** between adjacent frames' tokens.
2. Greedily **merge the most-similar pair** via weighted averaging.
3. Repeat until frame count drops to a target `R_L`.

A long, boring stretch collapses into just a few representative frames, while genuinely different moments survive. That's what **"dense tokens → sparse memory"** means.

Two nice properties:

- **Parameter-free** — no training, just a similarity-and-averaging rule.
- **Pluggable** — bolts onto any frame-based encoder.

Long-term memory holds **256 frames** total in their setup.

### The positional-encoding fix

Models normally track frame *order* using a numbering scheme, but that scheme runs out of numbers with this many frames. They borrow a smarter, hierarchically decomposed method (BERT-style) that stretches the available slots from length **`n` up to `n²`**, so every frame in the long buffer still gets a proper position label.

### One-sentence version

> Keep recent frames in full detail, and for older frames, merge the near-identical ones together so the AI stores a compact summary instead of thousands of repetitive frames.

---

## 3. Two inference modes

- **Global mode** — questions about the whole video, using **only long-term memory** as the video representation.
- **Breakpoint mode** — questions about a *specific moment* `t`. Here they concatenate **long-term memory + short-term memory + the current frame feature `x_t`**, since events have continuity and you need both nearby detail and distant context. Simple concatenation worked well.

The video representation then passes through the Q-former and a projection layer into the LLM (LLaMA/Vicuna family) for the actual Q&A.

---

## 4. MovieChat-1K benchmark

Because no long-video-QA benchmark existed, they built one:

- **1K video clips** from movies/TV across **15 categories**.
- **~13–14K manual annotations**.
- Each video gets 1 dense caption, 3 global QA pairs, and 10 breakpoint QA pairs with timestamps.
- Average duration **9.4 min** (~10K+ frames); over 90% of clips are 10K–12K frames.
- Split 800 / 100 / 100 train / test / val.

---

## 5. Results

- **Short video QA** (MSVD, MSRVTT, ActivityNet): competitive-to-best even though it isn't designed for short clips.
- **Long video QA**: reads **2048 frames** vs. baselines' 32–100, and wins in both global (62.3 acc) and breakpoint (48.3 acc) modes. Baselines had to be fed uniformly sub-sampled frames just to fit.
- **Evaluation** used GPT-3.5, Claude, and human blind rating with strong Pearson agreement (0.92–0.98). They added **manual filtering** because the LLM judges sometimes returned contradictory "yes + score 0" verdicts — a real methodological wrinkle worth remembering for LLM-assisted eval.
- **Ablations** confirm the memory mechanism matters a lot (removing it drops global accuracy from ~68 to ~51), and merged-token initialization of short-term memory beats last-few-tokens or uniform sampling.

---

## 6. Limitations they admit

1. **Limited perception** — bottlenecked by the pretrained short-video model.
2. **Coarse time processing** — only gives rough duration proportions, not precise temporal localization.

---

## 7. Clarification: the encoder, patches, frames, and the two "16"s

A common confusion: there are **two different "16"s** floating around, and they're unrelated. The trick is to separate three levels.

### Three separate levels

**Level 1 — pixels/patches (inside ONE frame).** This is the ViT-G/**14**. The "14" is the *patch size*: the image encoder chops a single frame into 14×14-pixel squares. A 224×224 frame becomes 224/14 = 16 patches across and 16 down → **256 patches**. This is about pixels within one image, not about frames. (The "16 patches per side" here is a coincidence — same number, totally different meaning from the frame window.)

**Level 2 — one frame → tokens.** Each frame goes through the ViT *and then a Q-former* (from BLIP-2). The Q-former compresses: it takes those ~256 patch embeddings and squeezes them down to a fixed **32 tokens per frame**. That's why the config says "32 tokens per frame."

**Level 3 — many frames (the sliding window of 16).** This is the *other* 16. It just means: as MovieChat marches through the video, it scoops up **16 whole frames per step**. Each of those 16 frames is still run through the image encoder **one at a time, independently**.

### Resolving the "contradiction"

There's no contradiction between "one frame at a time" and "16 frames per window." The window is just *how many frames get grabbed per step* — like grabbing 16 photos from a stack. Each photo is still fed into the image model **by itself**. The encoder never sees 16 frames together.

### Where does "things change over time" come from?

The image encoder is **deliberately blind to time.** It has no idea frame #5 comes after frame #4 — it just turns each frame into 32 tokens in isolation. So the temporal understanding comes **entirely from the memory mechanism, not the encoder.** That's the whole architectural bet:

- The **image encoder** handles *space* (what's in this one frame).
- The **memory mechanism** handles *time* (which frames are redundant, which are new, what to keep vs. merge).

That's exactly why they could get away with a plain image model instead of a video model. When things genuinely change, the consolidation step sees **low similarity** between neighboring frames and keeps them separate; when nothing changes, it merges them.

### The full chain for one step

```
16 frames grabbed (sliding window)
   │
   └─ for EACH frame, independently:
          frame (224×224 pixels)
            → ViT-G/14 cuts into 14×14 patches → 256 patches
            → Q-former compresses → 32 tokens
   │
   → 16 frames × 32 tokens land in short-term memory (holds 18 frames)
   │
   → when full, oldest frames pop out → similarity-merge → long-term memory
```

So: **patches** live inside one frame (spatial), **frames** get encoded solo, the **window of 16** is just the scoop size, and **time** is handled downstream by memory — never by the encoder.

---

## 8. Config quick reference

| Component | Value |
|---|---|
| Image encoder | ViT-G/14 (EVA-CLIP) + BLIP-2 Q-former |
| Patch size | 14×14 px → 256 patches per 224×224 frame |
| Tokens per frame | 32 (after Q-former) |
| Sliding window | 16 frames per slide |
| Short-term memory | 18 frames × 32 tokens (FIFO) |
| Long-term memory | 256 frames |
| Consolidation length | 2 |
| Merge rule | Greedy adjacent-frame cosine-similarity merge (ToMe), weighted average |
| Positional encoding | Hierarchically decomposed, extends `n → n²` |
| Inference modes | Global (long-term only) / Breakpoint (long + short + current frame) |
