# NTM & RAG — Study Notes

*A revisit-and-refresh guide. Read top to bottom for the story, or jump to any section to brush up one idea. Written from the ground up, with the worked examples we built together.*

---

## The big picture (read this first)

Two papers, one shared spine: **give a neural network access to memory it doesn't have to store inside its own weights.**

- **Neural Turing Machine (NTM), 2014** — bolt an external *notebook* (memory matrix) onto a network, and make reading/writing it learnable.
- **Retrieval-Augmented Generation (RAG), 2020** — same idea grown up: the "notebook" is now 21M Wikipedia passages, and the network learns *what to look up*.

The single trick that makes both work: **you can't hard-pick one memory slot (no gradient to learn from), so you softly blend several.** NTM blends notebook pages; RAG blends passages. Everything below is an elaboration of that one sentence.

---

# PART 1 — Neural Turing Machines (NTM)

## The problem it solves
A normal RNN/LSTM has a **tiny head** — a fixed-size hidden state. Tell it a long list and it forgets the beginning by the end. Its "memory" and its "processing" are the same cramped thing.

## The idea
Give the network a **notebook** (an external memory matrix, N locations × M numbers each). The network — the **controller** — learns to write things down and read them back later. Nobody hand-codes where to read/write; it emerges from training.

## Baby-language ↔ real terms

| Baby version | Actual term |
|---|---|
| Robot brain with a tiny head that forgets | standard RNN / LSTM |
| The notebook it writes in | external memory matrix (N × M) |
| Learns to use it by itself | differentiable end-to-end, trained by gradient descent |
| "Find the page that says *apple*" | content-based addressing (match by meaning) |
| "Go to the next page" | location-based addressing (shift/iterate) |
| Blurry pages vs sharp pages | soft attention (a weighting that sums to 1) |
| Writing = wipe then fill | erase vector + add vector (borrowed from LSTM gates) |

## How memory access works
- **Read** = a weighted blend of all memory rows (soft attention). Not "grab row 5" — instead "80% row 5, 15% row 6, 5% row 4."
- **Write** = *erase* (wipe parts of a location) then *add* (write new content).
- **Addressing = two mechanisms combined:**
  - **Content-based:** emit a *key*, compare to every row by cosine similarity, softmax into a weighting. (Associative lookup, Hopfield-style.)
  - **Location-based:** interpolate with the previous weighting, then a circular convolution *shifts* the focus, then a *sharpening* step keeps it crisp. This is what lets it iterate.

## The tasks (and the headline result)
Copy, repeat copy, associative recall, dynamic N-grams, priority sort.

**The headline finding:** NTM *generalizes past its training range*. Trained to copy sequences ≤20, it still copies length 100+. A plain LSTM falls apart beyond 20. That's the evidence NTM learned an actual **algorithm** (array iteration), not just memorized patterns.

## Why it mattered
Ancestor of all differentiable-memory / attention-augmented architectures (DNC came next; the soft-attention machinery is conceptually upstream of a lot of later work). The recipe: **external addressable memory + differentiable attention = learnable programs that extrapolate.**

---

# PART 2 — the concepts you asked me to unpack

These came up as questions along the way. They're the load-bearing ideas — worth re-reading whenever they feel fuzzy.

## "Differentiable" — what does it actually mean?
**Smooth enough that a tiny nudge to any knob produces a measurable, small change in the output.** That smoothness is what gradient descent needs to learn ("which way do I turn this knob to reduce error?").

- **Not differentiable = a light switch.** Off → flip → fully on. Nothing in between to feel your way along. "Read cell #5" is like this: nudge the address toward 6 and nothing happens, then suddenly you're reading a totally different cell. No gradient.
- **Differentiable = a dimmer knob.** Turn it a little, light rises a little. You can always feel which way to go.

**The NTM trick:** replace every sharp computer operation ("pick cell 5") with a dimmer version ("read a weighted blend of all cells"). That blur is the cost; the *sharpening* step exists to push the blur back toward "mostly one cell" once it has learned where to look.

## "Stores everything it knows in its weights" — parametric memory
A network is a giant pile of numbers (**weights** — BART has ~400M). Training nudges those numbers until predictions match reality. Facts end up **smeared across the weights**, not stored as readable entries.

- To predict the next word across millions of sentences like "Paris is the capital of France," the weights arranged themselves so "capital of France is ___" → "Paris." Nobody wrote a `France → Paris` table. The fact is emergent.
- Analogy to your world: your EfficientNet doesn't store a lookup table of "these pixels = fake." That knowledge is distributed across conv-filter weights. Same thing, but facts instead of image artifacts.

**Why this causes RAG's three complaints** (all consequences of knowledge-as-weights):
1. **Can't edit** one fact without retraining.
2. **Can't inspect** — no entry to point at → no provenance.
3. **Hallucination** — facts are blended, so the machinery can smoothly produce something that *sounds* right but was never true (interpolating in weight-space, not retrieving a record).

---

# PART 3 — Retrieval-Augmented Generation (RAG)

## The core problem
A pretrained LM (BART/T5) keeps all its knowledge in its weights = **parametric memory**. Fix the three problems above by adding a second, different kind of memory.

## The two memories (the single most important axis)

| | Parametric memory | Non-parametric memory |
|---|---|---|
| What it is | BART's weights (~400M numbers) | 21M real Wikipedia passages (actual text) |
| Form | knowledge smeared into numbers | knowledge as readable words |
| Editable? | no (retrain) | **yes — swap the text** |
| Inspectable? | no | **yes — point to the passage** |
| Role | fluency, general reasoning | specific, updatable facts |

**RAG is a HYBRID of both.** (This resolved your big confusion — see the library metaphor next.)

## The library metaphor (the mental model to keep)
You worried: *"if RAG is all numbers/vectors, where does the text come from?"* The fix:

- **The books** = the real Wikipedia passages (the knowledge, as text).
- **The catalog cards** = the fingerprint **vectors**. A card doesn't contain the book — it just helps you *locate* it fast.

**The vectors are NOT the knowledge. They're the address system.** You use vectors to find *which* passages to grab, then you fetch the **actual text** of those passages. The vector was never the answer — it was the pointer to the answer.

This is exactly why hot-swapping works: because the library is real editable text, you throw out the 2016 shelf and put in the 2018 shelf — zero retraining. (Impossible with knowledge baked into weights.)

## The two components

**Retriever — DPR (the finder).** A BERT bi-encoder = two separate BERT towers:
- one makes the **question** fingerprint `q(x)`
- one makes each **passage** fingerprint `d(z)` (done once, in advance, for all 21M)
- relevance = **dot product** `d(z)ᵀq(x)` (big = pointing the same way = related)
- "find the biggest dot products fast over 21M" = **MIPS**, solved approximately with **FAISS + HNSW** (takes shortcuts instead of checking all 21M)

**Generator — BART (the writer).** A seq2seq transformer (~400M). To use a passage, it just **concatenates** (glues) the passage onto your question and writes an answer. No fancy fusion.

**Pipeline:** `question → BERT fingerprint → MIPS search → top-K passages → concat each with question → BART writes`.

---

# PART 4 — the clever bit: retrieval as a latent variable

## The puzzle
Nobody ever labels *which passage is correct*. So how does the finder learn what to find?

## The trick (the "smoothie")
Don't bet on one passage. Grab the **top K**, give each a **trust score** `p_η(z|x)`, let the writer answer using each, and **blend the answers weighted by trust:**

```
p(y|x) ≈ Σ_z  p_η(z|x) · p_θ(y|x,z)
         └ trust ┘   └ writer's confidence ┘
```

Smoothie mapping:
- "pour more of the fruit you trust" = **p_η(z|x)** (the weight)
- "how good that fruit tastes" = **p_θ(y|x,z)** (writer's confidence)
- "the finished smoothie" = **p(y|x)** (final blended answer)

## Why this lets the finder learn
The trust score sits *inside* the final answer. So when the answer is graded, the trust scores get graded too — pushed **up** if they helped, **down** if they hurt. The finder learns purely from "did this help produce the right answer." **No passage labels needed.**

## The link to NTM (the mental hook)
- NTM: can't hard-pick page 5 → blurry blend of **notebook pages**
- RAG: can't hard-pick passage 5 → blurry blend of **Wikipedia passages**

Same trick, bigger scale. Don't hard-pick; softly blend so the system can *feel* which way to improve.

## RAG-Sequence vs RAG-Token (the distinction people forget)
Both blend over documents; the question is *how often you can switch documents*:
- **RAG-Sequence:** one document generates the *whole* answer. (`Σ` outside the `Π`.) Best for short factual QA.
- **RAG-Token:** *every token* can come from a different document. (`Σ` inside the `Π`.) Best for multi-fact/"braided" answers (e.g. Jeopardy clues combining two facts).

---

# PART 5 — one full worked example, end to end

**Question:** "Who wrote *A Farewell to Arms*?" **True answer:** Hemingway.

The crucial split you kept asking about:
- **Phase A = answering** (happens live, needs NO correct answer)
- **Phase B = training** (needs the known answer; this is where "trust gets graded")

## PHASE A — answering

1. **You type:** `x = "Who wrote A Farewell to Arms?"` (just text — can't do math on words yet)
2. **BERT → fingerprint:** `q(x) = [0.12, -0.40, 0.90, ...]` (GPS coordinates for *meaning*)
3. **Library already fingerprinted (in advance):**
   ```
   #418 : [0.11,-0.40,0.90]  "...Hemingway's novel A Farewell to Arms..."
   #77  : [0.90, 0.20,-0.30] "...The Sun Also Rises, 1926..."
   #2901: [-0.50,0.60, 0.10] "...World War I killed millions..."
   ```
   (fingerprint = address; passage text = content; stored together)
4. **Compare (dot product):** `#418 → 2.94` (close!), `#77 → 0.05`, `#2901 → -0.31`. (This is the MIPS/FAISS search.)
5. **Keep top 3:** z₁=#418, z₂=#77, z₃=#2901.
6. **Turn closeness into trust** (softmax → sums to 1): `trust z₁=0.60, z₂=0.30, z₃=0.10`.
   **← This answers "who evaluates trust": nobody. It's just fingerprint-closeness turned into percentages by math.**
7. **BART answers per passage** (see next section for where 0.95 comes from):
   ```
   read (z₁ + question) → "Hemingway"   confidence p_θ = 0.95   (passage names him)
   read (z₂ + question) → "Hemingway?"  confidence p_θ = 0.20   (wrong book, right era)
   read (z₃ + question) → "?"           confidence p_θ = 0.02   (useless passage)
   ```
8. **Blend by trust (the smoothie):**
   ```
   (0.60 × 0.95) + (0.30 × 0.20) + (0.10 × 0.02)
   =  0.570      +  0.060        +  0.002        = 0.632
   ```
   **Output: "Hemingway."** Done. Notice we never needed the true answer here.

## PHASE B — training (where trust is graded)

1. **Dataset gives us the KNOWN answer:** `y = "Hemingway"`. (Only the *answer* is labeled — never the passage.)
2. **Grade via the loss function:** ideal = 1.0, model got 0.632, shortfall = 0.368. **This is the "evaluator" — a formula comparing to the answer key. Not a user. Not a human.**
3. **Backprop flows blame backward:**
   - z₁ (trust 0.60) produced a great slice → *raise* its trust: `0.60 → 0.68` ⬆️
   - z₃ (trust 0.10) produced garbage → *lower* its trust: `0.10 → 0.04` ⬇️
   - Nobody told it z₁ was correct — the **known answer** made z₁ look good in hindsight, and the math rewarded it. The label was on the *answer*; the reward *leaked backward* onto the passage's trust. **That leak is the whole trick.**
   - *How* trust rises: go back to **BERT** and nudge its knobs so next time the question's fingerprint points *even closer* to #418's → higher dot product → higher trust.
4. **Repeat millions of times.** BERT learns fingerprints that pull genuinely-useful passages closer; BART learns to write better answers. After training, trust scores are already good — so Phase A works with no answer key.

---

# PART 6 — where BART's 0.95 comes from

BART never writes a whole word at once. It's **super-powered autocomplete**: after reading the input, it assigns a probability to *every word in its ~50,000-word vocabulary* for the next slot.

```
next word = "Hemingway"  → 0.95   ← 95% of its bet
next word = "Fitzgerald" → 0.02
next word = "Ernest"     → 0.01
...everything else...    → tiny leftover
                            = 1.00  (sums to 100%)
```

- **The 0.95 is BART's own next-word bet on "Hemingway."** Nobody grades it — it's BART's output.
- **It depends on the passage BART read:** z₁ names Hemingway → easy copy → 0.95. z₂ (wrong book) → guessing → 0.20. z₃ (useless) → bets spread thin → 0.02. **Passage quality shows up directly as the size of the number.**
- **Mechanically:** BART computes a raw score per vocabulary word, then **softmax** squishes them into percentages summing to 1.0. "Hemingway" had the highest raw score → biggest share → 0.95. (Same softmax that turned fingerprint-closeness into trust in Phase A Step 6.)

This 0.95 **is** `p_θ(y|x,z)`: *"probability BART assigns to answer 'Hemingway', given the question and this specific passage."*

---

# PART 7 — RAG results & limitations (worth knowing)

**Results that matter:**
- Beats T5-11B on NaturalQuestions (**44.5 vs 34.5 EM**) with **626M trainable params vs 11B** — ~18× smaller. Offloading knowledge to external memory buys huge parameter efficiency.
- Beats DPR *without* a re-ranker or extractive reader — pure generation beats extract-a-span.
- Killer stat: on NQ, RAG is correct **11.8% of the time even when the answer is in none of the retrieved passages** — an extractive model scores 0% there. RAG *synthesizes* (clues + parametric knowledge).
- Human eval (Jeopardy): RAG judged more factual in **42.7%** of cases vs BART's **7.1%**.

**Training shortcut that made it practical:** freeze the document encoder + index; only fine-tune the query encoder and BART. (REALM re-indexed all of Wikipedia during training — brutally expensive. RAG showed you don't need to.)

**The two demos proving the two-memory split is real:**
- **Index hot-swapping:** 2016 index → gets 2016 leaders right; swap to 2018 index → gets 2018 leaders right. Update world knowledge by swapping a file, zero retraining.
- **Retrieval ablations:** freezing the retriever hurts every task → learned retrieval genuinely helps. (BM25 word-overlap only wins on FEVER, which is entity-heavy.)

**Limitations:**
- **Retrieval collapse (Appendix H):** on tasks that don't *force* factual lookup (e.g. story generation), the retriever can collapse to returning the same passages regardless of input; the generator learns to ignore them, and RAG silently degrades to plain BART. **Watch for this in your own RAG builds — if retriever outputs stop varying with the query, it's collapsed.**
- Knowledge capped by Wikipedia (+ its biases); document encoder is frozen so it can't adapt document representations; the two memories aren't jointly pre-trained from scratch.

---

# One-page cheat sheet

- **NTM** = network + learnable external notebook. Read = blurry blend of rows; write = erase+add; address by content (match) and location (shift). Proof it learned an *algorithm*: generalizes past training length.
- **Differentiable** = dimmer knob, not light switch. Needed so gradient descent can feel which way to nudge. Blur is the price of learnability.
- **Parametric memory** = knowledge smeared into weights → can't edit/inspect, hallucinates.
- **Non-parametric memory** = real text you can point to, edit, swap.
- **RAG** = hybrid of both. Vectors = the *catalog* (find), passages = the *books* (know).
- **Latent-variable trick** = grab top-K, weight by trust `p_η(z|x)`, blend by writer confidence `p_θ(y|x,z)`; grade only the *answer* and let blame leak back to trust. No passage labels.
- **Phase A (answering):** no correct answer needed. **Phase B (training):** loss function compares blended output to the *known answer* — that's the only "evaluator," and it's pure math, never a user.
- **BART's confidence** = its own softmax next-word bet; high when the passage contains the answer.
- **Formula:** `p(y|x) ≈ Σ_z p_η(z|x) · p_θ(y|x,z)`.
