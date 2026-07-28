# VoxSight Recall — problem statement

*The deliverable the literature roadmap (§12) asks for after the reading is done: an event schema,
a safe-answer policy, a first benchmark task, and one hypothesis to test.*

---

## 1. The problem

A wearable visual assistant sees an object, stores an observation, and later is asked *"where are my
keys?"* Every published system in this lineage — Ego4D's episodic-memory benchmark, MemPal, and the
long-video memory models — answers a question about the **past**: *where was the object last seen?*
The user asked a question about the **present**: *where is it now?*

Those are different questions, and the gap between them is not a detail. Between the last sighting
and the query, the object may have been picked up, moved, or consumed. A system that reports the
last sighting as if it were the current location is not merely imprecise — it is confidently wrong,
and for a blind or low-vision user acting on that answer, a confident wrong answer is worse than an
admitted "I don't know."

The metric everyone reports — Recall@K, temporal IoU, rank of the ground-truth moment — cannot see
this failure. Those metrics ask *"did you find the right past moment?"* and the answer is often yes
at exactly the moment the system is about to mislead the user. This is the same defect written up
in `../MemGPT/memgpt-study-notes.md` §9: a metric that looks healthy one level down while the thing
that matters at the next level up is never measured.

## 2. Event schema

The retrievable unit is a **visual event record**, not a text chunk (roadmap §3.2). Fields marked ★
are the ones this study actually populates; the rest are part of the schema but out of scope here.

| Field | Example | Purpose |
|---|---|---|
| ★ `event_id` | `P04_113_e0271` | Auditable unit of retrieval |
| ★ `time` | frame 12480 (≈ 04:09.6) | Lets the assistant say *when* it observed this |
| ★ `kind` | `PUT` / `TAKE` | Placement vs removal — the state transition |
| ★ `object` | `knife` | Retrieval target |
| ★ `receptacle` | `drawer` | The location cue the answer is built from |
| ★ `evidence` | frame range + feature vector | Provenance; what the claim rests on |
| ★ `confidence` | calibrated score | Prevents a confident but unsupported answer |
| `place` | `kitchen` | Coarse room context (single-room in EPIC, so constant) |
| `privacy` | retention window, deletion flag | Makes the memory user-controllable |

A **PUT with an explicit receptacle** is the only event that can ground a location answer. A **TAKE**
grounds nothing but *invalidates* the standing answer — it is negative evidence, and the claim of
this study is that negative evidence is the part everyone drops.

## 3. Safe-answer policy

Given query *"where is X?"* at time T, let `last_put` be the most recent receptacle-bearing PUT of X.

- **No TAKE of X since `last_put`** → *"Your X is on the counter. I last saw it there at 10:18."*
- **A TAKE of X since `last_put`** → *"I last saw your X on the counter at 10:18, but it was picked
  up after that, so I can't tell you where it is now."*
- **X placed since, receptacle unobserved** → *"I saw you put your X down at 10:24 but I couldn't see
  where. The last place I'm sure about is the counter, at 10:18."*
- **No PUT of X on record** → *"I have no observation of your X."*

The unsafe answer, and the one a last-occurrence system gives in all four cases, is *"Your X is on
the counter."*

## 4. Benchmark task

**Dataset.** EPIC-KITCHENS-100, 11 videos across 9 participants (≈4 h egocentric kitchen video).
Narration annotations supply object state transitions for free: `put`/`insert` verb classes carry
object *and* receptacle in `all_nouns`; `take`/`remove`/`move` mark removal; every row has exact
frame bounds. Splits are **by participant**, so no kitchen appears in two splits.

**Query generation.** After each state event, emit one query per object whose location has ever been
asserted. Ground truth is the object's state at that instant:

| GT state | Meaning |
|---|---|
| `AT(r)` | Last PUT put it at `r`, nothing has removed it since |
| `GONE` | Taken/removed/moved since the last PUT — not at the last-seen receptacle |
| `PLACED_UNKNOWN` | Put down since, but the receptacle was not narrated |

**Evaluation levels** (roadmap §8.5), levels 1–3 in scope:

1. *Event retrieval* — Recall@1/@5, MRR. Did it find the right past event?
2. *State reasoning* — accuracy on current state, contradiction-detection P/R, ECE.
3. *Response faithfulness* — is the spoken answer supported by the retrieved evidence?
4. *Practical utility*, 5. *Human factors* — named as future work; no human subjects here.

**Perception.** Every experiment runs twice: with **oracle perception** (events read from
narrations) and with **learned perception** (events from a trained head over frozen visual
features). This separates a memory-policy failure from a perception failure — without it, a weak
detector would be indistinguishable from a wrong hypothesis.

## 5. Hypotheses

**H1 — last-seen ≠ current.** A last-occurrence system's retrieval score stays high while its
current-state accuracy is materially lower, because a large fraction of queries are stale. Adding an
explicit contradiction scan closes most of that gap.
*Status: the staleness premise is confirmed — see `results/results.md`.*

**H2 — the forgetting curve is on the wrong axis.** MemoryBank applies Ebbinghaus decay to
*retention*. In episodic visual memory an old event is frequently the **only** evidence that exists,
so decaying it away destroys the answer. Tested two ways: (a) decay-as-retention should reduce the
answerable rate without improving accuracy; (b) decay-as-confidence should be a weak staleness
predictor compared with the contradiction signal — elapsed time is a proxy for "has this changed?",
and negative evidence is the thing itself.

**H3 — event selection under budget.** Interaction-triggered capture (threshold on a learned keep
score) should retain placement-event recall at a fraction of the frames kept by uniform sampling.

**H4 — calibration / overtrust.** The headline safety number is the **confidently-wrong rate**: how
often the system asserts a location at high confidence and is wrong. A system may improve accuracy
and still be more dangerous if its errors move into the high-confidence band.

## 6. What this is not

Not a new foundation model, and not a long-context model (roadmap §11). It is a retrieval and
event-memory layer over selected observations, with the contribution sitting in the verification and
answer policy rather than in the backbone. Vector search over past observations is a standard
technique and is used here as a baseline component, not claimed as a contribution.
