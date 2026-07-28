# VoxSight Recall — what building it actually taught me

*Companion to `problem_statement.md` (the design) and `results/results.md` (the numbers).
This is the conceptual recap: what changed in my head, not what the tables say.*

---

## 1. The one-sentence version

An episodic visual-memory system can retrieve the correct past event **every single time** and still
be wrong about the present in a third of cases — and no metric in the episodic-retrieval literature
would reveal it.

That sentence is the whole project. Everything else is machinery to make it measurable.

---

## 2. The thing I didn't expect: a perfect retriever is not a partial credit

Going in, I assumed the story would be "retrieval is good but imperfect, and the residual errors
compound downstream." That's the ordinary shape of a pipeline critique.

What the oracle arm actually shows is stranger. Retrieval isn't *good*, it's **exactly perfect** —
Recall@1 = 100%, because object identity is symbolic and there is no embedding to get wrong. And the
answer built on top of it is wrong 30.9% of the time.

This matters because it kills a whole class of response. You cannot answer this finding with a
better encoder, a reranker, a bigger index, or query rewriting. The information required to answer
correctly **is not present in the retrieved object**. It lives in the events that came *after* the
one you retrieved, and last-occurrence retrieval is structurally incapable of consulting them.

The lesson generalises past this project: when a pipeline metric saturates, that is not evidence the
pipeline works. It can equally mean the metric is measuring a subproblem that was never the hard part.

---

## 3. Negative evidence is a first-class citizen

The technical core turned out to be a distinction I did not have language for beforehand:

- A **PUT** is *positive evidence*. It grounds an answer: "the keys are on the counter."
- A **TAKE** grounds nothing. It cannot answer any question. Its entire function is to **invalidate**
  a previous answer.

Every memory system I read stores the first kind. `MemoryBank`'s strength scores, `MemGPT`'s working
context, RAG's document index — all of them are built to retrieve *things that support an answer*.
None has a natural slot for a record whose only purpose is to destroy the credibility of another
record.

And the asymmetry in the data is brutal: **TAKE events outnumber PUT events** (19,866 vs 17,558),
and only 31% of PUTs even name a receptacle. So the majority of what a kitchen camera observes is
information that *reduces* what the system can safely claim. A memory design optimised for
accumulating knowledge is pointed in the wrong direction.

The third state matters as much. A bare *"put down plate"* — 69% of placements — means "it moved,
I don't know where." A schema with an object field and a location field cannot represent that
without lying. `PLACED_UNKNOWN` had to be a first-class state, and it is exactly the 0.9% that
S3 still gets wrong under oracle perception, because no contradiction scan can recover a location
that was never observed.

---

## 4. Where MemoryBank goes wrong, precisely

I came in expecting to find MemoryBank's Ebbinghaus forgetting curve *imperfect* here. It's worse
than that, and the failure is clean enough to state as a rule.

**Applied to retention, it is strictly harmful.** Not a tradeoff — harmful. Forgetting made 78.7% of
queries unanswerable and cost 29.7 accuracy points, and the decay-constant sweep is monotonic all
the way to "never forget." There is no setting that pays.

The reason is a domain difference I now think is the important insight:

> In conversational memory, an old message is usually **redundant** with a newer one — decay
> discards a copy. In episodic visual memory, the old event is frequently **the only record that
> the object was ever placed anywhere** — decay discards the answer.

Same mechanism, opposite consequence, because of a property of the data rather than anything about
the algorithm. That is what "know when a technique transfers" actually looks like in practice.

**Applied to confidence, it is on the wrong axis but not absurd.** Elapsed time predicts staleness at
AUC 0.547 — barely off chance — while a contradiction detector too weak to classify its own training
distribution manages 0.703. Time is a *proxy* for "might this have changed?"; negative evidence is
the thing itself. Given a choice, always instrument the thing itself.

The non-monotonicity is the detail I'd have missed by reasoning from the armchair: objects are
*staler* at <5 s than at 5–30 s, because something you just set down is something you're about to
pick straight back up. No monotonic decay function can express that shape.

---

## 5. The trap I nearly walked into

The contradiction signal scores **AUC 0.984** under oracle perception. For about ten minutes that
looked like the headline result.

It's circular. Under oracle perception, an object is stale *exactly when* something touched it since
the sighting — which is the definition of the contradiction signal. I was measuring a tautology and
would have reported it as a finding.

The fix was the `oracle-time` arm: keep the event *timing* from narrations, but take the PUT/TAKE
call from the model, so the signal is genuinely noisy and can be wrong. That arm is where every
honest claim in this project lives.

Generalising: whenever a signal scores suspiciously well, check whether it participated in
constructing the ground truth. The three-arm design wasn't methodological politeness — without the
middle arm I had no non-circular claim at all.

---

## 6. Calibration is not the same as not being confident

The most instructive row in the results is S1, the Ebbinghaus-confidence system.

It cuts confidently-wrong answers from 23.8% to 0.5%. On the safety metric it looks like the best
system in the table. And its **ECE is the worst of any system, worse than the naive baseline** —
0.462 against S0's 0.288.

It isn't calibrated. It's uniformly under-confident. It achieves safety by never committing to
anything, which for a blind user asking where their keys are is a different failure mode, not a fix.
"Rarely confidently wrong" and "well calibrated" are independent axes, and a system can buy the
first by abandoning usefulness.

The same trap sits inside the fully-learned arm, and I nearly shipped it: its 0.275 accuracy looks
respectable until you notice it asserts a location on 2.8% of queries. Abstention scores as correct
whenever the object really did move, so **most of that "accuracy" is earned by declining to answer**.
Any metric that rewards abstention needs the assertion rate printed directly beside it.

---

## 7. What the reading was actually for

Retrospectively, each paper contributed one specific thing, and none of them contributed what I
expected when I read it:

| paper | what I thought I'd use | what I actually used |
|---|---|---|
| **NTM** | memory architecture | the vocabulary of read/write/erase as *separable operations* — which is what made "TAKE is an erase with no write" thinkable |
| **RAG** | the retrieval pipeline | the provenance argument: retrieve evidence *before* generating, so the answer has something to be checked against |
| **MemGPT** | tiered memory | the critique in its §6 — self-edited memory with no provenance. The last-seen/current gap is that same defect with a camera attached |
| **MemoryBank** | forgetting policy | a hypothesis to *falsify*. Its decay curve is the thing this project measures and rejects |
| **Ego4D / MemPal** | baselines | the definition of the standard task, which is precisely what I needed to show is under-specified |

The roadmap was right that the contribution isn't vector search. It's the verification step between
retrieval and speech — the one nobody's benchmark scores.

---

## 8. Where it actually stands

Honest summary of confidence:

- **H1 (last-seen ≠ current)** — solid. 27–31% across every labelling variant, on 173,624 queries,
  and a lower bound because unnarrated pickups only hide staleness.
- **H2(a) (retention decay harmful)** — solid, and cleanly explained by a domain property.
- **H2(b) (time is the wrong axis)** — solid on the non-circular arm, consistent across 3 seeds.
- **H4 (overtrust)** — solid, and the S1 twist is the most transferable lesson here.
- **H3 (selection under budget)** — **not established.** Margins sit inside seed noise. I fixed one
  genuine design flaw in it (threshold sweeps couldn't reach low budgets) and still can't claim the
  result.

The binding constraint is perception: the event head reaches macro-F1 0.437, the receptacle head
gets **zero** of the 27 named-receptacle validation windows right, and consequently the end-to-end
arm doesn't work. The policy contribution is real — S3 beats S0 on every seed even with a weak
detector — but roughly half the oracle-arm gain survives contact with real vision, and I can't yet
say how that scales with a competent detector.

Which is the honest place to stop: the claim about *what the metric misses* is established, and the
claim about *how well a fixed version works in the wild* is not.
