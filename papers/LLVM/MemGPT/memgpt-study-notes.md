# MemGPT: Towards LLMs as Operating Systems

**Packer, Wooders, Lin, Fang, Patil, Stoica, Gonzalez — UC Berkeley**
arXiv:2310.08560 (v2, Feb 2024) · [research.memgpt.ai](https://research.memgpt.ai) · became **Letta**

---

## TL;DR

Don't wait for bigger context windows. Treat the small one you have as RAM, put everything else in a database, and let the model itself decide what to page in and out.

The argument in four steps:

1. Context windows are small, and the obvious fix — make them bigger — is quadratically expensive *and* doesn't work well anyway, because models ignore the middle of long contexts.
2. Operating systems already solved this exact problem with virtual memory: small fast tier, big slow tier, swap between them on demand.
3. So build that hierarchy for an LLM, and give the model function calls to move data across the boundary itself. Add a memory-pressure alarm, multi-step chaining, and error feedback so it can actually manage the thing.
4. Result: on long conversations and long documents, this beats the same base model with a plain summary or plain retrieval, and it stops degrading as the task gets longer.

**Why it mattered:** it reframed memory as an *allocation* problem rather than a *capacity* problem, and made the model the allocator.

---

## 1. The core analogy

### The OS version

Your laptop has 16 GB of RAM but runs programs needing far more. It cheats: keep what's in use in RAM, push the rest to disk, swap pieces back the moment a program reaches for them. The program *feels* like it has unlimited memory. It doesn't — something is just swapping pages very fast.

When a program touches an address that isn't currently in RAM, execution stops, the OS fetches the page from disk, drops it into RAM, and execution resumes as if it had been there all along. That stumble is a **page fault**.

### The desk model (working intuition)

| Desk | System |
|---|---|
| Desk — only 5 sheets fit, only these are readable | Context window / prompt |
| Filing cabinet — 10,000 sheets, invisible until fetched | External database |
| The clerk who swaps sheets between them | **The LLM itself** |
| Permanent sticky note on the desk, never thrown out | Working context |
| Loose pages, oldest get dumped when full | FIFO queue |

Anything in the cabinet may as well not exist until it's physically on the desk.

### The substitution

- Context window → RAM
- External databases (Postgres + pgvector) → disk
- **The LLM → the pager**

That last line is the actual contribution. In a real OS the pager is a separate program deciding what to swap. In MemGPT there is no separate clerk — the model decides what to file away and what to fetch, *for itself*, by emitting function calls that edit the very prompt it is reading from.

Hence the title. The model isn't a program running *on* an OS; it's doing the OS's job.

---

## 2. Why not just build a bigger context window?

Two claims, both answering the obvious objection.

### Claim 1 — quadratic cost

In self-attention every token attends to every other token: n tokens → an n×n attention matrix.

| Context length | Attention scores (per head, per layer) |
|---|---|
| 2k | 4 million |
| 8k | 64 million |
| 32k | ~1.07 billion |
| 128k | ~16.8 billion |

8k → 128k is **16× the tokens but 256× the attention compute** and activation memory.

Not impossible — FlashAttention, sparse attention, linear approximations all attack this — but each is a real research problem. Context length is a scarce, expensive resource. That's what justifies treating it like RAM at all.

### Claim 2 — "Lost in the Middle" (Liu et al., 2023a)

Even if you pay for it, the model doesn't use it.

**The experiment:** a QA task with K retrieved documents, exactly one containing the answer. Vary *where in the list* the gold document sits. If context were uniformly usable, accuracy should be flat across positions.

It isn't. You get a **U-shape** — strong at position 1, decent at the end, pronounced sag in the middle. In some settings, middle-position accuracy drops *below* what the same model scores with no documents at all.

So "128k context window" is a marketing number, not effective capacity. The usable window is smaller and position-dependent.

### Why the two together justify the paper

The expensive path is both costly *and* delivers less than advertised. So take the other path: keep the window small and fixed, and be smart about **what occupies it at any given moment**.

Back to the OS framing — the insight of virtual memory was never "buy more RAM." It was that with a good paging policy, a small physical memory serves a working set far larger than itself, because programs exhibit **locality**. MemGPT's bet is that conversations and documents have the same property: you don't need 50 sessions in context to answer one question, you need the three relevant messages, paged in on demand.

---

## 3. Architecture

### Main context (the prompt tokens)

Three contiguous regions:

| Region | Access | Purpose |
|---|---|---|
| **System instructions** | Read-only, static | Describes the memory hierarchy, the tiers, and the function schemas. The "kernel." |
| **Working context** | Read/write, **only via function calls** | Fixed-size unstructured text. Persistent facts: user's name, preferences, agent persona. The pinned page. |
| **FIFO queue** | Read/write, managed by queue manager | Rolling message history. **Index 0 holds a recursive summary** of everything already evicted. |

### External context

| Store | Written by | Read by | Contents |
|---|---|---|---|
| **Recall storage** | Queue manager, automatically | Function calls | Every message ever exchanged, forever |
| **Archival storage** | Function calls | Function calls | Arbitrary-length text objects. Postgres + pgvector, HNSW index, cosine similarity |

Key asymmetry: recall storage is written *for* the agent (automatic, complete), archival storage is written *by* the agent (deliberate, curated).

### Data flow

```
                 ┌─────────────── LLM context window (e.g. 8k) ───────────────┐
                 │                                                            │
   read-only ───▶│  System instructions                                       │
                 │                                                            │
  fn calls ─────▶│  Working context      (persistent facts, pinned)           │
                 │                                                            │
  queue mgr ────▶│  FIFO queue           ([0] = recursive summary, then msgs) │
                 │                                                            │
                 └────────────────────────────┬───────────────────────────────┘
                                              │  completion tokens
                                              ▼
                                      ┌───────────────┐
                                      │Function exec. │
                                      └───────┬───────┘
                          ┌───────────────────┴──────────────────┐
                          ▼                                      ▼
                  ┌───────────────┐                     ┌────────────────┐
                  │Recall storage │                     │Archival storage│
                  │ (all messages)│                     │ (vector search)│
                  └───────────────┘                     └────────────────┘
```

---

## 4. The three control mechanisms

The load-bearing parts. Remove any one and the system falls apart.

### 4.1 Memory pressure interrupts

Two alarms as the desk fills:

**At ~70% of the window** — the queue manager injects a system message warning of impending eviction. This is the model's chance to act: read what's on the desk and decide whether anything is worth keeping. If so, write it into working context (`working_context.append("Birthday is February 7")`) or file it to archival storage.

**At 100%** — no more asking. Flush ~50% of the queue. Before dumping, generate a **new recursive summary** from the existing summary plus the evicted messages, and leave that at queue index 0.

Two properties worth naming:

- **Lossy.** The summary is worse than the originals, and it happens repeatedly — a summary of a summary of a summary. Detail degrades each round.
- **Not destroyed.** Everything persists in recall storage indefinitely and can be searched back. *Evicted ≠ deleted.*

### 4.2 Function chaining via heartbeat

Default behaviour: the model calls one function, gets a result, yields control back to the user. One search, one reply.

But many tasks need several steps — key A points to B, B points to C, C is the answer. Yielding after each step would force the user to manually drive the loop.

So the model can attach `request_heartbeat=true` to a call, meaning: *don't give the human the floor, hand control straight back to me, I'm not done.*

This is what enables:
- multi-hop lookups (the nested KV task)
- pagination through search results (124 matches, 10 per page → flip to page 2, 3, 4)
- collating across multiple documents before answering

Without a heartbeat flag the call is a **yield**: no further inference until the next external event (user message, timed interrupt, system alert).

### 4.3 Error feedback

Function results — **including runtime errors** — are fed back into main context. If a write to working context fails because it's already at capacity, the model *reads that error* and can respond: delete an old entry first, or route the content to archival storage instead.

Simple, but it's the difference between a system that breaks on the first bad call and one that recovers. The failure lands in the same place as everything else the model reads.

### Why exactly these three

- **(1)** stops the desk overflowing
- **(2)** lets the model take multiple actions in sequence
- **(3)** lets it recover when an action fails

Without 1 you run out of room mid-conversation. Without 2 you can only do single-step retrieval. Without 3 one bad call derails the run.

### Control flow generally

**Events** trigger inference: user messages, system messages (memory warnings), user interactions (login, upload complete), and **timed events on a schedule** — which means the agent can act unprompted. Events are parsed to plain text and appended to main context.

---

## 5. Experiments and results

### 5.1 Deep Memory Retrieval (consistency)

Built on the Multi-Session Chat dataset (Xu et al., 2021) — five sessions per conversation, consistent personas. The authors add a session 6 with a single QA pair, generated by an LLM instructed to write a question answerable *only* from the chat log, never from the persona summary.

Scored by ROUGE-L recall (recall, because generated answers are more verbose than gold) plus an LLM judge.

Baselines see a lossy summary of the five prior sessions. MemGPT has the full history but must reach it via paginated search.

| Model | Accuracy | ROUGE-L (R) |
|---|---|---|
| GPT-3.5 Turbo | 38.7% | 0.394 |
| **+ MemGPT** | **66.9%** | **0.629** |
| GPT-4 | 32.1% | 0.296 |
| **+ MemGPT** | **92.5%** | **0.814** |
| GPT-4 Turbo | 35.3% | 0.359 |
| **+ MemGPT** | **93.4%** | **0.827** |

### 5.2 Conversation opener (engagement)

Can the agent open a new session with a message drawing on accumulated knowledge? Scored by similarity to gold persona labels (SIM-1/SIM-3) and to the human-written opener (SIM-H).

| Method | SIM-1 | SIM-3 | SIM-H |
|---|---|---|---|
| Human | 0.800 | 0.800 | 1.000 |
| GPT-3.5 Turbo | 0.830 | 0.812 | 0.817 |
| GPT-4 | 0.868 | 0.843 | 0.773 |
| GPT-4 Turbo | 0.857 | 0.828 | 0.767 |

Matches or exceeds human openers. The paper notes MemGPT's openers are more verbose and cover more persona aspects, and that **working context is key** here.

### 5.3 Multi-document QA

NaturalQuestions-Open, retriever-reader setup from Liu et al. Same retriever for all methods (`text-embedding-ada-002`, cosine similarity). 50 sampled questions, late-2018 Wikipedia dump, LLM judge.

- **Baselines** are capped at retriever performance — if the gold doc isn't in the top-K, they can never see it. Accuracy climbs with K, then degrades once truncation is needed to fit.
- **MemGPT** is **flat across K**. It can call the retriever repeatedly and page through results, so its ceiling isn't set by what fits in the window.

Flat, but flat-and-mediocre — see the critique below.

### 5.4 Nested key-value retrieval

New task. 140 UUID pairs (~8k tokens, matching the GPT-4 baseline window). Values may themselves be keys, requiring multi-hop lookup. Nesting levels 0–4, 30 orderings sampled.

| Nesting | GPT-3.5 | GPT-4 / Turbo | MemGPT + GPT-4 |
|---|---|---|---|
| 0 | good | good | good |
| 1 | **0%** | degrading | flat |
| 2 | 0% | degrading | flat |
| 3 | 0% | **0%** | **flat** |

GPT-3.5's failure mode is instructive: it just returns the original value without following the chain. MemGPT with GPT-4 is unaffected by depth. MemGPT with GPT-4 Turbo and GPT-3.5 beat their baselines but start dropping at 2 levels — failing to perform *enough* lookups.

Note the oddity: **MemGPT + GPT-4 Turbo performs worse than MemGPT + GPT-4** on this task, despite Turbo being the stronger baseline.

---

## 6. Critique — where it's weak

The interesting part.

### It's scaffolding, not learning

The entire memory policy lives in the system prompt as natural-language prose. No learned retrieval policy, no learned eviction policy, no gradient anywhere.

### Heavily dependent on base-model function calling

Same scaffolding, same prompts, same databases — the only variable is the model plugged in, and results swing from 66.9% to 93.4%. The paper states directly that MemGPT+GPT-3.5 degrades on document QA because of weak function calling. "The LLM decides" really means "GPT-4 decides, and GPT-4 happens to be decent at it."

### The agent gives up early

The authors' own admission: MemGPT **often stops paging through retriever results before exhausting the database**. Unbounded recall in theory; in practice bounded by the model's patience.

This isn't promptable away — it's the *absence of a policy*. No notion of expected value of one more search, no calibration, no stopping rule. Just a vibe about whether it's seen enough.

### Small evals

50 sampled questions for document QA. LLM-judge scoring throughout. No strong RAG baseline with reranking or query rewriting.

### No provenance

Working context is self-edited free text. When the agent overwrites `"Boyfriend named James"` with `"Ex-boyfriend named James"`, there's no record of:

- **when** it learned that
- **from which** observation
- **how confident** it was
- **what** it overwrote

A bad write is indistinguishable from a good one downstream. Recursive summarization compounds this — each flush re-summarizes a summary, so error accumulates with no path back to the source.

---

## 7. "How does the model become smart at deciding?"

It doesn't. It **borrows** smartness from the base model.

The decisions come from three places, none of which is learning:

1. **The system prompt** — a long natural-language description of the hierarchy and functions. The policy is *prose*.
2. **The base model's pretrained instruction-following** — GPT-4 was already trained to read a tool description and call it sensibly. MemGPT contributes none of that; it hands GPT-4 a new toolkit and a rulebook.
3. **Error feedback** — recovery *within a single conversation*. Nothing persists. Next session it makes the same mistake.

### What would actually make it smart

- **Fine-tune on memory-management traces** — collect episodes where paging decisions led to correct vs incorrect final answers, train on the good ones. (Letta later moved toward this for smaller models.)
- **RL over retrieval trajectories** — reward the final answer, let the model learn how many searches to run and what to keep. Turns eviction into a *learned* policy.
- **A learned eviction scorer** — a small model scoring each message for "will this matter later," instead of blanket-dumping the oldest 50%.

### The bit worth sitting with

Real operating systems don't use judgment at all. LRU, clock — dumb, fixed, hand-designed heuristics, decades old. They work because memory access has **locality**: a measurable statistical property you can design against and reason about.

MemGPT swaps a verifiable heuristic for an unverifiable model call. More flexible, certainly. But you lose the ability to say *why* a memory was dropped or kept, and you can't audit it after the fact.

---

## 8. Related work positioning

- **Long-context LLMs** (Longformer, sparse attention, Linformer, positional interpolation) — MemGPT *builds on* these rather than competing; a longer window just means bigger "main memory." Its contribution is the hierarchy on top.
- **Retrieval-augmented models** (RAG, REALM, DPR, RETRO) — MemGPT's external context is RAG-descended. Closest relative is **FLARE** (Jiang et al.), where the model actively decides *when* and *what* to retrieve. **IRCoT** (Trivedi et al.) interleaves retrieval with chain-of-thought.
- **LLMs as agents** — Generative Agents (Park et al.) added memory + planning; WebGPT (Nakano et al.) used similar pagination for context control; ReAct (Yao et al.) interleaved reasoning and acting. MemGPT's stated distinction: focus specifically on **long-term memory of user inputs**.

---

## 9. Links to VoxSight Recall

The gap in §6 is roughly the position VoxSight occupies.

MemGPT proves the hierarchy works. It does **not** touch whether a memory can be *trusted*. For a blind or low-vision user relying on an agent's recall of what its camera saw, that's the whole ballgame — a confidently wrong memory is worse than an absent one.

Framing for a related-work section:

> MemGPT introduces self-directed memory management via function calling, but treats every write as equally authoritative. Memory edits carry no source attribution, no timestamp of acquisition, and no confidence estimate, and recursive summarization compounds errors with no path back to the originating observation. Provenance and calibration over episodic memory are orthogonal to the paging mechanism and remain unaddressed.

Mechanisms worth carrying over regardless of the provenance layer:

- **The pressure/flush split** (soft warning → hard eviction) is a clean pattern for giving an agent a chance to preserve before losing.
- **Heartbeat chaining** is the minimal primitive for multi-hop retrieval.
- **Error-in-context** feedback for recovery.
- **Recall vs archival separation** — automatic complete log vs deliberate curated store. Provenance naturally attaches to the archival tier: every curated write should carry a pointer back into the recall tier.

Also note the parallel to the diagnostic-paradox thread: MemGPT optimizes retrieval *reachability* (can the fact be found?) without measuring *decision correctness downstream* (did the retrieved memory lead to the right action?). Same shape of gap — a metric that looks good at one level while the thing that matters at the next level is never measured.

---

## 10. Open questions

- What's a principled **stopping rule** for pagination? Expected-value-of-search, or a calibrated confidence threshold?
- Can eviction be **learned** rather than fixed-percentage? What's the LLM equivalent of locality — is there a measurable statistical property of conversational memory you can design a heuristic against?
- How much of MemGPT's gain survives with a **200k-token base model**? Does the hierarchy still pay for itself, or does it collapse into "just use the window"?
- How do you **audit** a self-edited memory store? What does a diff/version history over working context buy you?
- Does recursive summarization have a measurable **degradation curve**? How many flushes before the summary is useless?

---

## Glossary

| Term | Meaning |
|---|---|
| **Paging** | Swapping data between a fast small tier and a slow large tier |
| **Page fault** | Program touches data not currently in fast memory → fetch it, then resume |
| **Main context** | The prompt tokens; everything the model can actually see |
| **External context** | Everything outside the window; must be explicitly moved in |
| **Working context** | Fixed-size read/write block in the prompt, edited only by function call |
| **FIFO queue** | Rolling message history in the prompt |
| **Recall storage** | Database of every message ever exchanged |
| **Archival storage** | Vector-searchable store of arbitrary text objects |
| **Queue manager** | Component handling appends, eviction, summarization |
| **Memory pressure** | System warning at ~70% capacity, before eviction |
| **Flush** | Hard eviction at 100%, dropping ~50% of the queue |
| **Recursive summary** | Summary of evicted messages, itself re-summarized each flush |
| **Heartbeat** | `request_heartbeat=true` — return control to the model, not the user |
| **Yield** | Function call without heartbeat; pause until next external event |
| **Lost in the middle** | Liu et al. 2023a — U-shaped accuracy by position in long context |
