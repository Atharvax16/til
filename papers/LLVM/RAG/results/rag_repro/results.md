# RAG reproduction — results

Index: 15,077 passages (3,077 answerable + 12,000 distractors) | eval: 96 SQuAD-dev questions, open-domain | k=5

EM is reported ±1 binomial standard error.

| system | n | EM | F1 | answer-recall@k |
|---|---|---|---|---|
| no retrieval (closed book) | 96 | 1.0 ± 1.0 | 6.2 | n/a |
| random passages | 96 | 2.1 ± 1.5 | 7.0 | 1.0 |
| BM25 word overlap | 96 | 44.8 ± 5.1 | 55.7 | 81.2 |
| DPR, not fine-tuned (Frozen) | 96 | 22.9 ± 4.3 | 32.7 | 57.3 |
| RAG-Token | 96 | 24.0 ± 4.4 | 30.8 | 50.0 |
| RAG-Sequence (n=48) | 48 | 0.0 ± 0.0 | 0.0 | 60.4 |

## Answer recall@k (Fig 3 centre)

| retriever | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| RAG-Token (learned DPR) | 26.0 | 36.5 | 50.0 | 61.5 | 70.8 |
| Frozen DPR | 24.0 | 38.5 | 57.3 | 65.6 | 76.0 |
| BM25 | 67.7 | 75.0 | 81.2 | 86.5 | 89.6 |

## Distinct n-gram ratio (Table 5), n=48 generations

| system | 1-gram | 2-gram | 3-gram |
|---|---|---|---|
| Gold answers | 92.6% | 100.0% | 100.0% |
| BART (closed book) | 88.6% | 90.9% | 86.7% |
| RAG-Token | 88.4% | 87.8% | 83.3% |
| RAG-Sequence | 68.8% | 87.5% | 92.3% |

## Claim by claim

| claim | measurement | |
|---|---|---|
| §2   Eq (1) & Eq (2) from scratch match `transformers` | \|Δ\| = 0.0e+00 | ✓ |
| §4.1 retrieval beats closed-book (same generator weights) | 1.0±1.0 -> 24.0±4.4 EM | ✓ |
| §4.1 retrieval beats *irrelevant* retrieval (random-doc control) | 2.1±1.5 -> 24.0±4.4 EM | ✓ |
| §4.1 RAG-Sequence ≥ RAG-Token on short factoid QA (paired, n=48) | 0.0±0.0 vs 22.9±6.1 EM | ✗ |
| §4.5 learned retrieval > frozen DPR (paper: 43.5 vs 37.8) | 24.0±4.4 vs 22.9±4.3 EM | ~ |
| §4.5 learned retrieval > BM25 (paper: 43.5 vs 29.7) | 24.0±4.4 vs 44.8±5.1 EM | ✗ |
| §4.5 answer recall@k rises monotonically with k (Fig 3 centre) | 26 -> 36 -> 50 -> 61 -> 71 | ✓ |
| §4.5 EM shape differs between the variants (Fig 3 left) | tok peaks k=5, seq peaks k=1 | ~ |
| §4.5 generation diversity RAG-Seq > RAG-Token > BART (Table 5) | 69 / 88 / 89 %  (1-gram) | ~ |
| §4.1 correct even when the answer is in no retrieved doc (paper 11.8%) | 2.1% of 48 such questions | ✓ |
| §4.5 index hot-swap changes world knowledge with zero retraining | see §10 | ✓ |
| §2   answer-only loss produces non-zero gradient at the retriever | ‖g‖ = 6.6e+01 | ✓ |
| §2   gold-passage trust rises without passage labels | 0.097 -> 0.354 | ✓ |
| App H  retriever has not collapsed (most-returned doc share) | 427 distinct docs, top doc in 4% of queries | ✓ |

✓ reproduced · ~ right direction but inside the noise at this sample size · ✗ did not reproduce
