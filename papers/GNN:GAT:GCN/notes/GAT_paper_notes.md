# Graph Attention Networks — Paper Notes

Working notes on **Graph Attention Networks** (Veličković, Cucurull, Casanova, Romero, Liò,
Bengio — ICLR 2018), built up equation by equation. Paper: [`../GAT.pdf`](../GAT.pdf).
Companion notebook (both GCN and GAT implemented from scratch and trained on Cora):
[`../experiments/gcn_vs_gat_from_scratch.ipynb`](../experiments/gcn_vs_gat_from_scratch.ipynb).

---

## Contents

1. [The one-sentence idea](#1-the-one-sentence-idea)
2. [Where GAT sits: spectral vs non-spectral GNNs](#2-where-gat-sits-spectral-vs-non-spectral-gnns)
3. [The graph attentional layer, equation by equation](#3-the-graph-attentional-layer-equation-by-equation)
4. [Multi-head attention](#4-multi-head-attention)
5. [GAT vs GCN, side by side](#5-gat-vs-gcn-side-by-side)
6. [Why no Laplacian matters: inductive vs transductive](#6-why-no-laplacian-matters-inductive-vs-transductive)
7. [Comparison to GraphSAGE](#7-comparison-to-graphsage)
8. [Complexity](#8-complexity)
9. [Datasets and experimental setup](#9-datasets-and-experimental-setup)
10. [Results](#10-results)
11. [Qualitative analysis: t-SNE + attention map](#11-qualitative-analysis-t-sne--attention-map)
12. [Limitations the authors admit to](#12-limitations-the-authors-admit-to)
13. [Paper → code: how this maps onto the notebook](#13-paper--code-how-this-maps-onto-the-notebook)
14. [One-line summaries to remember](#14-one-line-summaries-to-remember)

---

## 1. The one-sentence idea

Replace a graph-convolution layer's **fixed, structurally-determined** neighbor weights with
**learned attention weights** computed from the neighbors' features — so a node decides *how
much to listen to each neighbor*, instead of that being baked in by degree alone.

The paper's framing: this is `Attention Is All You Need`'s self-attention, applied to graphs,
with the graph structure used only as a **mask** (a node may only attend to its actual
neighbors, not the whole graph).

---

## 2. Where GAT sits: spectral vs non-spectral GNNs

The intro sorts prior graph-convolution work into two families, and GAT is explicitly a
reaction to the weaknesses of both:

**Spectral approaches** — define convolution via the graph Fourier transform (eigendecomposition
of the graph Laplacian):
- Bruna et al. (2014): convolution in the Fourier domain — expensive eigendecomposition,
  filters aren't spatially localized.
- Henaff et al. (2015): smooth spectral filter parameterization → spatial localization.
- Defferrard et al. (2016): Chebyshev polynomial approximation of the filters — avoids computing
  eigenvectors, gives localized filters.
- **Kipf & Welling (2017), i.e. GCN**: simplifies further by restricting filters to a 1-hop
  neighborhood, giving the now-famous $H' = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}HW$.

The shared problem: **the learned filters depend on the Laplacian eigenbasis, which depends on
the specific graph.** A model trained on one graph's structure doesn't transfer to a
differently-structured graph. This is why GCN is fundamentally **transductive** — train and test
nodes must live in the same fixed graph.

**Non-spectral (spatial) approaches** — define the convolution directly on the graph, operating
on local neighborhoods, sidestepping the Laplacian:
- Duvenaud et al. (2015): a separate weight matrix per node degree (molecular fingerprints).
- Atwood & Towsley (2016): powers of a transition matrix define the neighborhood.
- Niepert et al. (2016): extract and normalize fixed-size neighborhoods.
- Hamilton et al. (2017), **GraphSAGE**: sample a fixed-size neighborhood, aggregate with
  mean/LSTM/pooling — the first method built explicitly for **inductive** learning.

GAT positions itself as a non-spectral approach that borrows the *self-attention* trick from
sequence models (Bahdanau et al. 2015; Vaswani et al. 2017) to solve the "different-sized,
differently-important neighborhoods" problem cleanly — no eigendecomposition, no fixed sampling
budget, no per-degree weight matrices.

---

## 3. The graph attentional layer, equation by equation

**Setup.** Input: node features $h = \{\vec{h}_1, \ldots, \vec{h}_N\}$, $\vec{h}_i \in
\mathbb{R}^F$. Output: transformed features $h' = \{\vec{h}'_1, \ldots, \vec{h}'_N\}$,
$\vec{h}'_i \in \mathbb{R}^{F'}$ (possibly a different dimension).

**Step 1 — shared linear transform.** Every node gets the same learned weight matrix $W \in
\mathbb{R}^{F' \times F}$ applied to it. This is the one non-negotiable "at least one learnable
transform" the paper insists on — without it there's nothing to actually learn beyond the
attention mechanism itself.

**Step 2 — raw attention score (eq. 1):**

$$e_{ij} = a(W\vec{h}_i, W\vec{h}_j)$$

$a$ is a *shared* attention mechanism, the same function applied to every edge in the graph.
$e_{ij}$ says "how important are node $j$'s features to node $i$." In full generality this could
be computed for **every pair of nodes** — which would throw away the graph structure entirely
and turn this into ordinary (quadratic-cost) self-attention over a set.

**Step 3 — masking (the graph part).** GAT restricts $e_{ij}$ to only be computed for $j \in
\mathcal{N}_i$, the first-order neighbors of $i$ (the paper explicitly includes $i$ itself here
via a self-loop). This is the only place graph structure enters the model — everything else is
generic set-attention. It's also why GAT doesn't need the *whole* graph up front: computing
$e_{ij}$ only ever needs $i$'s local neighborhood.

**Step 4 — softmax normalization (eq. 2):**

$$\alpha_{ij} = \mathrm{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

Normalizing per destination node $i$ over just its own neighbors makes coefficients comparable
across nodes with wildly different degrees — a node with 2 neighbors and a node with 200
neighbors both get attention weights that sum to 1 over their own neighborhood.

**Step 5 — the concrete mechanism (eq. 3).** In the experiments, $a$ is a single-layer
feedforward net: a weight vector $\vec{a} \in \mathbb{R}^{2F'}$ applied to the *concatenation* of
the two transformed node vectors, followed by LeakyReLU (negative slope 0.2):

$$\alpha_{ij} = \frac{\exp\big(\mathrm{LeakyReLU}(\vec{a}^\top[W\vec{h}_i \| W\vec{h}_j])\big)}
{\sum_{k \in \mathcal{N}_i} \exp\big(\mathrm{LeakyReLU}(\vec{a}^\top[W\vec{h}_i \| W\vec{h}_k])\big)}$$

Splitting $\vec{a}$ into two halves — one dotted with $W\vec{h}_i$, one with $W\vec{h}_j$, then
summed — is algebraically identical to the concat-then-dot form and is how basically every
from-scratch implementation (including the notebook here) actually computes it, since it avoids
materializing an `[E, 2F']` tensor.

**Step 6 — aggregate (eq. 4).** The normalized weights become a weighted sum over the
(transformed) neighbor features, passed through a nonlinearity $\sigma$:

$$\vec{h}'_i = \sigma\Big(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W\vec{h}_j\Big)$$

This line is the whole layer. Everything before it was about how to *compute* $\alpha_{ij}$;
this is the message-passing step, and it looks structurally identical to GCN's aggregation —
the only difference is where the weight $\alpha_{ij}$ comes from.

---

## 4. Multi-head attention

Single-head self-attention is unstable to train (same motivation as in Transformers). The fix
(eq. 5): run $K$ independent attention mechanisms in parallel — each with its own $W^k$ and
$\vec{a}^k$ — and concatenate their outputs:

$$\vec{h}'_i = \Big\Vert_{k=1}^K \sigma\Big(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k \vec{h}_j\Big)$$

Output dimensionality becomes $KF'$, not $F'$ — this is why the paper's hidden layer with $K=8$
heads and $F'=8$ features/head produces 64-dim features per node. Each head can specialize:
the notebook's attention-weight bar chart shows exactly this — different heads assigning
different weight distributions to the same node's neighbors.

**At the final (prediction) layer**, concatenation stops making sense (you don't want the output
dimension to explode by $K\times$ right before a softmax over classes) — so the paper switches
to **averaging** across heads and delays the nonlinearity until after the average (eq. 6):

$$\vec{h}'_i = \sigma\Big(\frac{1}{K}\sum_{k=1}^K \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k \vec{h}_j\Big)$$

This concat-then-average pattern (concat on hidden layers, average on the output layer) is
exactly what both the paper's Cora/Citeseer/Pubmed architecture and the notebook's `GAT` model
do: `gat1` with `concat=True, heads=8`, `gat2` with `concat=False, heads=1` (or 8, for Pubmed —
see §9).

---

## 5. GAT vs GCN, side by side

| | GCN (Kipf & Welling, 2017) | GAT (this paper) |
|---|---|---|
| Aggregation weight for neighbor $j$ of node $i$ | fixed: $\frac{1}{\sqrt{\deg(i)\deg(j)}}$ | learned: softmax over $\mathrm{LeakyReLU}(\vec{a}^\top[W\vec{h}_i \Vert W\vec{h}_j])$ |
| Depends on node features? | no — purely structural | yes — the entire point |
| Derived from | 1-hop truncation of a spectral (Laplacian eigenbasis) filter | self-attention, masked by the graph |
| Needs eigendecomposition / matrix inversion? | no (Kipf's simplification avoids it too) | no |
| Weights sum to 1 over a neighborhood? | no (it's a symmetric normalization, not a distribution) | yes, by softmax construction |
| Multi-head? | no | yes |
| Naturally inductive? | not by default — filter is tied to the specific graph's degree structure | yes — attention is computed locally, per edge, from features alone |
| Time complexity (per layer, per head) | $O(\lvert V\rvert FF' + \lvert E\rvert F')$ | $O(\lvert V\rvert FF' + \lvert E\rvert F')$ — same order |

The complexity row is worth sitting with: **GAT is not more expensive than GCN in big-O terms.**
The attention mechanism doesn't cost anything asymptotically beyond the linear transform and
one scalar computation per edge — this is the paper's main computational selling point over
spectral methods, not just over other attention-based ideas.

---

## 6. Why no Laplacian matters: inductive vs transductive

This is the paper's biggest practical claim, and it's worth being precise about *why* it's true:

- GCN's filter is a function of the graph Laplacian $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$, a
  matrix tied to the *specific* graph's adjacency and degree structure. Even though Kipf &
  Welling's 1-hop simplification made this cheap to *compute*, the filter itself still encodes
  global structural information (degrees) that only makes sense for the graph it was computed
  on.
- GAT's attention coefficient $e_{ij}$ is a function purely of $W\vec{h}_i$ and $W\vec{h}_j$ —
  the node **features**, transformed by a weight matrix that is shared across the entire graph
  and, critically, across *any* graph. Nothing in the computation references global degree,
  Laplacian eigenvalues, or any other graph-wide quantity.
- Practical upshot: a trained GAT layer can be dropped onto a completely unseen graph at test
  time and it will compute sensible attention weights, because "attend to features that look
  relevant" generalizes; "here is the precomputed normalization for *this* graph's degree
  sequence" does not.

This is exactly why the PPI (protein-protein interaction) benchmark is inductive — training
graphs and test graphs are **disjoint tissues**, and GCN-style approaches can't be evaluated in
that setting without retraining, while GAT (and GraphSAGE) can.

---

## 7. Comparison to GraphSAGE

GraphSAGE (Hamilton et al., 2017) was the prior state of the art for inductive graph learning,
and the paper is careful to explain what GAT does differently:

- GraphSAGE **samples a fixed-size neighborhood** per node to keep computation bounded — it
  never sees a node's *entire* neighborhood at inference time. GAT attends over the whole
  neighborhood (computational cost scales with actual degree, not a fixed sample budget).
- GraphSAGE's best results used an **LSTM aggregator** over neighbor features — which requires
  imposing an artificial ordering on an unordered set (fixed by feeding random orderings during
  training, a workaround rather than a real solution). GAT's attention aggregation is
  order-invariant by construction (it's a weighted sum).
- On PPI, GAT beats the best tuned GraphSAGE variant by **20.5%** (0.973 vs 0.768 micro-F1) —
  the paper's strongest inductive result.

---

## 8. Complexity

Per attention head, computing $F'$ output features costs:

$$O(\lvert V\rvert F F' + \lvert E\rvert F')$$

— linear transform cost ($\lvert V\rvert F F'$) plus one attention computation per edge
($\lvert E\rvert F'$). "On par with GCN." $K$ heads multiply storage/parameters by $K$, but the
heads are independent and fully parallelizable across both nodes and edges.

The paper also mentions a practical limitation of their own implementation: sparse matrix
multiplication support in their framework (2017-era TensorFlow) only covered rank-2 tensors,
limiting batching across multiple graphs — an engineering constraint of the time, not a
fundamental one (modern implementations like PyTorch Geometric handle this with scatter-based
message passing, which is what the from-scratch notebook here does too).

---

## 9. Datasets and experimental setup

| | Cora | Citeseer | Pubmed | PPI |
|---|---|---|---|---|
| Task | transductive | transductive | transductive | inductive |
| Nodes | 2,708 (1 graph) | 3,327 (1 graph) | 19,717 (1 graph) | 56,944 (24 graphs) |
| Edges | 5,429 | 4,732 | 44,338 | 818,716 |
| Features/node | 1,433 | 3,703 | 500 | 50 |
| Classes | 7 | 6 | 3 | 121 (multi-label) |
| Train nodes | 140 (20/class) | 120 (20/class) | 60 (20/class) | 44,906 (20 graphs) |

**Transductive architecture** (Cora/Citeseer, tuned on Cora and reused): 2-layer GAT. Layer 1:
$K=8$ heads $\times$ $F'=8$ features, concatenated (64-dim), ELU nonlinearity. Layer 2: 1
attention head computing $C$ (num classes) features, softmax. Heavy regularization for such
small training sets: $L_2$ with $\lambda = 0.0005$, dropout $p=0.6$ applied to **both** the
layer inputs and the normalized attention coefficients themselves — meaning every training step
each node effectively sees a randomly *thinned* neighborhood, a built-in stochastic
regularizer that GCN's fixed weights don't have an equivalent of. Pubmed needed $K=8$ output
heads (not 1) and stronger $L_2$ ($\lambda=0.001$) because its training set is even smaller (60
nodes).

**Inductive architecture** (PPI): 3-layer GAT. Layers 1–2: $K=4$ heads $\times$ 256 features
(1024-dim), ELU. Layer 3: $K=6$ heads $\times$ 121 features, averaged, logistic sigmoid
(multi-label). Training sets are large enough that no $L_2$/dropout was needed; **skip
connections** across the intermediate layer were used instead, since without an
inductive-bias-heavy regularizer like dropout, depth alone would otherwise start hurting.

Both settings: Glorot init, Adam, lr $0.01$ (Pubmed) / $0.005$ (everything else), early stopping
on validation loss + accuracy/F1 with patience 100.

---

## 10. Results

**Transductive (accuracy):**

| Method | Cora | Citeseer | Pubmed |
|---|---|---|---|
| GCN (Kipf & Welling) | 81.5% | 70.3% | 79.0% |
| GCN-64* (best-tuned GCN, 64 hidden) | 81.4 ± 0.5% | 70.9 ± 0.5% | 79.0 ± 0.3% |
| **GAT** | **83.0 ± 0.7%** | **72.5 ± 0.7%** | 79.0 ± 0.3% |

+1.5pp on Cora, +1.6pp on Citeseer over GCN, tied on Pubmed. The paper reads this as evidence
that "assigning different weights to nodes of the same neighborhood" is genuinely useful, not
just extra parameters — the GCN-64* control (more hidden units, same aggregation rule) doesn't
close the gap.

**Inductive (micro-F1, PPI):**

| Method | PPI |
|---|---|
| GraphSAGE (best variant, tuned) | 0.768 |
| Const-GAT (same architecture, but $\alpha_{ij} = $ constant — i.e., GCN-like uniform aggregation) | 0.934 ± 0.006 |
| **GAT** | **0.973 ± 0.002** |

The **Const-GAT ablation is the cleanest experiment in the paper**: identical architecture,
identical everything, except attention weights are replaced with uniform weights (equivalent to
a GCN-style mean aggregator). GAT beats Const-GAT by 3.9pp — isolating the actual contribution
of learned attention from every other architectural choice (depth, skip connections, hidden
size).

---

## 11. Qualitative analysis: t-SNE + attention map

Figure 2 in the paper: a t-SNE projection of the first hidden layer's Cora features, colored by
the 7 ground-truth classes, with **edge thickness proportional to aggregated attention**
($\sum_k \alpha_{ij}^k + \alpha_{ji}^k$ across all 8 heads). Two things it demonstrates:

1. The learned representations cluster by class even though the layer was only ever trained
   with a classification loss on 140 labeled nodes — the graph structure plus attention is doing
   real representation learning, not just memorizing labels.
2. Attention isn't uniform — some edges are visibly thicker than others, but the paper is
   explicit that *interpreting* which edges get more weight (i.e., attaching domain meaning to
   it, à la the machine-translation alignment visualizations in Bahdanau et al.) is left as
   future work, not claimed here.

This qualitative point is exactly what the notebook's §7 bar chart tries to make concrete on a
single node instead of a whole t-SNE plot: pick one node, show that GCN's weights are flat/fixed
by degree while GAT's vary — and vary *differently per head*.

---

## 12. Limitations the authors admit to

Straight from §2.2/§4, worth remembering so as not to oversell GAT:

- **Receptive field is still bounded by depth**, same as GCN — a 2-layer GAT only ever sees
  2-hop neighborhoods. Skip connections are suggested as the fix for going deeper, not attention
  itself.
- **Batching across multiple graphs was awkward** in their framework at the time (rank-2-only
  sparse ops) — an implementation limitation, not a modeling one, but it constrained the PPI
  setup to batch size 2.
- **No edge features.** The mechanism only consumes node features; incorporating edge
  attributes (e.g., bond type in molecules, relation type in knowledge graphs) is left as future
  work.
- **Node-level only.** Graph-level classification (pooling an entire graph to one label) isn't
  addressed.
- **Redundant computation** when neighborhoods overlap heavily and computation is parallelized
  per-edge — no amortization across shared neighbors is done.
- **Attention interpretability is claimed as a *potential* benefit, not demonstrated rigorously**
  — see §11 above.

---

## 13. Paper → code: how this maps onto the notebook

The companion notebook (`../experiments/gcn_vs_gat_from_scratch.ipynb`) implements this paper's
`GATLayer` and Kipf & Welling's `GCNLayer` from raw PyTorch tensor ops (no `torch_geometric.nn`),
trains both on Cora with the paper's own hyperparameters, and got:

| | Paper (Table 2) | Notebook run |
|---|---|---|
| GCN test accuracy | 81.5% (Kipf) / 81.4% (GCN-64*) | 81.0% |
| GAT test accuracy | 83.0 ± 0.7% | 83.1% |

Matches within noise of a single run vs. the paper's 100-run average — good sanity check that
the from-scratch implementation is faithful to eq. 1–6.

Where the code lines map to the paper:

- `add_self_loops` + neighborhoods $\mathcal{N}_i$ (including $i$) — §2.1, "these will be exactly
  the first-order neighbors of $i$ (including $i$)."
- `att_src` / `att_dst` split of `att` in `GATLayer.forward` — the two halves of $\vec{a}$ in
  eq. 3, computed separately then summed instead of concatenating $W\vec{h}_i \Vert W\vec{h}_j$
  first (same result, avoids building an `[E, 2F']` tensor).
- `scatter_max_broadcast` + `exp` + `scatter_add`-normalize — eq. 2's softmax, computed edge-wise
  via scatter ops instead of a dense $N \times N$ attention matrix (which the paper's own
  "sparse version" footnote alludes to).
- `heads=8, concat=True` on `gat1`, `heads=1, concat=False` on `gat2` — eq. 5 vs eq. 6, exactly
  the "concatenate on hidden layers, average on the output layer" rule.
- `dropout` applied to `alpha` inside `GATLayer.forward`, not just to node features — the
  paper's "dropout... applied to the normalized attention coefficients" detail from §3.3, which
  is easy to miss and changes what's being regularized (which *edges* get zeroed per step, not
  just which feature dimensions).

---

## 14. One-line summaries to remember

- **The whole paper in one line:** replace GCN's degree-based fixed neighbor weights with a
  learned, softmax-normalized, feature-based attention weight — same aggregation skeleton,
  different source for the weight.
- **Why it's inductive and GCN isn't:** the weight computation never touches anything
  graph-global (Laplacian, degree sequence) — only the two endpoint nodes' own features.
- **Why multi-head:** stabilizes training (same motivation as Transformers); heads can
  specialize in different notions of "relevant neighbor."
- **The cleanest evidence it's the attention doing the work, not just more parameters:**
  Const-GAT (same net, uniform weights) vs GAT — 3.9pp gap on PPI, isolating the mechanism.
- **What it doesn't solve:** receptive field still bounded by depth, no edge features, no
  graph-level pooling — same open problems as GCN, just with better neighbor weighting.
