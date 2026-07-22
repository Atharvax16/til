# LLVM

Paper studies and reproductions. The folder name is historical — nothing here is about the
LLVM compiler. Each subfolder is one paper: the PDF, notes, and any reproduction notebook.

| Folder | Paper | What's in it |
|---|---|---|
| `RAG/` | Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | `RAG_repro.ipynb` (full reproduction), `RAG.ipynb` (from-scratch retrieval intuition), results |
| `SteerViT/` | *SteerViT* | `SteerViT_repro_v2.ipynb` (current), `SteerViT_repro_v1.ipynb` (earlier pass), recap notes, checkpoints, results |
| `NTM/` | Graves et al. 2014, *Neural Turing Machines* | paper only — notes are in the shared file below |
| `MemGPT/` | *MemGPT* | study notes |
| `VoxSight/` | *A Literature Roadmap for VoxSight Recall* | literature roadmap |

`NTM_and_RAG_study_notes.md` sits at this level rather than inside either folder: it is one
continuous document that builds NTM (Part 1) and RAG (Part 3) against each other, and splitting
it would break the argument.

## Running the notebooks

Every notebook uses **paths relative to its own directory**, so run it from where it sits and
the artifacts resolve on their own:

- `RAG/RAG_repro.ipynb` → writes to `RAG/results/rag_repro/`
- `SteerViT/SteerViT_repro_v2.ipynb` → uses `./checkpoints`, `./results`, `./data/train2014`

Two things that are deliberately **not** in git, and why:

- `SteerViT/data/` (~25 GB of COCO) and `SteerViT/refer/` — bulk data and a vendored clone of
  [lichengunc/refer](https://github.com/lichengunc/refer). Fetch them separately.
- `RAG/results/rag_repro/index_15077.npy` (46 MB) — a derived DPR index. The notebook rebuilds
  it if absent and caches it thereafter, so committing it only bloats history.

One MPS note, because it costs an afternoon to rediscover: `RAG_repro.ipynb` sets an explicit
`PYTORCH_MPS_HIGH_WATERMARK_RATIO`. Do not set it to `0.0` — that disables the allocation
ceiling, and an over-allocation then swaps the machine instead of raising a catchable error.
Run one kernel at a time; the weights alone are ~5 GB.
