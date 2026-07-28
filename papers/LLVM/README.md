# LLVM

Paper studies and reproductions. The folder name is historical — nothing here is about the
LLVM compiler. Each subfolder is one paper: the PDF, notes, and any reproduction notebook.

| Folder | Paper | What's in it |
|---|---|---|
| `RAG/` | Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | `RAG_repro.ipynb` (full reproduction), `RAG.ipynb` (from-scratch retrieval intuition), results |
| `SteerViT/` | *SteerViT* | `SteerViT_repro_v2.ipynb` (current), `SteerViT_repro_v1.ipynb` (earlier pass), recap notes, checkpoints, results |
| `NTM/` | Graves et al. 2014, *Neural Turing Machines* | paper only — notes are in the shared file below |
| `MemGPT/` | *MemGPT* | study notes |
| `VoxSight/` | *A Literature Roadmap for VoxSight Recall* | roadmap, `problem_statement.md`, `voxsight_recall.ipynb` (original study on EPIC-KITCHENS-100), results |

`NTM_and_RAG_study_notes.md` sits at this level rather than inside either folder: it is one
continuous document that builds NTM (Part 1) and RAG (Part 3) against each other, and splitting
it would break the argument.

## Running the notebooks

Every notebook uses **paths relative to its own directory**, so run it from where it sits and
the artifacts resolve on their own:

- `RAG/RAG_repro.ipynb` → writes to `RAG/results/rag_repro/`
- `SteerViT/SteerViT_repro_v2.ipynb` → uses `./checkpoints`, `./results`, `./data/train2014`
- `VoxSight/voxsight_recall.ipynb` → uses `./annotations`, `./data`, `./features`, `./checkpoints`, `./results`

`VoxSight/` is the one folder here that is not a reproduction — it is an original study built on the
roadmap's §8.2 question, measuring how often an episodic-memory assistant's "last seen" answer is
already wrong about the present. Read `VoxSight/problem_statement.md` first; it defines the event
schema and the hypotheses the notebook tests.

Things that are deliberately **not** in git, and why:

- `SteerViT/data/` (~25 GB of COCO) and `SteerViT/refer/` — bulk data and a vendored clone of
  [lichengunc/refer](https://github.com/lichengunc/refer). Fetch them separately.
- `RAG/results/rag_repro/index_15077.npy` (46 MB) — a derived DPR index. The notebook rebuilds
  it if absent and caches it thereafter, so committing it only bloats history.
- `VoxSight/annotations/` (89 MB) — an upstream clone carrying its own `.git`:
  `git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations VoxSight/annotations`
- `VoxSight/data/` (~19 GB of EPIC-KITCHENS RGB frames, 11 videos) and `VoxSight/features/` —
  bulk data and derived CLIP features. The notebook's feature cell is per-video and idempotent, so
  a restart re-derives only what is missing. Frame tars come from
  `https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/<P>/rgb_frames/<video_id>.tar`.

One MPS note, because it costs an afternoon to rediscover: `RAG_repro.ipynb` sets an explicit
`PYTORCH_MPS_HIGH_WATERMARK_RATIO`. Do not set it to `0.0` — that disables the allocation
ceiling, and an over-allocation then swaps the machine instead of raising a catchable error.
Run one kernel at a time; the weights alone are ~5 GB.
