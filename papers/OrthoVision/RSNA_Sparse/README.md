# Sparse Anatomical Memory

Interpreting frozen BioMedCLIP representations of knee imaging with a sparse autoencoder, and
compressing the resulting sparse features into a question-aware memory for anatomical QA.

Implementation of `rsna_sparse_memory_linkedin_outline.pdf` — Stages A–E plus the evaluation plan,
in one runnable notebook: **`sparse_anatomical_memory.ipynb`**.

This is a **conceptual prototype**, not a clinically validated diagnostic system. Every feature
label it produces is a *candidate* concept until a qualified reader confirms it.

## Run it

A virtualenv is already set up at `.venv` (Python 3.11, torch with MPS). To use it:

```bash
.venv/bin/python -m ipykernel install --user --name rsna-sparse --display-name "RSNA Sparse"
.venv/bin/jupyter notebook sparse_anatomical_memory.ipynb
```

From scratch instead:

```bash
uv venv --python 3.11 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

Then `Run All`. Cell 2 installs anything still missing into the running kernel.

**First run on MRNet:** roughly 30–50 min on an M-series MacBook — slicing 3,750 volumes to PNG
(~10–20 min, disk-bound), downloading BioMedCLIP, encoding ~11k images (~2–4 min on MPS), training
the SAE on cached embeddings (~30 s), analysis. Re-runs reuse the cached images and embeddings and
take under a minute to reach the analysis. No overnight run required.

Environment overrides: `SAM_SMOKE=1` (tiny run), `SAM_ENCODER=stub` (no model download, offline),
`SAM_JEPA=1` (also run the section 8 appendix), `SAM_DATA=synthetic` (fall back to the synthetic
corpus), `SAM_REAL_DIR=<path>` (look for the dataset elsewhere), `SAM_ROOT=<path>` (write
`data/`, `artifacts/`, `reports/` under another root, to try a corpus without clobbering a run).

## Data

The notebook runs on **real MRNet knee MRI** (`cfg.data_source = "real"`, the default). MRNet is
access-controlled: sign the research-use agreement at
<https://stanfordmlgroup.github.io/competitions/mrnet/> and Stanford emails a download link.

```bash
python scripts/fetch_mrnet.py '<emailed-url>'      # or: path/to/MRNet-v1.0.zip
```

That unpacks, flattens the archive's own top-level folder, and verifies the layout under
`data/raw/`. Both the `train` (1,130 cases) and `valid` (120 cases) releases are used as corpus —
they are disjoint case ranges, and the notebook makes its own patient-level split. Three planes per
case, three slices per volume ≈ 11k images. Volumes with no label row are dropped, not silently
counted as negative. If the dataset is missing the notebook stops in cell 5 with these
instructions rather than quietly substituting synthetic data.

MRNet labels are coarse — `abnormal`, `acl`, `meniscus` — and the release carries **no pulse
sequence or scanner metadata**, so on real data the only nuisance attribute available to section 5
is the imaging plane. The confounder analysis is correspondingly weaker than it is on synthetic
data; see below.

### The synthetic control corpus (`SAM_DATA=synthetic`)

Still available, and still the only place the interpretability claims are falsifiable. It renders
known generative factors — pathology labels *and* nuisance factors (view, pulse sequence,
scanner), with effusion and marrow edema deliberately more visible on T2/PD than T1, and
correlated findings (an ACL tear raises the chance of effusion).

Because that ground truth includes the confounders, the synthetic run can *test* the outline's
warning that a feature correlating with a label may be encoding acquisition protocol rather than
pathology. On real MRNet that warning goes back to being an untested caveat — the metadata needed
to check it was never released. Worth stating plainly in any write-up of the real-data results.

Splits are patient-level throughout; the notebook asserts zero patient overlap.

## What it produces

| Output | Contents |
|---|---|
| `reports/results.csv` | every headline number in one table |
| `reports/candidate_features.csv` | top features with candidate concepts and best-matching attributes |
| `reports/qa_sweep.csv` | full QA accuracy × memory-size × policy sweep |
| `reports/limitations.txt` | limitations, auto-filled with this run's numbers |
| `reports/figures/` | dataset sample, training curves, top-activating grids, selectivity, memory trade-off, one-slide summary |
| `artifacts/` | cached embeddings and the trained SAE |

## Notes on the method

Two places where the notebook deviates from a naive reading of the outline, both flagged inline:

1. **The sparsity multiplier λ is adapted during training** (dual ascent on a target L0). A fixed λ
   does not transfer between encoders — the same value that gives L0 ≈ 30 on one embedding gives
   L0 ≈ 1000 on another. The objective is unchanged; only the multiplier is scheduled.
2. **Selectivity is measured by precision@k and monosemanticity, not AUROC.** AUROC is
   structurally capped for sparse units: a feature firing on 5 % of images cannot exceed 0.625
   against a 20 %-prevalence label no matter how clean it is, so an AUROC comparison against dense
   dimensions is rigged before it starts. Section 5.1 derives the ceiling and reports it.

Controls included so the interpretability claims are falsifiable: a width- and L0-matched random
feature bank, a shuffled-label null, and nuisance attributes (view/protocol/scanner) scored
alongside the pathology labels.
