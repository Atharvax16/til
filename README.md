# til — Today I Learned

A learning journal. Short notes from what I study — papers, tools, cloud, etc. — kept in
plain Markdown so they stay searchable and easy to revisit. One note, one commit, most days.

## Structure

```
til/
├── README.md
└── papers/
    ├── Quantum_chemistry/
    │   ├── README.md          ← summary + what each paper upgraded
    │   ├── paper/             ← MPNN / SchNet / DimeNet / NequIP PDFs
    │   └── notes/             ← figures
    ├── Attention/
    │   ├── paper/             ← Attention_is_all_you_need.pdf
    │   └── notes/             ← Attention_is_all_we_need.md (ground-up Transformer guide)
    └── Generative_Adversarial_Nets/
        ├── README.md          ← summary + how the notebooks relate
        ├── paper/             ← generative-adversarial-nets-Paper.pdf
        ├── notes/             ← GAN_paper_study_notes.md
        └── experiments/       ← the .ipynb notebooks (data/ auto-downloaded, gitignored)
```

Each topic folder splits its contents into `paper/` (source PDFs), `notes/` (Markdown
write-ups + figures) and, where there's code, `experiments/` (notebooks). A `README.md` at
the topic root summarizes the material and ties the pieces together.

## Topics

- **[papers/Quantum_chemistry](papers/Quantum_chemistry/README.md)** — graph neural
  networks for predicting molecular properties: MPNN → SchNet → DimeNet → NequIP, and what
  each one improved over the last.
- **[papers/Attention](papers/Attention/notes/Attention_is_all_we_need.md)** — a ground-up guide
  to *Attention Is All You Need*: Q/K/V, scaled dot-product and multi-head attention, and
  the full Transformer architecture worked from the numbers up.
- **[papers/Generative_Adversarial_Nets](papers/Generative_Adversarial_Nets/README.md)** — the
  original GAN (Goodfellow et al., 2014) and the **non-saturating loss trick**, with notebooks
  that make the saturating-vs-non-saturating gradient gap visible and show why Adam hides it
  while SGD reveals it (on an 8-blob ring and on MNIST).

## Conventions

- Notes are Markdown. Files are named so they sort and search well
  (`YYYY-MM-DD-short-slug.md` for dated notes, or a clear topic name).
- Every topic folder gets a `README.md` that ties its contents together.
- Commit messages: `notes: <what I learned>`.

## Why keep this

Writing a study down in my own words is the cheapest way to actually remember it, and the
folder doubles as a quick reference I can grep later.
