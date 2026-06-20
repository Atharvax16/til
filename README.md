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
    │   ├── Neural_MPNN.pdf
    │   ├── SchNet.pdf
    │   ├── DimeNet.pdf
    │   └── NequIP.pdf
    └── Attention/
        ├── Attention_is_all_we_need.md   ← ground-up guide to the Transformer
        └── Attention_is_all_you_need.pdf
```

Each topic folder has its own `README.md` that summarizes the material and how the pieces
relate, with the source files (PDFs, snippets) alongside it.

## Topics

- **[papers/Quantum_chemistry](papers/Quantum_chemistry/README.md)** — graph neural
  networks for predicting molecular properties: MPNN → SchNet → DimeNet → NequIP, and what
  each one improved over the last.
- **[papers/Attention](papers/Attention/Attention_is_all_we_need.md)** — a ground-up guide
  to *Attention Is All You Need*: Q/K/V, scaled dot-product and multi-head attention, and
  the full Transformer architecture worked from the numbers up.

## Conventions

- Notes are Markdown. Files are named so they sort and search well
  (`YYYY-MM-DD-short-slug.md` for dated notes, or a clear topic name).
- Every topic folder gets a `README.md` that ties its contents together.
- Commit messages: `notes: <what I learned>`.

## Why keep this

Writing a study down in my own words is the cheapest way to actually remember it, and the
folder doubles as a quick reference I can grep later.
