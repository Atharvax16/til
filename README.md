# til — Today I Learned

A learning journal. Short notes from what I study — papers, tools, cloud, etc. — kept in
plain Markdown so they stay searchable and easy to revisit. One note, one commit, most days.

## Structure

```
til/
├── README.md
└── papers/
    └── Quantum_chemistry/
        ├── README.md          ← summary + what each paper upgraded
        ├── Neural_MPNN.pdf
        ├── SchNet.pdf
        ├── DimeNet.pdf
        └── NequIP.pdf
```

Each topic folder has its own `README.md` that summarizes the material and how the pieces
relate, with the source files (PDFs, snippets) alongside it.

## Topics

- **[papers/Quantum_chemistry](papers/Quantum_chemistry/README.md)** — graph neural
  networks for predicting molecular properties: MPNN → SchNet → DimeNet → NequIP, and what
  each one improved over the last.

## Conventions

- Notes are Markdown. Files are named so they sort and search well
  (`YYYY-MM-DD-short-slug.md` for dated notes, or a clear topic name).
- Every topic folder gets a `README.md` that ties its contents together.
- Commit messages: `notes: <what I learned>`.

## Why keep this

Writing a study down in my own words is the cheapest way to actually remember it, and the
folder doubles as a quick reference I can grep later.
