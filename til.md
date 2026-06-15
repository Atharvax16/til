# learning-journal

Daily notes from what I'm studying — papers, Git/GitHub, AWS. One note, one commit, most days.

## Structure

```
learning-journal/
├── README.md
├── papers/
│   └── 2026-06-15-hallugen.md
├── git/
│   └── 2026-06-15-rebase-vs-merge.md
├── aws/
│   └── 2026-06-15-s3-basics.md
└── templates/
    ├── paper.md
    ├── git.md
    └── aws.md
```

Naming convention: `YYYY-MM-DD-short-slug.md`. The date prefix keeps files sorted and makes it obvious you contributed that day; the slug makes them searchable later.

---

## Daily routine (2 minutes)

1. Copy the relevant template into the right folder.
2. Rename it with today's date and a slug.
3. Fill it in while the thing is fresh.
4. `git add . && git commit -m "notes: <slug>" && git push`

That's it. One commit, genuinely earned.

---

## Template: paper

```markdown
# <Paper title>

**Date:** YYYY-MM-DD
**Link:** <arxiv / DOI>
**Tags:** #diffusion #restoration #xai

## Problem
What gap does this address? (1–2 sentences)

## Approach
The core idea, in my own words.

## Key result
The number or finding that matters.

## Relevance to my work
How this connects to the restoration pipeline / hallucination detection / thesis.

## One thing to remember
The single takeaway.
```

## Template: git

```markdown
# <Concept, e.g. interactive rebase>

**Date:** YYYY-MM-DD
**Tags:** #git

## What it does
Plain explanation.

## When to use it
The situation where this is the right tool.

## Commands
\`\`\`bash
# example I actually ran
\`\`\`

## Gotcha
What tripped me up / what to avoid.
```

## Template: aws

```markdown
# <Service, e.g. S3>

**Date:** YYYY-MM-DD
**Tags:** #aws #storage

## What it is
One-line definition.

## When to use it
Typical use case, especially for ML/AI workloads.

## Hands-on
\`\`\`bash
# CLI snippet or config I tried
\`\`\`

## Gotcha / cost note
Limits, pricing surprises, IAM permissions needed.
```

---

## Setup (one time)

```bash
mkdir learning-journal && cd learning-journal
git init
mkdir papers git aws templates
# add this file as README.md, add the three templates
git add .
git commit -m "init: learning journal"
gh repo create learning-journal --public --source=. --push
```

Then in **GitHub → Settings → Profile**, make sure *"Include private contributions on my profile"* is on if you ever add private repos.
