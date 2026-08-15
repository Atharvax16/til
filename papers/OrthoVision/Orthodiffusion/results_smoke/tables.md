### M4 — linear probe (AUROC, mean ± std over seeds [0])

| arm            | t   | val AUROC     | test AUROC    | test AUPRC    |
|----------------|-----|---------------|---------------|---------------|
| diffusion      | 100 | 0.878 ± 0.000 | 0.890 ± 0.000 | 0.909 ± 0.000 |
| mae            | 0   | 0.957 ± 0.000 | 0.885 ± 0.000 | 0.880 ± 0.000 |
| jepa           | 0   | 0.885 ± 0.000 | 0.853 ± 0.000 | 0.852 ± 0.000 |
| random         | 0   | 0.935 ± 0.000 | 0.896 ± 0.000 | 0.904 ± 0.000 |
| supervised-100 | 0   | 0.920 ± 0.000 | 0.904 ± 0.000 | 0.915 ± 0.000 |
| supervised-10  | 0   | 0.751 ± 0.000 | 0.706 ± 0.000 | 0.742 ± 0.000 |

### M5 — diffusion probe AUROC by timestep

| t   | val mid2      | test mid2     |
|-----|---------------|---------------|
| 10  | 0.928 ± 0.000 | 0.931 ± 0.000 |
| 100 | 0.878 ± 0.000 | 0.890 ± 0.000 |

### M6 — FINAL: test metrics, mean ± std over seeds [0] (selection on val only)

| arm            | feature site | probe AUROC   | probe AUPRC   | finetune AUROC | finetune AUPRC |
|----------------|--------------|---------------|---------------|----------------|----------------|
| diffusion      | t=10/mid2    | 0.931 ± 0.000 | 0.940 ± 0.000 | 0.668 ± 0.000  | 0.626 ± 0.000  |
| mae            | t=0          | 0.885 ± 0.000 | 0.880 ± 0.000 | 0.810 ± 0.000  | 0.811 ± 0.000  |
| jepa           | t=0          | 0.853 ± 0.000 | 0.852 ± 0.000 | 0.648 ± 0.000  | 0.711 ± 0.000  |
| random-init    | t=0          | 0.896 ± 0.000 | 0.904 ± 0.000 | —              | —              |
| supervised-100 | t=0          | —             | —             | 0.904 ± 0.000  | 0.915 ± 0.000  |
| supervised-10  | t=0          | —             | —             | 0.706 ± 0.000  | 0.742 ± 0.000  |


encoder params (identical across arms): 4,058,656

pretrain budget: 40 steps @ batch 128

val-selected diffusion operating point: t=10, block=mid2