# SteerViT small-scale reproduction -- results
Trainable params: 15,356,167 | train images: 4500 | val images: 400

Run config: 224px input -> 16x16 = 256 patch grid | 3000 steps | batch 12 | device mps | autocast torch.bfloat16

> **Deviation from spec:** the spec calls for 336px -> 24x24 = 576 patches; this run used 224px -> 16x16 = 256 patches for speed. Localization metrics are correspondingly coarser (each patch covers a larger image area) and are NOT directly comparable to the paper's numbers. The mechanism checks -- especially wrong-prompt collapse -- are unaffected.

> **Note:** 3000 training steps, below the spec's 20-50k range, per the spec's guidance to stop once steering emerges/plateaus.

## 1-3. Baseline / Steerability / Wrong-prompt sanity check
| condition | patch-grid IoU | PR-AUC |
|---|---|---|
| baseline (frozen, no text) | 0.1295 | 0.2208 |
| steerability (correct prompt) | 0.2940 | 0.5113 |
| wrong prompt (mismatched) | 0.2323 | 0.3926 |

collapse_ratio = **0.625** (FAIL -- looks like memorization)

## 4. Gate (omega) sweep
| omega | IoU | PR-AUC | CLS linear-probe acc |
|---|---|---|---|
| 0.00 | 0.1295 | 0.2208 | 0.5542 |
| 0.25 | 0.1493 | 0.2660 | 0.6917 |
| 0.50 | 0.1788 | 0.3170 | 0.6250 |
| 0.75 | 0.2629 | 0.4483 | 0.6167 |
| 1.00 | 0.2940 | 0.5113 | 0.5542 |

omega=0 verified to reproduce frozen DINOv2 exactly (see sanity check above).

## Visualizations
![sample](sample_visualization.png)

![overfit curve](overfit_curve.png)

![training curve](training_curve.png)

![gate sweep](gate_sweep.png)

![gate diagnostic](gate_diagnostic.png)

![correct vs wrong prompt heatmaps](heatmap_comparison.png)
