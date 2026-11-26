# Weighted robustness scores and rankings of the main-experiment networks

> **Directory** `result/scoring`  
> **Data source** `result/criteria/clean` (main experiment: clean-training branch)  
> **Generated** 2026-11-19 15:43 (CST)  
> **Script** `src/eval/scoring.py` (reproduces everything in one run)

## 0. Summary of conclusions

1. Under all four weightings the **top model on the combined scope is always VGG-11** (mean 0.6487, PGD-dominant 0.5277, Gauss-dominant 0.6511, salt-dominant 0.6892).
2. Per task, under **mean weighting** the classification leader is Mix-8 (0.6597) and the regression leader is VGG-11 (0.6489); as soon as the weights favor PGD, the classification leader becomes VGG-11 (0.5116) and the regression leader becomes Res-18 (0.5444), where the regression Res-18 leads VGG-11 by only 0.0007, inside the 2xSE threshold, so the two are in fact tied (see 4.4).
3. Across the four weightings the four models produce only **3 distinct rankings (classification) / 3 (regression)**: the differences concentrate on a few pairs that are close in total score or whose strength lies in a single condition, rather than on a full reshuffle.
4. **Adjacent ranks whose score gap is below the combined 2xSE threshold should be treated as tied** (threshold 0.0009 to 0.0028 on this data; per pair, see section 4.4); only gaps above the threshold are worth stating as a conclusion.

## 1. Metric definition and weight design

### 1.1 Metric

- We use the **whole-network OVR accuracy**: the last row (output layer) of the robustness matrix, averaged across facets. There are 4 one-vs-rest facets for classification and 9 quantile-threshold buckets for regression; this is the "final" column of `result/criteria/clean/<task>.csv`.
- Evaluation is in-sample: 20000 training samples, and only samples where the last-layer projection agrees with the whole-network prediction are kept. It is therefore a **representation-level robustness**, not test-set accuracy.
- Five perturbations: `attack` (PGD adversarial), `gauss` (Gaussian), `salt` (salt-pepper), `move` (translation), `occ` (occlusion), each the mean over 10 repeats.
- **Higher is more robust.** `base` keeps samples where the probe prediction equals the network prediction, so the whole-network value is always 1.0000; the weighted score can therefore also be read as a weighted retention rate.
- We do not use `drop = base - cond`: with base ~ 1 it is only a constant shift of accuracy and ranks identically to accuracy, while accuracy itself is more intuitive and directly comparable across models.

### 1.2 The four weighting schemes

| Scheme | PGD adversarial<br>`attack` | Gaussian noise<br>`gauss` | Salt-pepper noise<br>`salt` | Translation<br>`move` | Occlusion<br>`occ` | Total |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean weighting | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 1.0 |
| PGD-dominant | 0.6 | 0.1 | 0.1 | 0.1 | 0.1 | 1.0 |
| Gauss-dominant | 0.1 | 0.6 | 0.1 | 0.1 | 0.1 | 1.0 |
| Salt-dominant | 0.1 | 0.1 | 0.6 | 0.1 | 0.1 | 1.0 |

Each of the three "dominant" schemes raises its leading condition to **0.6** and gives the remaining four **0.1** each: the leading condition alone carries 60% of the weight, just above the sum of the other four (0.4), so it can change a rank on its own, whereas mean weighting is the only prior-free setting.

## 2. Raw panel: whole-network robustness per condition

Start from the raw scores before weighting. The table lists whole-network OVR accuracy; the final "mean over 5 conditions" column is exactly the mean-weighting score (0.2 per condition).

### Classification (classification)

| Model | base | PGD adversarial | Gaussian noise | Salt-pepper noise | Translation | Occlusion | Mean over 5 conditions |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG-11 | 1.0000 | 0.3747 | 0.6652 | 0.7376 | 0.6739 | 0.7909 | 0.6485 |
| Res-18 | 1.0000 | 0.3525 | 0.6163 | 0.7071 | 0.6859 | 0.8106 | 0.6345 |
| ViT-8 | 1.0000 | 0.3049 | 0.6334 | 0.7015 | 0.6823 | 0.7758 | 0.6196 |
| Mix-8 | 1.0000 | 0.3284 | 0.6773 | 0.7442 | 0.7377 | 0.8108 | 0.6597 |

- PGD adversarial: VGG-11 is highest (0.3747), ViT-8 is lowest (0.3049).
- Gaussian noise: Mix-8 is highest (0.6773), Res-18 is lowest (0.6163).
- Salt-pepper noise: Mix-8 is highest (0.7442), ViT-8 is lowest (0.7015).
- Translation: Mix-8 is highest (0.7377), VGG-11 is lowest (0.6739).
- Occlusion: Mix-8 is highest (0.8108), ViT-8 is lowest (0.7758).

### Regression (regression)

| Model | base | PGD adversarial | Gaussian noise | Salt-pepper noise | Translation | Occlusion | Mean over 5 conditions |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG-11 | 1.0000 | 0.4386 | 0.6419 | 0.7218 | 0.6956 | 0.7467 | 0.6489 |
| Res-18 | 1.0000 | 0.4624 | 0.5868 | 0.6812 | 0.6491 | 0.7532 | 0.6265 |
| ViT-8 | 1.0000 | 0.4240 | 0.5069 | 0.6473 | 0.6958 | 0.7897 | 0.6127 |
| Mix-8 | 1.0000 | 0.4120 | 0.5619 | 0.6290 | 0.7491 | 0.7411 | 0.6186 |

- PGD adversarial: Res-18 is highest (0.4624), Mix-8 is lowest (0.4120).
- Gaussian noise: VGG-11 is highest (0.6419), ViT-8 is lowest (0.5069).
- Salt-pepper noise: VGG-11 is highest (0.7218), Mix-8 is lowest (0.6290).
- Translation: Mix-8 is highest (0.7491), Res-18 is lowest (0.6491).
- Occlusion: ViT-8 is highest (0.7897), Mix-8 is lowest (0.7411).

## 3. Weighted scores and rankings

Cells read `weighted score (rank)`, with rank 1 = most robust. Scores are accuracies on a [0,1] scale and are not renormalized (all conditions share the same scale and the same base). **Scores are comparable only within one scope**: classification, regression and combined measure different facets, so their magnitudes cannot be compared across scopes and only the orderings within each scope are meaningful. The paper table is `score_table.tex` (scores kept to 3 significant figures, recomputed from the pkl at full precision and then rounded).

### 3.1 Classification task

| Model | Mean weighting | PGD-dominant | Gauss-dominant | Salt-dominant |
| --- | :---: | :---: | :---: | :---: |
| VGG-11 | 0.6485 (2) | 0.5116 (1) | 0.6568 (2) | 0.6931 (2) |
| Res-18 | 0.6345 (3) | 0.4935 (3) | 0.6254 (4) | 0.6708 (3) |
| ViT-8 | 0.6196 (4) | 0.4622 (4) | 0.6265 (3) | 0.6605 (4) |
| Mix-8 | 0.6597 (1) | 0.4940 (2) | 0.6685 (1) | 0.7019 (1) |

Leader: Mean weighting -> Mix-8; PGD-dominant -> VGG-11; Gauss-dominant -> Mix-8; Salt-dominant -> Mix-8.

### 3.2 Regression task

| Model | Mean weighting | PGD-dominant | Gauss-dominant | Salt-dominant |
| --- | :---: | :---: | :---: | :---: |
| VGG-11 | 0.6489 (1) | 0.5437 (2) | 0.6454 (1) | 0.6853 (1) |
| Res-18 | 0.6265 (2) | 0.5444 (1) | 0.6066 (2) | 0.6538 (2) |
| ViT-8 | 0.6127 (4) | 0.5184 (3) | 0.5598 (4) | 0.6300 (3) |
| Mix-8 | 0.6186 (3) | 0.5153 (4) | 0.5902 (3) | 0.6238 (4) |

Leader: Mean weighting -> VGG-11; PGD-dominant -> Res-18; Gauss-dominant -> VGG-11; Salt-dominant -> VGG-11.

### 3.3 Both tasks combined

| Model | Mean weighting | PGD-dominant | Gauss-dominant | Salt-dominant |
| --- | :---: | :---: | :---: | :---: |
| VGG-11 | 0.6487 (1) | 0.5277 (1) | 0.6511 (1) | 0.6892 (1) |
| Res-18 | 0.6305 (3) | 0.5190 (2) | 0.6160 (3) | 0.6623 (3) |
| ViT-8 | 0.6162 (4) | 0.4903 (4) | 0.5931 (4) | 0.6453 (4) |
| Mix-8 | 0.6392 (2) | 0.5047 (3) | 0.6294 (2) | 0.6629 (2) |

Leader: Mean weighting -> VGG-11; PGD-dominant -> VGG-11; Gauss-dominant -> VGG-11; Salt-dominant -> VGG-11.

> Combined scope = 0.5 weight on each task's score (equivalent to equal weight per task); standard errors are propagated under an independence approximation.

![rank bump](../../figure/generated/scoring/rank_bump.png)

*Figure: rank trajectories under the four weightings (left to right: mean / PGD-dominant / Gauss-dominant / salt-dominant); the three panels are classification, regression and the combined scope. Flips between close scores (e.g. classification under PGD-dominant) show up as crossings in the plot.*

## 4. How the four rankings differ

### 4.1 Rank comparison and shift

`Δ` is the rank change relative to mean weighting (negative = moved up). The narrower the `rank range`, the less sensitive that model's rank is to the weights.

**Classification task**

| Model | Mean weighting | PGD-dominant | Gauss-dominant | Salt-dominant | ΔPGD | ΔGauss | ΔSalt | Rank range |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG-11 | 2 | 1 | 2 | 2 | -1 | +0 | +0 | 1–2 |
| Res-18 | 3 | 3 | 4 | 3 | +0 | +1 | +0 | 3–4 |
| ViT-8 | 4 | 4 | 3 | 4 | +0 | -1 | +0 | 3–4 |
| Mix-8 | 1 | 2 | 1 | 1 | +1 | +0 | +0 | 1–2 |

**Regression task**

| Model | Mean weighting | PGD-dominant | Gauss-dominant | Salt-dominant | ΔPGD | ΔGauss | ΔSalt | Rank range |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG-11 | 1 | 2 | 1 | 1 | +1 | +0 | +0 | 1–2 |
| Res-18 | 2 | 1 | 2 | 2 | -1 | +0 | +0 | 1–2 |
| ViT-8 | 4 | 3 | 4 | 3 | -1 | +0 | -1 | 3–4 |
| Mix-8 | 3 | 4 | 3 | 4 | +1 | +0 | +1 | 3–4 |

**Both tasks combined**

| Model | Mean weighting | PGD-dominant | Gauss-dominant | Salt-dominant | ΔPGD | ΔGauss | ΔSalt | Rank range |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG-11 | 1 | 1 | 1 | 1 | +0 | +0 | +0 | 1–1 |
| Res-18 | 3 | 2 | 3 | 3 | -1 | +0 | +0 | 2–3 |
| ViT-8 | 4 | 4 | 4 | 4 | +0 | +0 | +0 | 4–4 |
| Mix-8 | 2 | 3 | 2 | 2 | +1 | +0 | +0 | 2–3 |

### 4.2 Rank agreement (Kendall τ)

τ = 1 means the two schemes rank identically, τ = 0 means unrelated, and τ < 0 means reversed. Four models give only 6 model pairs, so τ takes the discrete values 1, 0.667, 0.333, 0.

**Classification task**

| Scheme A | Scheme B | Kendall τ | Concordant : discordant |
| --- | --- | :---: | :---: |
| Mean weighting | PGD-dominant | 0.667 | 5 : 1 |
| Mean weighting | Gauss-dominant | 0.667 | 5 : 1 |
| Mean weighting | Salt-dominant | 1.000 | 6 : 0 |
| PGD-dominant | Gauss-dominant | 0.333 | 4 : 2 |
| PGD-dominant | Salt-dominant | 0.667 | 5 : 1 |
| Gauss-dominant | Salt-dominant | 0.667 | 5 : 1 |

**Regression task**

| Scheme A | Scheme B | Kendall τ | Concordant : discordant |
| --- | --- | :---: | :---: |
| Mean weighting | PGD-dominant | 0.333 | 4 : 2 |
| Mean weighting | Gauss-dominant | 1.000 | 6 : 0 |
| Mean weighting | Salt-dominant | 0.667 | 5 : 1 |
| PGD-dominant | Gauss-dominant | 0.333 | 4 : 2 |
| PGD-dominant | Salt-dominant | 0.667 | 5 : 1 |
| Gauss-dominant | Salt-dominant | 0.667 | 5 : 1 |

**Both tasks combined**

| Scheme A | Scheme B | Kendall τ | Concordant : discordant |
| --- | --- | :---: | :---: |
| Mean weighting | PGD-dominant | 0.667 | 5 : 1 |
| Mean weighting | Gauss-dominant | 1.000 | 6 : 0 |
| Mean weighting | Salt-dominant | 1.000 | 6 : 0 |
| PGD-dominant | Gauss-dominant | 0.667 | 5 : 1 |
| PGD-dominant | Salt-dominant | 0.667 | 5 : 1 |
| Gauss-dominant | Salt-dominant | 1.000 | 6 : 0 |

### 4.3 Which pairs actually flip

Taking mean weighting as the reference, we check pair by pair whether the relative order reverses.

**Classification task**

- PGD-dominant: VGG-11 ↔ Mix-8 (mean: Mix-8 first -> PGD-dominant: VGG-11 first)
- Gauss-dominant: Res-18 ↔ ViT-8 (mean: Res-18 first -> Gauss-dominant: ViT-8 first)
- Salt-dominant: **identical** to the mean-weighting ranking

**Regression task**

- PGD-dominant: VGG-11 ↔ Res-18 (mean: VGG-11 first -> PGD-dominant: Res-18 first); ViT-8 ↔ Mix-8 (mean: Mix-8 first -> PGD-dominant: ViT-8 first)
- Gauss-dominant: **identical** to the mean-weighting ranking
- Salt-dominant: ViT-8 ↔ Mix-8 (mean: Mix-8 first -> Salt-dominant: ViT-8 first)

**Both tasks combined**

- PGD-dominant: Res-18 ↔ Mix-8 (mean: Mix-8 first -> PGD-dominant: Res-18 first)
- Gauss-dominant: **identical** to the mean-weighting ranking
- Salt-dominant: **identical** to the mean-weighting ranking

### 4.4 Score resolution: which rank gaps are meaningful

Propagating the standard error of the 10-repeat mean (SE = s/sqrt(n)) through the sum of squared weights gives the standard error of each score; if the gap between two adjacent ranks is below 2x the combined standard error of the pair, the rank difference cannot be resolved within repeat noise.

| Scope | Scheme | Adjacent ranks | Score gap | 2×SE threshold | Verdict |
| --- | --- | --- | ---: | ---: | --- |
| Classification task | Mean weighting | Mix-8 > VGG-11 | 0.0112 | 0.0017 | resolvable |
| Classification task | Mean weighting | VGG-11 > Res-18 | 0.0140 | 0.0017 | resolvable |
| Classification task | Mean weighting | Res-18 > ViT-8 | 0.0149 | 0.0017 | resolvable |
| Classification task | PGD-dominant | VGG-11 > Mix-8 | 0.0175 | 0.0022 | resolvable |
| Classification task | PGD-dominant | Mix-8 > Res-18 | 0.0005 | 0.0020 | **tied** |
| Classification task | PGD-dominant | Res-18 > ViT-8 | 0.0313 | 0.0020 | resolvable |
| Classification task | Gauss-dominant | Mix-8 > VGG-11 | 0.0117 | 0.0020 | resolvable |
| Classification task | Gauss-dominant | VGG-11 > ViT-8 | 0.0304 | 0.0017 | resolvable |
| Classification task | Gauss-dominant | ViT-8 > Res-18 | 0.0011 | 0.0013 | **tied** |
| Classification task | Salt-dominant | Mix-8 > VGG-11 | 0.0089 | 0.0028 | resolvable |
| Classification task | Salt-dominant | VGG-11 > Res-18 | 0.0222 | 0.0027 | resolvable |
| Classification task | Salt-dominant | Res-18 > ViT-8 | 0.0103 | 0.0027 | resolvable |
| Regression task | Mean weighting | VGG-11 > Res-18 | 0.0224 | 0.0015 | resolvable |
| Regression task | Mean weighting | Res-18 > Mix-8 | 0.0079 | 0.0015 | resolvable |
| Regression task | Mean weighting | Mix-8 > ViT-8 | 0.0059 | 0.0015 | resolvable |
| Regression task | PGD-dominant | Res-18 > VGG-11 | 0.0007 | 0.0015 | **tied** |
| Regression task | PGD-dominant | VGG-11 > ViT-8 | 0.0254 | 0.0017 | resolvable |
| Regression task | PGD-dominant | ViT-8 > Mix-8 | 0.0031 | 0.0018 | resolvable |
| Regression task | Gauss-dominant | VGG-11 > Res-18 | 0.0387 | 0.0017 | resolvable |
| Regression task | Gauss-dominant | Res-18 > Mix-8 | 0.0164 | 0.0015 | resolvable |
| Regression task | Gauss-dominant | Mix-8 > ViT-8 | 0.0304 | 0.0013 | resolvable |
| Regression task | Salt-dominant | VGG-11 > Res-18 | 0.0315 | 0.0026 | resolvable |
| Regression task | Salt-dominant | Res-18 > ViT-8 | 0.0238 | 0.0026 | resolvable |
| Regression task | Salt-dominant | ViT-8 > Mix-8 | 0.0062 | 0.0024 | resolvable |
| Both tasks combined | Mean weighting | VGG-11 > Mix-8 | 0.0095 | 0.0012 | resolvable |
| Both tasks combined | Mean weighting | Mix-8 > Res-18 | 0.0086 | 0.0011 | resolvable |
| Both tasks combined | Mean weighting | Res-18 > ViT-8 | 0.0144 | 0.0011 | resolvable |
| Both tasks combined | PGD-dominant | VGG-11 > Res-18 | 0.0087 | 0.0013 | resolvable |
| Both tasks combined | PGD-dominant | Res-18 > Mix-8 | 0.0143 | 0.0013 | resolvable |
| Both tasks combined | PGD-dominant | Mix-8 > ViT-8 | 0.0144 | 0.0013 | resolvable |
| Both tasks combined | Gauss-dominant | VGG-11 > Mix-8 | 0.0217 | 0.0014 | resolvable |
| Both tasks combined | Gauss-dominant | Mix-8 > Res-18 | 0.0133 | 0.0011 | resolvable |
| Both tasks combined | Gauss-dominant | Res-18 > ViT-8 | 0.0229 | 0.0009 | resolvable |
| Both tasks combined | Salt-dominant | VGG-11 > Mix-8 | 0.0263 | 0.0018 | resolvable |
| Both tasks combined | Salt-dominant | Mix-8 > Res-18 | 0.0006 | 0.0018 | **tied** |
| Both tasks combined | Salt-dominant | Res-18 > ViT-8 | 0.0171 | 0.0019 | resolvable |

4 adjacent rank gaps fall within 2xSE:

- Classification task · PGD-dominant: Mix-8 and Res-18 differ by 0.0005 (threshold 0.0020)
- Classification task · Gauss-dominant: ViT-8 and Res-18 differ by 0.0011 (threshold 0.0013)
- Regression task · PGD-dominant: Res-18 and VGG-11 differ by 0.0007 (threshold 0.0015)
- Both tasks combined · Salt-dominant: Mix-8 and Res-18 differ by 0.0006 (threshold 0.0018)

At these positions the rank is only indicative; describing them as "essentially tied" is safer than claiming "A beats B".

### 4.5 Where the differences come from: how to read them

1. **The classification flip is driven by PGD alone.** Mix-8 ranks first on gauss/salt/move/occ (all four), but only 3 on PGD (0.3284); the strongest model on classification PGD is VGG-11 (0.3747). Once the PGD weight rises to 0.6, Mix-8 moves from rank 1 to rank 2 and VGG-11 moves from rank 2 to rank 1.
2. **PGD-dominant reshuffles regression the most.** The strongest model on PGD is Res-18 (0.4624), lifted to rank 1 by the 0.6 weight; the weakest is Mix-8 (0.4120), pushed down to rank 4 (it is rank 3 under mean weighting).
3. **Gauss- and salt-dominant are comparatively mild.** Under these two conditions the four networks' strength profile follows the overall one, and their τ against mean weighting lies in 0.67-1.00; only the closest pairs flip: Classification · Gauss-dominant: Res-18 ↔ ViT-8 (gap 0.0011); Regression · Salt-dominant: ViT-8 ↔ Mix-8 (gap 0.0062).
4. **Every network has one weak condition**, and weighting only decides whose weakness gets amplified: in classification VGG-11 is weakest on translation (rank 4 of four) and Mix-8 on PGD (rank 3); in regression ViT-8 is weakest on Gaussian (rank 4) and Mix-8 on PGD (rank 4).

## 5. Conclusions and recommendations

- **Use mean weighting as the default**: it assumes no threat model and suits conclusions about overall robustness. On this data the classification order is Mix-8 > VGG-11 > Res-18 > ViT-8 and the regression order is VGG-11 > Res-18 > Mix-8 > ViT-8.
- **Switch to PGD-dominant for white-box adversarial settings**: the conclusions change substantially, with the classification winner moving from Mix-8 to VGG-11 and the regression winner from VGG-11 to Res-18; this regression swap differs by only 0.0007 (within 2xSE), so the more accurate statement is that Res-18 and VGG-11 are tied for first. The two sets of conclusions do not contradict each other; they answer different questions (average robustness vs robustness to the strongest attack).
- **Gauss- and salt-dominant are better treated as ablations**: their τ against mean weighting is 0.67-1.00, so they add little information; if only one scheme is reported, do not substitute them for mean weighting.
- **Report rank gaps below the 2xSE threshold (0.0009 to 0.0028 on this data; see section 4.4) as tied**; otherwise a different random seed will flip them.
- **The combined scope gives the most stable conclusion**: VGG-11 ranks first under all four weightings, so its lead does not depend on the weight choice.

## Appendix A: metric sensitivity - what changes under the layer mean

The main metric uses the whole network (output layer). Switching to the layer mean (average over the hidden-layer rows, excluding the whole-network last row, consistent with the `layer_mean` column of `criteria/clean/<task>.csv`) changes the rankings markedly:

**Classification (layer-mean metric)**

| Model | Mean weighting | PGD-dominant | Gauss-dominant | Salt-dominant |
| --- | :---: | :---: | :---: | :---: |
| VGG-11 | 0.6301 (3) | 0.5442 (4) | 0.6192 (2) | 0.6484 (2) |
| Res-18 | 0.6242 (4) | 0.5559 (3) | 0.6116 (3) | 0.6412 (3) |
| ViT-8 | 0.6304 (2) | 0.5695 (2) | 0.6109 (4) | 0.6394 (4) |
| Mix-8 | 0.6480 (1) | 0.5939 (1) | 0.6221 (1) | 0.6653 (1) |

**Regression (layer-mean metric)**

| Model | Mean weighting | PGD-dominant | Gauss-dominant | Salt-dominant |
| --- | :---: | :---: | :---: | :---: |
| VGG-11 | 0.6327 (3) | 0.5532 (3) | 0.6151 (2) | 0.6474 (2) |
| Res-18 | 0.6347 (2) | 0.5620 (2) | 0.6130 (3) | 0.6458 (3) |
| ViT-8 | 0.6197 (4) | 0.5412 (4) | 0.5948 (4) | 0.6296 (4) |
| Mix-8 | 0.6741 (1) | 0.5957 (1) | 0.6800 (1) | 0.6867 (1) |

Against the main text (mean weighting, whole-network metric): regression rank 1 moves from VGG-11 to Mix-8; in classification VGG-11 moves from rank 2 to rank 3.
The reason is that the layer mean also averages in the shallow layers, whose accuracy is generally low, which favors networks with better cross-layer consistency.
The two metrics answer different questions. This directory uses the **whole network** throughout, because network robustness means how the output layer behaves under perturbation; the layer mean is better suited to discussing how decodable the representation is at each layer. **Do not mix conclusions across metrics.**

## Appendix B: files and reproduction

| File | Content |
| --- | --- |
| `score_report.md` | This document: scores, rankings, and how the four rankings differ |
| `src/eval/scoring.py` | Computation script (read pkl -> compute scores -> emit tables and figures) |
| `score_table.tex` | Booktabs paper table (booktabs + multirow, scores to 3 significant figures) |
| `src/eval/make_score_table.py` | Generates the .tex above |
| `scores_wide.csv` | Wide table: whole-network accuracy per model x condition plus score / SE / rank for the 4 schemes |
| `scores_long.csv` | Long table: `scope, scheme, model, score, se, rank` |
| `ranking_diff.csv` | Rank comparison, shift relative to mean weighting, rank range |
| `kendall_tau.csv` | Pairwise rank agreement |
| `figure/generated/scoring/rank_bump.png` | Rank-trajectory figure |

Reproduction:

```bash
cd <repository root>
python -m src.eval.scoring    # report + csv + figures
python -m src.eval.make_score_table     # paper booktabs table .tex
```

Dependencies: `result/criteria/clean/<task>/<Model>_robustness.pkl`, `_std.pkl`, and `criteria/clean/<task>.csv` (the csv serves only for cross-checking; the script verifies that the pkl and the csv `final` column agree to within 5e-4).

> Note: the noise-training branch `result/criteria/noise` provides only the three conditions `base/gauss/salt` and lacks `attack/move/occ`, so the same 5-condition weights cannot be applied and it takes no part in this scoring; to compare it, fill in the missing conditions first or define a separate two-condition weighting.
