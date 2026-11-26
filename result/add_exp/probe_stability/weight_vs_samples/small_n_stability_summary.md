# Stability cosine of the 10 repeated weights (8 sites)

> Stability cosine = mean over the 45 pairs `cos(w_i,w_j)` of the weights from the 10 independent draws at the same (site, n/d, mode); averaged over the 4 OVR facets. **Never compared with the full-sample weights.**


## Per site: average over the 13 ratios / value at n/d=1.0

| Site | d | exact mean ± std | diag mean ± std | exact @1.0 | diag @1.0 |
|---|---:|---:|---:|---:|---:|
| VGG_L3 | 4096 | 0.051 ± 0.045 | 0.929 ± 0.022 | -0.005 | 0.931 |
| VGG_L5 | 2048 | 0.062 ± 0.030 | 0.987 ± 0.002 | -0.002 | 0.988 |
| VGG_L7 | 1024 | 0.039 ± 0.098 | 0.146 ± 0.166 | 0.021 | 0.207 |
| VGG_L9 | 256 | 0.157 ± 0.145 | 0.976 ± 0.017 | 0.017 | 0.981 |
| RES_L3 | 4096 | 0.060 ± 0.037 | 0.966 ± 0.011 | -0.002 | 0.972 |
| RES_L4 | 4096 | 0.056 ± 0.020 | 0.985 ± 0.001 | 0.000 | 0.987 |
| RES_L6 | 2048 | 0.097 ± 0.028 | 0.992 ± 0.002 | 0.000 | 0.993 |
| RES_L8 | 1024 | 0.143 ± 0.097 | 0.839 ± 0.075 | -0.008 | 0.837 |

- 8 sites × 13 ratios: exact mean **0.083**, diag mean **0.852**; cells with diag > exact: **98.1%**.
- Excluding VGG_L7: exact **0.089**, diag **0.953**; cells with diag > exact: **100.0%**.

![stability](../../../../figure/add_exp/probe_stability/weight_vs_samples/exp3_weight_stability_pairwise.png)
