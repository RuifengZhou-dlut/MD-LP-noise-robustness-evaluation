# 10-draw noise experiment: repeatability vs similarity to the clean weights

> 8 sites × 8 noise levels × 10 independent draws × 4 OVR facets. `pairwise` compares the 10 noisy weights only with each other; `cos_clean` compares the noisy weights with the clean full-sample weights.


## Per site: average over the 8 rho values / value at rho=100%

| Site | d | pair exact | pair diag | clean exact | clean diag |
|---|---:|---:|---:|---:|---:|
| VGG_L3 | 4096 | 0.747 | 0.990 | 0.790 | 0.992 |
| VGG_L5 | 2048 | 0.843 | 0.999 | 0.786 | 0.995 |
| VGG_L7 | 1024 | 0.683 | 0.802 | 0.613 | 0.870 |
| VGG_L9 | 256 | 0.957 | 1.000 | 0.212 | 0.996 |
| RES_L3 | 4096 | 0.762 | 0.995 | 0.788 | 0.996 |
| RES_L4 | 4096 | 0.745 | 0.998 | 0.768 | 0.997 |
| RES_L6 | 2048 | 0.883 | 0.999 | 0.799 | 0.995 |
| RES_L8 | 1024 | 0.928 | 0.998 | 0.751 | 0.990 |

![pairwise](../../../../figure/add_exp/probe_stability/weight_vs_noise/exp3_noise_pairwise_stability.png)

![clean](../../../../figure/add_exp/probe_stability/weight_vs_noise/noise_vs_clean_supplementary.png)


## Per-site values at rho=100%

| Site | pair exact | pair diag | clean exact | clean diag |
|---|---:|---:|---:|---:|
| VGG_L3 | 0.367 | 0.985 | 0.467 | 0.986 |
| VGG_L5 | 0.643 | 0.998 | 0.470 | 0.990 |
| VGG_L7 | 0.306 | 0.371 | 0.324 | 0.591 |
| VGG_L9 | 0.966 | 1.000 | -0.004 | 0.992 |
| RES_L3 | 0.422 | 0.993 | 0.473 | 0.993 |
| RES_L4 | 0.384 | 0.996 | 0.419 | 0.994 |
| RES_L6 | 0.738 | 0.999 | 0.503 | 0.991 |
| RES_L8 | 0.839 | 0.996 | 0.447 | 0.981 |