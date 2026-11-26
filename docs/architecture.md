# Architecture and Directory Guide

This page explains what each directory is responsible for, how the modules depend on each other,
and how one evaluation flows through the repository. 

## 1. Functional top level

```text
github_2026-11-19/
├── src/                         # code
│   ├── common/ data/ noise/ models/ probe/ eval/ viz/ pipelines/
│   └── add_exp/                 # add-on experiments grouped by paper subsection
├── model/                       # 20 .h5 weights
│   ├── classification/{clean,noise_trained}/
│   ├── classification_full/clean/
│   └── regression/{clean,noise_trained}/
├── dataset/                     # clean splits and manifests; noise/features generated on demand
│   ├── clean/{classification,regression}/
│   └── manifests.json
├── result/                      # probes, criteria, SE, scoring, add-on data
│   ├── probe/ criteria/ repeat_raw/ standard_error/ scoring/
│   └── add_exp/
├── figure/                      # paper PDFs, PNG previews, manifest
│   ├── pdf/  png/  manifest.json
│   └── generated/               # runtime output, gitignored
├── docs/                        # English documentation
├── scripts/                     # run_main.sh / run_addons.sh / check.sh
├── tools/                       # validation, seeds, figure-preview tool
├── README.md
├── requirements.txt
└── .gitignore  .gitattributes
```

The top level is organised by **function**, not by experiment number. Add-on experiments extend the
same functional tree with `src/add_exp/<group>/` sub-folders.

| Directory | Content | Notes |
|---|---|---|
| `src/` | training, noise, probes, evaluation, plotting, pipelines, add-ons | see Sections 2.1 and 2.1.1 |
| `model/` | `.h5` weights of the four backbones | `clean` and `noise_trained`; full CIFAR only `clean` |
| `dataset/` | clean splits and `manifests.json` | `noise/`, `features/`, full CIFAR-10 are generated on demand and not committed |
| `result/` | probe weights, criteria, raw repeats, SE, scoring, add-on data | see Section 2.4 |
| `figure/` | paper PDF panels, PNG previews, manifest | only paper-used figures are kept; runtime figures go to `figure/generated/` |
| `docs/` | architecture, experiments, data/results, reproduction | English, see Sections 2.6–2.8 |
| `scripts/` | `run_main.sh`, `run_addons.sh`, `check.sh` | one-command runs and smoke checks |
| `tools/` | `check_repo.py`, `list_seeds.py`, `make_figure_previews.py`, `seeds.json` | validation, seed inventory, preview generation |
| root | `README.md`, `requirements.txt`, `.gitignore`, `.gitattributes` | entry points and dependencies |

## 2. Repository layout

### 2.1 `src/` modules
| Module | Responsibility | Key files |
|---|---|---|
| `src/common` | paths, task configuration, model registry, seeding | `config.py`, `seeding.py` |
| `src/data` | dataset readers and raw-data builders | `datasets.py`, `build_dataset.py` |
| `src/noise` | Gaussian / salt-pepper / motion-blur / occlusion noise, PGD attacks | `perturbations.py`, `attacks.py`, `build_noise.py` |
| `src/models` | four backbones, weight loading, training loop and optimisers | `network.py`, `loader.py`, `trainer.py` |
| `src/probe` | MD-LP weights/bias, streaming evaluation, susceptible-sample selection, legacy cache I/O | `weights.py`, `streaming.py`, `evaluate.py`, `cache.py` |
| `src/eval` | SE aggregation, SE heatmap, weighted scores and rankings | `aggregate.py`, `se_heatmap.py`, `scoring.py`, `make_score_table.py` |
| `src/viz` | curves, radar charts, paper panels, heatmaps | `clean_plots.py`, `noise_plots.py`, `panels*.py` |
| `src/pipelines` | command-line entry points | `run_train.py`, `run_probe_eval.py`, `run_full_cifar10.py`, `plot_clean.py`, `plot_noise.py` |
| `src/add_exp/common` | shared add-on utilities: eight sites, feature cache, OVR, CKA/SVCCA/t-SNE, plot style | `sites.py`, `features.py`, `ovr.py`, `mdlp.py`, `metrics.py` |
| `src/add_exp/<group>` | add-ons grouped by paper subsubsection (see below) | each group folder |

#### 2.1.1 Add-on package layout
If a group has several experiments they live in subfolders; otherwise the scripts sit directly in
the group folder:

```text
src/add_exp/probe_stability/
├── __init__.py
├── weight_vs_samples/      # compute_stability.py, plot_stability.py
└── weight_vs_noise/        # compute_stability.py, plot_stability.py
```

Inputs are always `dataset/features/features4_<Model>.npz` (built by
`python -m src.add_exp.common.features`); outputs always go to `result/add_exp/<group>/`, figures to
`figure/generated/add_exp/<group>/`.

| Group folder | Paper | Main scripts | Outputs |
|---|---|---|---|
| `probe_efficiency` | 5.2.1 | `benchmark_ovr.py`, `bench_one_ovr.py` | `result/add_exp/probe_efficiency/benchmark_ovr.json` |
| `probe_stability/weight_vs_samples` | 5.2.2 | `compute_stability.py`, `plot_stability.py` | `result/add_exp/probe_stability/weight_vs_samples/` |
| `probe_stability/weight_vs_noise` | 5.2.2 | same | `result/add_exp/probe_stability/weight_vs_noise/` |
| `similarity_comparison` | 5.4.2 | `compute_rotation.py`, `plot_rotation.py` | `result/add_exp/similarity_comparison/` |
| `s_sensitivity` | 5.4.3 | `scan_s.py`, `make_figure.py`, `make_rank_table*.py` | `result/add_exp/s_sensitivity/`, `figure/generated/add_exp/s_sensitivity/` |

### 2.2 Datasets
```text
dataset/
├── clean/
│   ├── classification/data.npz
│   └── regression/parts/part-000.npz  part-001.npz
├── manifests.json
├── noise/                       # generated on demand, not committed
└── features/                    # add-on feature cache, generated on demand
```

- `dataset/clean/<task>/`: clean paper splits. Classification is `data.npz`; regression uses
  `parts/part-000.npz`, `part-001.npz` (sharded to stay below 100 MB per file).
- `dataset/noise/<condition>/<task>/<repeat>.npy`: corruption noise, `condition` in
  `gauss / salt / move / occ`, `repeat` is a two-digit group index (`00`–`09`).
- `dataset/noise/attack/<task>/<repeat>/<Model>_pgd.npy`: PGD samples.
- `dataset/features/features4_<Model>.npz`: hidden-layer feature cache for add-ons (generated, not committed).

### 2.3 Models
```text
model/
├── classification/{clean,noise_trained}/VGG_11.h5 ...
├── classification_full/clean/VGG_11.h5 ...
└── regression/{clean,noise_trained}/VGG_11.h5 ...
```

`clean` is standard training; `noise_trained` is Gaussian+salt augmented training used by the
reliability comparison.

### 2.4 Results
```text
result/
├── probe/<exp>/<task>/<Model>/          # _w.npy, _b.npy, probe_acc.pkl
├── criteria/<exp>/<task>/               # robustness/std/se/drop pkl
├── criteria/<exp>/<task>.csv            # summary
├── repeat_raw/<task>/                   # per-repeat .npz
├── standard_error/                      # SE / STD CSVs
├── scoring/                             # weighted score tables
└── add_exp/<group>/                     # add-on data
```

- Probes: `result/probe/<exp>/<task>/<Model>/<layer>_w.npy`, `_b.npy`, `probe_acc.pkl`.
- Criteria: `result/criteria/<exp>/<task>/<Model>_{robustness,std,se,drop}.pkl`; summary
  `result/criteria/<exp>/<task>.csv`.
- Matrix shape: `(n_layers + 1, n_facets)`; the last row is the whole network. Classification has
  4 (or 10) OVR facets; regression has 9 quantile facets.
- Per-repeat matrices: `result/repeat_raw/<task>/<Model>_<condition>.npz` (`stack`, `final`).

### 2.5 Figures
```text
figure/
├── pdf/
│   ├── panels/                 # original paper PDF panels (131)
│   └── s-1.pdf  s-2.pdf
├── png/
│   ├── sample_selection_noise.png
│   ├── weight_similarity_sampling.png
│   ├── weight_stability_noise.png
│   ├── classification_overall.png
│   ├── classification_per_class.png
│   ├── classification_full_cifar10.png
│   ├── regression_overall.png
│   ├── regression_thresholds_1_5.png
│   ├── regression_thresholds_6_9.png
│   ├── standard_error_heatmap.png
│   ├── noise_trained_radar.png
│   ├── classification_noise_metric.png
│   ├── regression_noise_metric.png
│   └── susceptible_ratio_sensitivity.png
├── manifest.json               # figure id, caption, panel paths
└── generated/                  # runtime output, gitignored
```

- `figure/pdf/` is authoritative for typesetting;
- `figure/png/` is for GitHub/Markdown preview only;
- `figure/manifest.json` maps each paper figure to its PDF panels;
- previews are produced by `tools/make_figure_previews.py`, independently of the paper build;
- plotting scripts write runtime figures under `figure/generated/`, which is not committed.

### 2.6 Documentation

```text
docs/
├── architecture.md
├── experiments.md
├── data_and_results.md
└── reproduction.md
```

- `docs/architecture.{zh,en}.md`: repository layout, module responsibilities, data flow;
- `docs/experiments.{zh,en}.md`: settings, commands, inputs/outputs for each experiment;
- `docs/data_and_results.{zh,en}.md`: data, experimental results and figures;
- `docs/reproduction.{zh,en}.md`: from-scratch reproduction and troubleshooting;

### 2.7 Scripts

| Script | Purpose |
|---|---|
| `scripts/run_main.sh` | main training, noise, probes, figures, SE, scoring |
| `scripts/run_addons.sh` | add-on figures/tables from stored results |
| `scripts/check.sh` | smoke check |

### 2.8 Tools

| File | Purpose |
|---|---|
| `tools/check_repo.py` | directory/file-count/forbidden-item checks |
| `tools/list_seeds.py` | print the paper seeds |
| `tools/seeds.json` | seed inventory |
| `tools/make_figure_previews.py` | compose PNG previews from paper PDFs |

## 3. Data flow of one evaluation

```text
dataset/clean/<task>
        │  src.data.datasets
        ▼
src.models.network ──► src.models.trainer ──► model/<task>/<clean / noise_trained>/*.h5
        │                                              │
        │  src.noise.perturbations / attacks           │  src.models.loader
        ▼                                              ▼
dataset/noise/<cond>/<task>/<repeat>.npy      src.probe.streaming.solve_base
        │                                              │
        └──────────────► src.probe.streaming.eval_condition ◄── result/probe/<exp>/<task>/<Model>/
                                      │
                                      ▼
                    result/criteria/<exp>/<task>/{robustness,std,se,drop}.pkl
                                      │
                     ┌────────────────┼─────────────────┐
                     ▼                ▼                 ▼
             result/repeat_raw    result/criteria/*.csv   figure/generated/...
                     │
                     ▼
             src.eval.aggregate
```

### 3.1 Stage-by-stage explanation

Read the diagram in this order: train once -> generate noise once -> evaluate each model -> aggregate:

| Stage | Input | Code | Output |
|---|---|---|---|
| 1. Train networks | `dataset/clean/<task>` | `src.models.trainer` via `src.pipelines.run_train` | `model/<task>/<clean / noise_trained>/*.h5` |
| 2. Generate noise | clean samples, noise seeds | `src.noise.build_noise`, `src.noise.attacks` | `dataset/noise/<condition>/<task>/<repeat>.npy`, `dataset/noise/attack/...` |
| 3. Fit MD-LP | network, training features, task facets | `src.probe.streaming.solve_base` | `result/probe/<exp>/<task>/<Model>/<layer>_{w,b}.npy`, `probe_acc.pkl`, base projections `P` |
| 4. Select susceptible samples | base projections, full-network predictions | `src.probe.streaming.select_from_base` | per-facet `select_list` in memory, reused by all conditions |
| 5. Base evaluation | base projections, `select_list` | `base_accuracy`, `final_from_preds` | `base` matrix `(L+1, n_splits)` |
| 6. Condition evaluation | one noise file, `w/b`, `select_list` | `eval_condition` | one `(L+1, n_splits)` matrix per repeat |
| 7. Repeat statistics | all repeat matrices | `run_probe_eval` | `criteria/*_{robustness,std,se,drop}.pkl`, CSV, `repeat_raw/*.npz` |
| 8. Post-processing | criteria matrices | `src.eval.aggregate`, `se_heatmap`, `scoring`, `src.viz` | `standard_error/`, `scoring/`, `figure/generated/` |

Key points:

- **The probe is not gradient-trained**: `solve_base` computes per-layer, per-facet `w/b` in one shot from feature moments and the MD-LP diagonal surrogate, so the probe is deterministic for a given model.
- **Susceptible samples are selected only once on the base set**: every noise condition and every repeat reuses the same `select_list`; otherwise conditions would not be comparable.
- **`exp` is not a noise condition**: `exp=clean` is the original network (6 conditions, 10 repeats each); `exp=noise` is the gauss+salt noise-trained network (base/gauss/salt, single evaluation, used for the 5.4.1 reliability check).
- **`base` has no std/SE**: it is evaluated once; `std`, `se` and `drop` are defined only for attack/gauss/salt/move/occ.
- **Classification and regression have different facet counts**: classification has 4 OVR facets (10 for full CIFAR-10), regression has 9 threshold facets, so the last matrix dimension is 4/10/9.
