# MD-LP Noise-Robustness Evaluation: Experiment Code

Experiment code for the paper **“On Evaluating Noise Robustness of Deep Networks
via Minkowski-Difference Linear Probe”**. It covers the main experiment and the six additional
experiments. Exploratory scripts, stale result versions and intermediate caches have been removed;
only the code, data splits, model weights, result matrices and figures used by the paper are kept.

## Repository versions

- **Current implementation** — this repository (`main`): reorganized code, English documentation,
  paper figures, main experiment and add-on experiments.
- **Original MDLP_NR implementation** — [MDLP_NR_OLD](https://github.com/RuifengZhou-dlut/MDLP_NR_OLD):
  the original version, kept for reference.

## 1. Method summary

- **Problem**: how to evaluate the noise robustness of deep networks and their substructures.
- **Core idea**: MD-LP, a linear probe that characterizes the discriminative structure of substructure
  outputs and uses it to select the samples most susceptible to noise.
- **Evaluation framework**: apply multiple noise types to the selected samples and derive criteria for
  both the whole network and its substructures in a unified pipeline.
- **Validation**: classification and regression experiments verify the method and analyze how noise
  interferes with deep networks.

## 2. Repository layout

```text
.
├── src/                         # all code, grouped by function
│   ├── common/                  # configuration, paths, seeding
│   ├── data/                    # dataset readers and raw-data builders
│   ├── noise/                   # four corruption noises + PGD attacks
│   ├── models/                  # architectures, weight loading, training loop
│   ├── probe/                   # MD-LP weights/bias, streaming evaluation, sampling
│   ├── eval/                    # SE aggregation over repeated noise
│   ├── viz/                     # curves, radar charts, paper panels
│   ├── pipelines/               # CLI entry points: train / noise / evaluate / plot
│   └── add_exp/                 # add-ons grouped by paper subsubsection
│       ├── common/              # shared: sites, feature cache, CKA/SVCCA/t-SNE
│       ├── probe_efficiency/    # 5.2.1
│       ├── probe_stability/     # 5.2.2 (weight_vs_samples / weight_vs_noise)
│       ├── similarity_comparison/ # 5.4.2
│       └── s_sensitivity/       # 5.4.3
├── model/                       # trained weights by task / training mode
├── dataset/                     # clean splits and generated noise by condition
├── result/                      # probes, criteria matrices, add-on raw data
├── figure/                      # paper PDF/PNG previews; runtime outputs in figure/generated/
├── docs/                        # English documentation
├── scripts/                     # command wrappers
└── tools/                       # validation / helper utilities
```

Each add-on experiment has a matching `result/add_exp/<group>/` and `figure/generated/add_exp/<group>/` folder (`figure/generated/` is runtime output and is not committed).
See [docs/architecture.md](docs/architecture.md) for the full tree and module responsibilities.

## 3. Environment

- Linux, Python 3.9, TensorFlow 2.x (GPU recommended), NumPy / SciPy / scikit-learn / Matplotlib /
  h5py / Pillow / psutil.
- A single NVIDIA A16/A30 (12–24 GB) is sufficient; the probe and add-on experiments also run on CPU.

```bash
conda create -n mdlp python=3.9 -y
conda activate mdlp
pip install -r requirements.txt
```

## 4. Data and models

The exact paper splits are included in a compact form:

| Path | Content | Notes |
|---|---|---|
| `dataset/clean/classification/data.npz` | four-class CIFAR-10 (airplane/automobile/ship/truck), 20k train / 4k test | uint8 images, decoded to float32 |
| `dataset/clean/regression/parts/part-00*.npz` | stratified UTKFace, 20k train / 2k test | float16 images, float64 age labels |

The full ten-class CIFAR-10 split (used by add-on 01) is not committed because of its size; build it with
`python -m src.data.build_dataset --dataset cifar10_full` (about 150 MB, git-ignored).
See [docs/data_and_results.md](docs/data_and_results.md) for the raw-data recipes.

Weights for all four backbones on classification, full CIFAR-10, regression, and Gaussian+salt
noise-augmented training are provided under `model/`, so the paper numbers can be recomputed
without retraining.

## 5. Quick start (main experiment)

Run from the repository root. Example: classification with VGG-11.

```bash
# 1) train (optional: weights are already provided)
python -m src.pipelines.run_train --task classification --model VGG-11

# 2) generate ten noise realisations and PGD samples
python -m src.noise.build_noise --task classification --what all --repeats 10

# 3) build probes and evaluate (probe_acc / robustness / std / se / drop / summary csv)
python -m src.pipelines.run_probe_eval --task classification --model VGG-11 --exp clean

# 4) main-experiment curves
python -m src.pipelines.plot_clean --task classification

# 5) SE aggregation over repeated noise
python -m src.eval.aggregate
python -m src.eval.se_heatmap
```

For other tasks/models, change `--task` to `regression` or `classification_full`, and `--model` to
`Res-18`, `ViT-8` or `Mix-8`. For the noise-augmented branch, train with `--exp noise --noise-aug`
and evaluate with `--exp noise`.

## 6. Add-on experiments

Add-ons live in a single `src/add_exp/` tree and are grouped by the paper's subsubsections (no
`add_001`-style numbering). Experiments from the same subsubsection share a folder; for example the
sampling-stability and noise-stability experiments of 5.2.2 both live under `probe_stability/`.

| Paper | Folder | Content | Entry point |
|---|---|---|---|
| 5.2.1 | `src/add_exp/probe_efficiency` | time / memory cost of probe construction | `python -m src.add_exp.probe_efficiency.benchmark_ovr` |
| 5.2.2 | `src/add_exp/probe_stability` | ten-draw weight stability under subsampling / feature noise (two subfolders) | see below |
| 5.4.2 | `src/add_exp/similarity_comparison` | rotated noise direction: MD-LP vs CKA/SVCCA/t-SNE | `python -m src.add_exp.similarity_comparison.compute_rotation` |
| 5.4.3 | `src/add_exp/s_sensitivity` | sensitivity to the susceptible-sample ratio s | `python -m src.add_exp.s_sensitivity.scan_s` |

Inside `probe_stability/`:

```bash
python -m src.add_exp.probe_stability.weight_vs_samples.compute_stability --tag 8sites_10rep
python -m src.add_exp.probe_stability.weight_vs_noise.compute_stability --tag 8sites_10draws
```

Purposes, settings, inputs/outputs and paper values are documented in
[docs/experiments.md](docs/experiments.md).

## 7. Evaluation protocol (matching the paper)

- **In-sample**: probes are fitted and evaluated on the 20k training samples; only samples whose
  last-layer probe prediction agrees with the network prediction are kept.
- **Susceptible ratio** `s = 0.2`: per facet, keep the 20% smallest-margin samples.
- **Noise**: gauss `a=0.2`, salt `n=64`, move `degree=15`, occ `v=192`, PGD
  `eps=2.0 / alpha=0.2 / steps=20`; ten independent realisations per condition.
- **Uncertainty**: `s` is the across-realisation standard deviation (ddof=0), `SE = s/sqrt(n-1)`;
  ties use the `2.77·SE` threshold.
- **Facets**: four one-vs-rest classes for CIFAR (ten for full CIFAR-10), nine age-quantile
  thresholds for UTKFace.
- **Seeds**: training, noise, PGD and add-on seeds are fixed; the full list and derivation rules are
  in [Reproduction and seeds](docs/reproduction.md) and `tools/seeds.json`; verify with `python tools/list_seeds.py`.

## 8. Documentation

- [Architecture](docs/architecture.md)
- [Experiment guide](docs/experiments.md)
- [Data, results and figures](docs/data_and_results.md)
- [Reproduction and seeds](docs/reproduction.md)
