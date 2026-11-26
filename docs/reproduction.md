# Reproduction Guide and Random Seeds

Shortest path to reproduce the paper, expected runtimes, and random seeds. 

## 1. Environment

```bash
conda create -n mdlp python=3.9 -y
conda activate mdlp
pip install -r requirements.txt
```

Main dependencies: `tensorflow>=2.8`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `h5py`,
`Pillow`, `psutil`. A single NVIDIA A16/A30 is enough; probes and add-ons also run on CPU.

Check the installation:

```bash
python -c "import tensorflow as tf, numpy, scipy, sklearn, matplotlib; print(tf.__version__)"
```

## 2. Shortest reproduction path

### 2.1 Recompute tables/figures from the shipped assets

The repository already contains:

- clean data: `dataset/clean/classification`, `dataset/clean/regression`;
- weights for classification / regression / full CIFAR-10 / noise-augmented training: `model/`;
- main probes and criteria: `result/probe`, `result/criteria`;
- add-on raw data and figures: `result/add_exp/<group>`, `figure/generated/add_exp/<group>`.

So the paper tables/figures can be recomputed **without training**:

```bash
python -m src.pipelines.plot_clean --task classification
python -m src.pipelines.plot_clean --task regression
python -m src.pipelines.plot_clean --task classification_full

python -m src.add_exp.probe_stability.weight_vs_samples.plot_stability
python -m src.add_exp.probe_stability.weight_vs_noise.plot_stability
python -m src.add_exp.s_sensitivity.make_figure
```

### 2.2 Rerun the main experiment from scratch

```bash
for m in VGG-11 Res-18 ViT-8 Mix-8; do
  python -m src.pipelines.run_train --task classification --model $m
  python -m src.pipelines.run_train --task regression    --model $m
done

python -m src.noise.build_noise --task classification --what all --repeats 10
python -m src.noise.build_noise --task regression     --what all --repeats 10

for m in VGG-11 Res-18 ViT-8 Mix-8; do
  python -m src.pipelines.run_probe_eval --task classification --model $m --exp clean
  python -m src.pipelines.run_probe_eval --task regression     --model $m --exp clean
done

python -m src.pipelines.plot_clean --task classification
python -m src.pipelines.plot_clean --task regression
python -m src.eval.aggregate
python -m src.eval.se_heatmap
python -m src.eval.scoring
python -m src.eval.make_score_table
```

### 2.3 Noise-augmented branch (reliability comparison)

```bash
for m in VGG-11 Res-18 ViT-8 Mix-8; do
  python -m src.pipelines.run_train --task classification --model $m --exp noise --noise-aug
  python -m src.pipelines.run_train --task regression     --model $m --exp noise --noise-aug
  python -m src.pipelines.run_probe_eval --task classification --model $m --exp noise --max-repeats 1
  python -m src.pipelines.run_probe_eval --task regression     --model $m --exp noise --max-repeats 1
done
python -m src.pipelines.plot_noise --task classification
python -m src.pipelines.plot_noise --task regression
```

### 2.4 5.3.1 Full ten-class CIFAR-10

```bash
python -m src.data.build_dataset --dataset cifar10_full
for m in VGG-11 Res-18 ViT-8 Mix-8; do
  python -m src.pipelines.run_train --task classification_full --model $m
done
python -m src.noise.build_noise --task classification_full --what all --repeats 10
for m in VGG-11 Res-18 ViT-8 Mix-8; do
  python -m src.pipelines.run_probe_eval --task classification_full --model $m --exp clean
done
python -m src.pipelines.plot_clean --task classification_full
```

### 2.5 Feature cache for probe_efficiency / probe_stability / similarity_comparison

```bash
python -m src.add_exp.common.features                  # eight sites, about 1.5 GB
```

Then run 04–08; full commands are in [experiments.md](experiments.md).

## 3. Expected runtime

| Step | Single A16/A30 (reference) |
|---|---|
| train one network (classification/regression, 200 epochs) | 20–60 min |
| ten corruption realisations for one task | 10–30 min (motion blur dominates) |
| ten PGD realisations for one task (four models) | 20–60 min |
| probe evaluation, one task × one model (ten repeats) | 3–10 min |
| `add_exp.common.features` (two models, eight sites) | 5–15 min |
| scoring | seconds |
| weight_vs_samples / weight_vs_noise (8 sites × 13 ratios / 8 rho × 10 draws) | tens of minutes (CPU) |

## 4. Quick self-check

```bash
python -c "from src.data.datasets import load_train_xy; x,y,_,_=load_train_xy('classification'); print(x.shape,y.shape)"
python -c "from src.models.loader import load_model; m=load_model('classification','VGG-11'); print(m.output_shape)"
python -m src.pipelines.run_probe_eval --task classification --model VGG-11 \
    --conditions base --max-layers 2 --out-dir /tmp/smoke
python -m src.eval.scoring --out /tmp/smoke_score
```

## 5. Random seeds and determinism

### 5.1 Master seeds

| Purpose | Code location | Value |
|---|---|---:|
| Network training | `BASE_SEED` | `20240101` |
| Noise generation | `NOISE_SEED` | `20250101` |
| Rotation | `compute_rotation.SEED` | `20261108` |
| Weight stability under subsampling | `compute_stability --seed` default | `20261114` |
| Weight stability under feature noise | internal base in `weight_vs_noise/compute_stability.py` | `20261116` |

### 5.2 Network training

`src/pipelines/run_train.py` defaults to `--seed 20240101` and derives:

```text
data_seed = derive_seed("train", exp, task, model, "noise" if noise_aug else "clean", base=BASE_SEED)
shuffle_seed = derive_seed("shuffle", data_seed)
```

`derive_seed` takes the first four bytes of the MD5 of the tag string, adds `base`, and reduces
modulo `2^31-1` (see `src/common/seeding.py`). Training, augmentation, mixup and shuffling all use
seeds derived from `data_seed`.

| Task | Model | clean training seed | noise training seed |
|---|---|---:|---:|
| Classification | VGG-11 | `111306650` | `486216303` |
| Classification | Res-18 | `1106460013` | `326443724` |
| Classification | ViT-8 | `364542236` | `2043233944` |
| Classification | Mix-8 | `274180967` | `641520412` |
| Regression | VGG-11 | `1003949926` | `1032411599` |
| Regression | Res-18 | `1357236050` | `1625260464` |
| Regression | ViT-8 | `104965290` | `486978136` |
| Regression | Mix-8 | `394594355` | `155875218` |
| Full CIFAR-10 | VGG-11 | `289865669` | `—` |
| Full CIFAR-10 | Res-18 | `443655890` | `—` |
| Full CIFAR-10 | ViT-8 | `878470718` | `—` |
| Full CIFAR-10 | Mix-8 | `1004198446` | `—` |

> `classification_full` has a clean branch only; the noise branch is used for the reliability comparison.

### 5.3 Noise group seeds

`build_noise.py` derives one group seed per task and repeat:

```text
group_seed = derive_seed("noise-group", task, repeat, base=NOISE_SEED)
```

Within a group, `gauss → salt → move → occ` are generated sequentially after
`set_global_seed(group_seed)`. PGD is not part of this sequence; it is derived per model from the
group seed (Section 3).

| Task | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classification | `775146566` | `387842391` | `1145465218` | `1130927290` | `971198644` | `1748176035` | `97484259` | `545991222` | `1090598157` | `2020163214` |
| Regression | `2028089737` | `1237224883` | `770570401` | `84615470` | `924950933` | `2072326519` | `2027508135` | `417468430` | `1352391393` | `904329886` |
| Full CIFAR-10 | `734274547` | `1580729939` | `1034134236` | `1552103057` | `1901833437` | `421135290` | `1176194767` | `1890449226` | `1348571291` | `1495831827` |

### 5.4 PGD seeds

```text
pgd_seed       = derive_seed("pgd",       model, base=group_seed)
pgd_start_seed = derive_seed("pgd-start", model, base=group_seed)
```

`pgd_seed` controls the PGD random start and batch order; `pgd_start_seed` controls the initial
perturbation. All 120 `(task, repeat, model)` values are in the `pgd` field of
`tools/seeds.json`.

| Task | Model | PGD seed for repeat 00 |
|---|---|---:|
| Classification | VGG-11 | `710768497` |
| Classification | Res-18 | `1764460016` |
| Classification | ViT-8 | `1417450837` |
| Classification | Mix-8 | `835462197` |
| Regression | VGG-11 | `1963711668` |
| Regression | Res-18 | `869919540` |
| Regression | ViT-8 | `522910361` |
| Regression | Mix-8 | `2088405368` |
| Full CIFAR-10 | VGG-11 | `669896478` |
| Full CIFAR-10 | Res-18 | `1723587997` |
| Full CIFAR-10 | ViT-8 | `1376578818` |
| Full CIFAR-10 | Mix-8 | `794590178` |

### 5.5 Data split and susceptible-sample selection

- **UTKFace stratified split**: `derive_seed("split", "regression", ..., base=BASE_SEED)` gives
  `919681617`; the shipped `dataset/clean/regression/parts/` follows this split.
- **Four-class CIFAR subset**: `dataset/clean/classification/data.npz` is the fixed paper split;
  `build_dataset.py --dataset cifar10_subset` rebuilds the class counts but is not guaranteed to
  reproduce the archived sample-for-sample split.
- **Susceptible-sample selection**: `s=0.2`, stable sort by the last-layer probe score, then take
  `floor(s*n_pos)` / `floor(s*n_neg)`; **no RNG**.
- **Probe weights/bias**: closed-form on fixed data; **no RNG**. With fixed models and data, probes
  and selection are deterministic.

### 5.6 Add-on experiments

| Experiment (group) | Seed / randomness |
|---|---|
| 5.2.1 `probe_efficiency` | none (timing/memory only); feature extraction is deterministic |
| 5.2.2 `probe_stability/weight_vs_samples` | base `20261114`; per `(site, ratio, repeat)` a CRC32-derived NumPy generator |
| 5.2.2 `probe_stability/weight_vs_noise` | `20261116 + site_index`; one `default_rng` per site, ten draws taken sequentially over rho |
| 5.3.1 Full CIFAR-10 | training seeds from Section 1; noise group seeds from Section 2 |
| 5.3.3 (main-experiment SE) | no new draws; reuses the ten main-experiment noise realisations |
| 5.4.1 `reliability` | `noise` training seeds and the main-experiment noise |
| 5.4.2 `similarity_comparison` | `SEED=20261108`; `e2` uses `SEED+9`, noise uses `SEED`; angle/amplitude order is fixed |
| 5.4.3 `s_sensitivity` | no new draws; reuses main probes/noise and only changes the selection ratio `s` |
| 6 Weighted scores and rankings (Application Example) | none (reads pkl files and computes weighted scores/ranks) |

### 5.7 Verification

```bash
python tools/list_seeds.py          # regenerate tools/seeds.json
git diff tools/seeds.json           # should be empty
```

Seeds guarantee reproducibility **for the same code path and environment**. Different GPUs, cuDNN,
TensorFlow/NumPy/OpenBLAS versions can still shift results at the `1e-3` level. The paper tables
are shipped in `result/` for direct bit-level checking.

### 5.8 Determinism levels and recommendations

Fixed seeds make the random streams reproducible; they do not guarantee bitwise equality across
hardware or library versions. Three practical levels:

| Level | How | Expected agreement |
|---|---|---|
| L1 verify paper numbers | use the shipped `model/*.h5` and `result/*`, only re-run evaluation/plotting | table values match; linear algebra differs at 1e-12–1e-6 |
| L2 retrain in the same environment | same GPU, CUDA/cuDNN, TensorFlow/NumPy/OpenBLAS; single GPU; `--deterministic` | training/noise nearly bitwise; criteria usually 1e-6–1e-4 |
| L3 retrain across environments | seeds only, libraries/hardware differ | statistically equivalent; criteria often differ at 1e-3 |

Strict mode:

```bash
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

python -m src.pipelines.run_train --task classification --model VGG-11 --deterministic
```

Notes:

- `--deterministic` disables some non-deterministic GPU kernels and is slower; without it,
  cuDNN/atomic operations can shift weights at the `1e-3` level.
- PGD is iterative; tiny numerical differences can flip individual samples, while mean criteria
  usually remain stable.
- Add-on 08 (t-SNE/PCA) is sensitive to library versions; Spearman rho may vary by about ±0.05.
- Add-ons 02/03/09 only read stored results; re-generating tables/figures is deterministic.
