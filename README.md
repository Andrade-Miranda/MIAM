# MIAM

MIAM is a research framework for three-dimensional, multi-modal medical image
segmentation. It contains hybrid CNN-Transformer architectures, transformer
baselines, CNN baselines, nnU-Net-based preprocessing, and training and
evaluation utilities used to study modality fusion in oncology imaging.

This repository is associated with two publications:

> Gustavo Andrade-Miranda et al. **Multi-modal medical Transformers: A
> meta-analysis for medical image segmentation in oncology.** Computerized
> Medical Imaging and Graphics, 110, 102308, 2023.
> <https://doi.org/10.1016/j.compmedimag.2023.102308>

> Gustavo Andrade-Miranda, Pedro Soto Vega, Kamilia Taguelmimt, Hong-Phuong
> Dang, Dimitris Visvikis, and Julien Bert. **Exploring transformer reliability
> in clinically significant prostate cancer segmentation: A comprehensive
> in-depth investigation.** Computerized Medical Imaging and Graphics, 118,
> 102459, 2024. <https://doi.org/10.1016/j.compmedimag.2024.102459>

The 2023 publication is a review and meta-analysis, with MIAM serving as its
related multimodal experimental framework. The 2024 publication uses the
framework for an in-depth reliability investigation of clinically significant
prostate cancer segmentation. The current checkout does not contain every
original dataset, preprocessing plan, checkpoint, or expected result table, so
it should not be described as a complete numerical reproduction package.

## Publication Map

| Publication | Main repository components |
| --- | --- |
| 2023 multimodal transformer meta-analysis | `models/`, `main_seg.py`, `data/nnUNet_dataset.py`, and the shared training and nnU-Net preprocessing stack. |
| 2024 prostate cancer reliability investigation | `main_picai.py`, `config/PICAI_Config.py`, `picai_eval/`, PI-CAI split helpers, and the shared models, calibration, uncertainty, and evaluation utilities. |

## Models

The principal paper-related model families are defined in `models/`:

| Configuration name | Description |
| --- | --- |
| `CNN_h+VIT_n` | A shared 3D CNN encoder followed by a ViT bottleneck. |
| `MCNN_h+VIT_n` | One CNN encoder per modality with naive feature/token fusion. |
| `MCNN_h+VIT_s` | Modality-specific CNN encoders and embeddings with a shared transformer. |
| `UNETR` | UNETR baseline with deep supervision. |
| `SwinTrans3D` | Swin Transformer segmentation baseline with deep supervision. |
| `nnUNetPlain`, `nnUNetRes` | Plain and residual nnU-Net-style CNN baselines. |
| `MedNeXt` | MedNeXt comparison model. |

Some additional architectures remain experimental. Consult
[`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md) before selecting a
model.

## Requirements

The supplied `MedicalImg.yml` describes the development environment, including
Python 3.12, PyTorch 2.6, MONAI 1.4, CUDA 12 packages, SimpleITK, nibabel,
batchgenerators, timm, and OmegaConf.

```bash
conda env create -f MedicalImg.yml
conda activate MedicalImg
```

GPU training requires a compatible NVIDIA driver. CPU-only use may require a
separate PyTorch installation. The bundled nnU-Net code is based on nnU-Net v1
and has local modifications; do not assume compatibility with nnU-Net v2.

Commands in this project generally expect the repository root to be the current
working directory.

## Data Layout

MIAM uses the nnU-Net v1 dataset convention:

```text
nnUNet/data/nnUnet_raw/nnUNet_raw_data/TaskXXX_Name/
├── dataset.json
├── imagesTr/
│   ├── CASE_0000.nii.gz
│   ├── CASE_0001.nii.gz
│   └── ...
├── labelsTr/
│   └── CASE.nii.gz
└── imagesTs/
```

Each `_0000`, `_0001`, and subsequent suffix identifies one image modality.
The channel meanings and label definitions belong in `dataset.json`. Medical
datasets are intentionally ignored by Git and must be obtained under their
original licenses and usage agreements.

## Preprocessing

The modified nnU-Net planning entry point analyzes, crops, plans, and
preprocesses a task:

```bash
python nnUNet_plan_and_preprocess.py \
  -t TASK_ID \
  -pl3d ExperimentPlanner3D_v21 \
  -pl2d None \
  --verify_dataset_integrity
```

Generated plans and preprocessed arrays are written below `nnUNet/data/` and
are local-only. Review all planner and cropping options before processing a new
dataset.

## Configuration

Training is configured with OmegaConf YAML files. Important settings include:

- `dataroot`: nnU-Net task name.
- `dataset_mode`: normally `nnUNet` for the paper-related workflow.
- `plan`: preprocessing plan filename.
- `loadsplit` and `fold`: cross-validation split and selected fold.
- `encoder`: model name from `models/models.py`.
- `input_nc` and `output_nc`: input channels and output classes.
- `TrainConfig`, optimizer, loss, scheduler, and epoch settings.

Values such as `imageSize: 0` and `batchSize: 0` are resolved from the nnU-Net
plan. Start from a relevant YAML under `args/`, then use paths and channel
definitions matching your own dataset.

## Training

```bash
python main_seg.py args/segmentation/CONFIG.yaml
```

The first positional argument must be a YAML configuration. Training writes
resolved options and checkpoints below `checkpoints/`, and metrics and
segmentations below `Output/`. These directories are ignored by Git.

The standard workflow uses a held-out cross-validation fold for model selection
and final fold prediction. In the current code, this is not an independent
external test cohort.

The prostate reliability workflow uses its dedicated entry point and
configuration:

```bash
python main_picai.py args/UNETR_PICAI-Clas.yaml
```

Dataset locations, plans, splits, channels, and labels must be adapted to the
licensed prostate MRI data available to the user. The checked-in YAML is a
research configuration reference, not a self-contained downloadable dataset.

## Inference And Evaluation

The custom-model inference entry point is:

```bash
python predict.py --help
```

The repository also provides modified nnU-Net prediction and ensemble entry
points. Input cases must follow nnU-Net's modality suffix convention. Verify
image orientation, spacing, labels, and checkpoint compatibility before using
predictions for research analysis.

Typical outputs include:

- `BestCHK.pth` and `LastCHK.pth` checkpoints;
- `metrics.xlsx` training history;
- NIfTI segmentation volumes;
- Dice, ASSD, HD95, volumetric similarity, precision, recall, and Jaccard
  measurements.

## Repository Status

This is research software, not a clinical product. It must not be used for
clinical decisions. Full training and numerical reproduction require licensed
medical datasets, matching preprocessing plans, considerable compute, and the
original experiment metadata.

Deprecated code and later unrelated experiments are retained only in local
working copies and are excluded from the public project. See
[`docs/DEPRECATED.md`](docs/DEPRECATED.md).

## Documentation

- [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md): reproducibility workflow and
  known limitations.
- [`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md): maintained component
  map and entry points.
- [`docs/DEPRECATED.md`](docs/DEPRECATED.md): local-only file policy.
- [`nnUNet/readme.md`](nnUNet/readme.md): upstream nnU-Net v1 documentation.

## Future Work

- Archive the exact paper configurations, data manifests, preprocessing plans,
  random seeds, and expected fold-level results.
- Add public pretrained weights where dataset and model licenses permit.
- Expand automated unit, integration, CPU, and GPU tests with continuous
  integration.
- Improve robustness to missing modalities and investigate additional
  modality-fusion strategies.
- Replace remaining working-directory assumptions with an installable package
  and explicit configuration paths.
- Evaluate migration from the locally modified nnU-Net v1 implementation to a
  maintained preprocessing and training stack.
- Improve calibration, uncertainty estimation, and external-cohort validation.

## Citations

```bibtex
@article{andrade_miranda_2023_multimodal,
  title = {Multi-modal medical Transformers: A meta-analysis for medical image segmentation in oncology},
  journal = {Computerized Medical Imaging and Graphics},
  volume = {110},
  pages = {102308},
  year = {2023},
  doi = {10.1016/j.compmedimag.2023.102308},
  author = {Andrade-Miranda, Gustavo and others}
}

@article{andrade_miranda_2024_reliability,
  title = {Exploring transformer reliability in clinically significant prostate cancer segmentation: A comprehensive in-depth investigation},
  journal = {Computerized Medical Imaging and Graphics},
  volume = {118},
  pages = {102459},
  year = {2024},
  doi = {10.1016/j.compmedimag.2024.102459},
  author = {Andrade-Miranda, Gustavo and Soto Vega, Pedro and Taguelmimt, Kamilia and Dang, Hong-Phuong and Visvikis, Dimitris and Bert, Julien}
}
```

Use the publishers' citation records as the authoritative bibliographic
metadata.
