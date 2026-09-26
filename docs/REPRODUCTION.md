# Reproduction Guide

## Scope

MIAM supports experiments on 3D multi-modal oncology segmentation, including
early channel fusion, modality-specific CNN encoders, modality-aware token
embeddings, shared transformer fusion, and established CNN/transformer
baselines.

The repository also contains the prostate-specific workflow associated with
the 2024 reliability investigation. It uses `main_picai.py`,
`config/PICAI_Config.py`, the PI-CAI split helpers, `picai_eval/`, and shared
calibration, uncertainty, model, and training components.

The repository does not currently provide a bit-for-bit reproduction of the
2023 publication. Exact paper configurations, original datasets, generated
plans, all fold assignments, trained checkpoints, and expected numerical
results are not available together in this checkout.

The same limitation applies to exact numerical reproduction of the 2024 study:
the licensed prostate MRI cohorts, generated preprocessing plans, all trained
weights, and complete expected fold-level outputs are not distributed here.

## Recommended Workflow

1. Create and activate the environment from `MedicalImg.yml`.
2. Place an nnU-Net-formatted dataset under
   `nnUNet/data/nnUnet_raw/nnUNet_raw_data/`.
3. Verify `dataset.json`, image modality suffixes, labels, orientation, and
   affine metadata.
4. Run `nnUNet_plan_and_preprocess.py` for the task.
5. Create or select a split file below `splits_plk/<task>/<dataset_mode>/`.
6. Create a YAML configuration with the matching task, plan, channels, labels,
   model, and fold.
7. Run `python main_seg.py path/to/config.yaml`.
8. Repeat all folds and model variants required by the experiment design.
9. Preserve the resolved `opt.yaml`, software environment, plan, split,
   checkpoints, and metric outputs for every run.

For the prostate reliability workflow, use a matching PI-CAI configuration and
run:

```bash
python main_picai.py args/UNETR_PICAI-Clas.yaml
```

Do not run the example unchanged until its dataset, plan, split, channel, and
output settings have been reviewed for the local environment.

## Reproducibility Record

For a publishable experiment, record at least:

- dataset release, license, inclusion criteria, and case identifiers;
- channel order and label mapping;
- preprocessing planner, target spacing, crop policy, and plan hash;
- split file and random seed;
- complete resolved YAML configuration;
- Python, CUDA, PyTorch, MONAI, and nnU-Net versions;
- GPU model and number of devices;
- fold-level metrics, aggregate method, and confidence intervals;
- checkpoint and prediction checksums.

## Validation Caveats

The current segmentation workflow selects the best epoch using validation Dice
and then exports predictions for that same held-out fold. These results are
cross-validation results, not an independent external test. External validation
should use a cohort that was not involved in model selection or calibration.

Medical segmentation results are sensitive to resampling, interpolation,
orientation, connected-component processing, empty masks, and metric
definitions. Report these choices with every result.

## Expected Outputs

Training normally creates:

```text
checkpoints/<task>/<encoder>/<experiment>/
├── BestCHK.pth
├── LastCHK.pth
├── opt.txt
└── opt.yaml

Output/<task>/<encoder>/<experiment>/
├── metrics.xlsx
└── predictions/
```

These artifacts can be large or contain dataset-derived information and are
therefore local-only by default.
