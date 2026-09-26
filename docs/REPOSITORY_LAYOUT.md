# Repository Layout

## Maintained Core

| Path | Purpose |
| --- | --- |
| `main_seg.py` | Primary custom segmentation training entry point. |
| `predict.py` | Custom-model fold and ensemble inference. |
| `models/` | MIAM architectures, model factory, and shared network blocks. |
| `data/` | nnU-Net-preprocessed dataset loaders and augmentations. |
| `config/` | Loss, optimizer, scheduler, inference, and checkpoint setup. |
| `options/` | YAML and command-line option handling. |
| `util/` | Training engines, metrics, export, and distributed utilities. |
| `args/` | Example experiment configurations. |
| `splits_plk/` | Versionable split definitions that do not expose sensitive data. |
| `nnUNet/nnunet/` | Vendored, locally modified nnU-Net v1 implementation. |
| `nnUNet_plan_and_preprocess.py` | Modified planning and preprocessing CLI. |
| `run_training_nnUnet.py` | Modified nnU-Net training CLI. |
| `run_prediction_nnUNet.py` | Modified nnU-Net inference CLI. |
| `MetricsReloaded/` | Vendored metric implementations. |

## Paper-Related Architecture Flow

1. `options/` merges command-line defaults with a supplied YAML configuration.
2. `data/custom_dataset_data_loader.py` selects a dataset implementation.
3. `data/nnUNet_dataset.py` reads nnU-Net plans, splits, and preprocessed cases.
4. `models/models.py` constructs the requested architecture.
5. `config/custom_TrainSetup.py` constructs optimization and inference tools.
6. `util/engineSeg.py` trains, validates, checkpoints, predicts, and evaluates.

## Experimental Status

The primary multimodal variants are `CNN_h+VIT_n`, `MCNN_h+VIT_n`, and
`MCNN_h+VIT_s`. Branches marked `TO CHECK` in `models/models.py` should be
treated as experimental until they have dedicated forward-pass and training
tests. A model appearing in the factory does not by itself guarantee that all
parameter combinations are supported.

## Local Data And Artifacts

The following are intentionally not part of source control:

- raw, cropped, and preprocessed medical images;
- trained models and pretrained binary weights;
- predictions, metrics workbooks, and visualization outputs;
- local virtual environments, IDE state, and Python caches;
- deprecated implementations and later project-specific experiments.

See `docs/DEPRECATED.md` for the local-only policy.
