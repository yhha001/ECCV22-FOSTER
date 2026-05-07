# FOSTER Guide for CIL Submission

This branch keeps the FOSTER code used to train and export the continual
learning checkpoints that are later adapted by
`Unlearn-Saliency/Classification` through the FOSTER adapter.

## Hardware Note

This submission-clean branch reflects the RTX 3090 environment used for the
current FOSTER-side preparation and export workflow. The included configs and
launch wrappers should be read as RTX-3090-oriented defaults.

A separate branch or release can be prepared later for the NVIDIA B200-based
data center if you want hardware-specific settings there.

## What Classification Uses From FOSTER

The unlearning code consumes FOSTER outputs such as:

- task checkpoints
- compressed/single-network state dicts
- replay buffer exports

In the current CIL pipeline, the most important use case is the ImageNet
FOSTER path that feeds `Classification` through `--cil_framework foster`.

## Important Files

- `main.py`
  Entry point that loads the JSON config and launches training.

- `trainer.py`
  Main training loop, checkpoint handling, and replay-buffer export path.

- `models/foster.py`
  Core FOSTER implementation.

- `models/base.py`
  Base continual-learning model utilities used by FOSTER.

- `utils/data_manager.py`
  Dataset schedule and class-order handling, including the ImageNet task setup.

- `utils/data.py`
  Dataset loading utilities.

- `configs/foster-imagenet100.json`
- `configs/foster-imagenet100-b0inc10.json`
- `configs/foster-imagenet1000-b0inc100.json`
- `configs/foster-imagenet1000-b0inc100-retrain-low-to-high-no-task0-6tasks50.json`
  Configurations used by the current paper-facing CIL path.

- `run_foster_*.sbatch`
  Cluster launch wrappers used for the release-relevant runs.

## Branch Scope

This branch removes large or machine-specific artifacts and keeps the source
needed for the FOSTER training/export path:

- `checkpoints/`
- `logs/`
- generated outputs not needed for the paper source release
