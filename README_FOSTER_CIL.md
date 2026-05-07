# FOSTER Guide for CIL Submission

This branch keeps the FOSTER code used to train and export the continual
learning checkpoints that are later adapted by
`Unlearn-Saliency/Classification` through the FOSTER adapter.

## Repository Provenance

This `submission-clean` branch is a trimmed paper-release snapshot of the
existing FOSTER repository. It is included because the paper's ImageNet-1K CIL
pipeline depends on FOSTER-side checkpoint and replay-buffer preparation, not
because this release is claiming a new from-scratch FOSTER framework.

Please preserve the original FOSTER license, citations, and repository
attribution when reusing this branch.

## Hardware Note

This submission-clean branch reflects the RTX 3090 environment used for the
current FOSTER-side preparation and export workflow. The included configs and
export assumptions should be read as RTX-3090-oriented defaults.

A separate branch or release can be prepared later for the NVIDIA B200-based
data center if you want hardware-specific settings there.

## Dependencies

For this trimmed branch, install:

- `pip install -r requirements.txt`

## What Classification Uses From FOSTER

The unlearning code consumes FOSTER outputs such as:

- task checkpoints
- compressed/single-network state dicts
- replay buffer exports

In the current `submission-clean` pipeline, the retained use case is the
ImageNet-1K FOSTER path that feeds `Classification` through
`--cil_framework foster`.

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

- `configs/foster-imagenet1000-b0inc100.json`
- `configs/foster-imagenet1000-b0inc100-retrain-low-to-high-no-task0-6tasks50.json`
  Configurations used by the current paper-facing CIL path.

## Branch Scope

This branch removes large or machine-specific artifacts and keeps the source
needed for the FOSTER training/export path:

- `checkpoints/`
- `logs/`
- generated outputs not needed for the paper source release

It also trims away CIFAR, ImageNet-100, and RMM-specific files so the release
matches the retained ImageNet-1K path more closely.

It also intentionally omits the local sbatch launch wrappers from the
submission release because they encoded machine-specific repository paths and
cluster assumptions. Use `main.py --config ...` directly with the retained
ImageNet-1K configs instead.
