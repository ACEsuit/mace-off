# MACE-OFF Evaluation on SPICE

This repository contains scripts for checking force quality in SPICE extended XYZ files, evaluating MACE-OFF models on a SPICE test set, and plotting the resulting error metrics. Large datasets and model checkpoints are not included.

## Repository layout

```text
.
├── data/                      # Place SPICE extxyz inputs here (not tracked)
├── results/                   # Example evaluation figure
├── scripts/
│   ├── check_data.py          # Flag frames with unusually large reference forces
│   ├── evaluate_models.py     # Evaluate MACE-OFF23 models and save metrics/figure
│   └── training/              # MACE training command templates
└── models/                    # Place model checkpoints here (not tracked)
```

## Requirements

Python 3.10 or newer is recommended. The scripts use NumPy, ASE, Matplotlib, and the MACE Python package. Install versions compatible with your PyTorch and accelerator setup, for example:

```bash
pip install numpy ase matplotlib mace-torch
```

## Input files

Provide the SPICE extxyz files locally:

- `data/test_large_neut_all.xyz` is the default evaluation and force-check input.
- `data/train_large_neut_no_bad_clean.xyz` is the training set referenced by this project; it is not read by the evaluation script.

The files are intentionally excluded from Git because of their size. Obtain them from the appropriate SPICE source and ensure you follow its data-use terms.

Place the MACE-OFF23 checkpoints at:

```text
models/mace_off23/MACE-OFF23_small.model
models/mace_off23/MACE-OFF23_medium.model
models/mace_off23/MACE-OFF23_large.model
```

The model files are also excluded from Git. The evaluation script currently targets these three MACE-OFF23 variants.

## Usage

Scan one or more extxyz files for reference-force issues:

```bash
python scripts/check_data.py data/test_large_neut_all.xyz
```

The default report path is `data/force_quality_issues.csv`. Frames are flagged if any atomic force exceeds 15 eV/Å or the whole-configuration net force exceeds 0.1 eV/Å. Molecular-fragment net force is reported as a diagnostic and does not itself trigger exclusion.

Evaluate the three models and write a metrics CSV and plot:

```bash
python scripts/evaluate_models.py --device cpu
```

By default, this reads `data/test_large_neut_all.xyz` and writes `results/fig1_metrics.csv` and `results/fig1.png`. Use `--limit N` for a short inference check, `--device cuda` for a CUDA-enabled setup, or `--xyz`, `--csv`, and `--figure` to override paths.

## Training templates

The shell files under `scripts/training/` contain MACE training command templates. Replace the `/path/to/...` dataset and statistics placeholders with paths valid in your environment before running them. The templates share a common configuration and currently use the same experiment name; set a distinct `--name` for separate runs.

## Notes

- Input energies are assumed to be in eV and forces in eV/Å, as represented by the extxyz data.
- Evaluation subsets are selected using the `config_type` metadata in each frame.
- `results/fig1.png` is a checked-in example figure. Re-running evaluation overwrites the default figure and generates a metrics CSV that is ignored by Git.
