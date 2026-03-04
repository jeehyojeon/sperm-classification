# An End-to-End Framework for Automated Sperm Normality Classification

This repository corresponds exactly to the submitted MICCAI 2026 manuscript. No post-submission modifications affecting experimental results have been made.

## Description
This project implements an end-to-end framework for the automated detection and classification of sperm normality. The system utilizes a deep learning architecture with a realistic CNN backbone, integrated with specialized heads for localization and classification. It addresses class imbalance using weighted loss functions to ensure robust performance across diverse sperm samples.

## Environment Specification
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (for GPU acceleration)
- Ubuntu 20.04/22.04

## Installation Instructions
1. Clone the repository (anonymous version).
2. Create a virtual environment or use Conda:
   ```bash
   conda env create -f environment.yml
   conda activate miccai2026_sperm
   ```
3. Alternatively, install via pip:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure
The dataset should be organized as follows:
```
datasets/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```
Please refer to [datasets/README.md](datasets/README.md) for more details on the annotation format.

## Training Command
To train the model from scratch, run:
```bash
python train.py --config configs/miccai2026.yaml
```

## Evaluation Command
To evaluate the model using the best-saved weights:
```bash
python test.py --config configs/miccai2026.yaml
```

## Expected Output Metrics
| Metric | Value (YOLO Pipeline) |
| --- | --- |
| ROC-AUC | 0.8576 ± 0.0114 |
| PR-AUC (AP) | 0.4283 ± 0.0150 |
| F2-Score | 0.6088 ± 0.0243 |
| Recall | 0.8265 ± 0.0110 |
| Adjusted Recall | 0.2036 ± 0.0206 |

## Reproducibility Statement
Random seeds for PyTorch, NumPy, and Python's random module are fixed (default `seed: 42` in `configs/miccai2026.yaml`) to ensure exact reproduction of the results presented in the manuscript.

## Anonymous Citation Block
```bibtex
@inproceedings{anonymous2026sperm,
  title={An End-to-End Framework for Automated Sperm Normality Classification},
  author={Anonymous Authors},
  booktitle={MICCAI},
  year={2026}
}
```
