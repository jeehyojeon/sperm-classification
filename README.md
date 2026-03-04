# An End-to-End Framework for Automated Sperm Normality Classification

This repository provides the anonymous implementation of **ESCIMO**, proposed in the submitted MICCAI 2026 manuscript.

This repository corresponds exactly to the submitted MICCAI 2026 manuscript. No post-submission modifications affecting experimental results have been made.

---

## Description

ESCIMO is an end-to-end framework for automated sperm detection and normality classification.  
The detection module is based on a YOLO-style architecture, followed by a morphology-aware classification head for sperm normality assessment.  

To address severe class imbalance inherent in infertility-center datasets, ESCIMO employs weighted loss functions to stabilize training and improve minority-class sensitivity.

---

## Environment Specification

- Python 3.10  
- PyTorch 2.1.0  
- CUDA 11.8  
- Ubuntu 20.04 / 22.04  

---

## Dataset Structure

The dataset should be organized as follows:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The dataset used in this study is not publicly available due to privacy and ethical restrictions.  
Please refer to `datasets/README.md` for detailed annotation format specifications.

---

## Training

To train ESCIMO from scratch:

```bash
python train.py --config configs/miccai2026.yaml
```

---

## Evaluation

To evaluate using the best saved weights:

```bash
python test.py --config configs/miccai2026.yaml
```

---

## Expected Output Metrics (Test Set)

| Metric | Performance |
|--------|------------|
| ROC-AUC | 0.883 ± 0.012 |
| PR-AUC (AP) | 0.438 ± 0.015 |
| F2-Score | 0.620 ± 0.025 |
| Recall | 0.852 ± 0.011 |
| Adjusted Recall | 0.216 ± 0.022 |

The reported results correspond to Table 1 in the submitted manuscript.

---

## Reproducibility Statement

Random seeds for PyTorch, NumPy, and Python's `random` module are fixed (default `seed: 42` in `configs/miccai2026.yaml`) to ensure exact reproduction of the results presented in the manuscript.

All hyperparameters strictly follow the submitted version of the paper.

---

## Anonymous Citation Block
```bibtex
@inproceedings{anonymous2026sperm,
  title={An End-to-End Framework for Automated Sperm Normality Classification},
  author={Anonymous Authors},
  booktitle={MICCAI},
  year={2026}
}
```