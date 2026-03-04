# Dataset

The dataset used in this study is not publicly available due to privacy and ethical restrictions.

## Expected Directory Structure

```
dataset/
├── train/
│   ├── images/
│   ├── labels/
│   ├── obb/
│   └── obb_labels.csv
├── val/
│   ├── images/
│   ├── labels/
│   ├── obb/
│   └── obb_labels.csv
└── test/
    ├── images/
    ├── labels/
    ├── obb/
    └── obb_labels.csv
```

## Annotation Format

Each subset (train/val/test) contains:
- `images/`: High-resolution microscopy images.
- `labels/`: YOLO-format bounding box labels.
- `obb_labels.csv`: Oriented Bounding Box (OBB) labels with normality labels (0: abnormal, 1: normal).
- `obb/`: Cropped sperm images based on alignment and OBB.

## Data Access

Access to the dataset may be granted upon reasonable request and subject to institutional approval.