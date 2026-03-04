# Weights Directory

This directory stores the trained model weights.

- `best_model.pt`: The classification model weights that achieved the minimum loss on the validation set.
- `yolo_detector.pt`: The specialized YOLO11n-based detection model used for sperm localization in the end-to-end pipeline.

Note: Pre-trained weights for the CNN backbone (e.g., ImageNet) are automatically downloaded during the first execution.
