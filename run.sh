#!/bin/bash

# MICCAI 2026 Anonymous Release - Full Pipeline Execution Script
# This script reproduces the training and evaluation steps described in the manuscript.

set -e

echo "Starting full pipeline reproduction..."

# 0. Data Preparation
if [ ! -d "dataset" ]; then
    echo "Dataset not found. Generating dummy dataset for verification..."
    python tools/generate_dummy_data.py
fi

# 1. Training
echo "Running Training Phase..."
python train.py --config configs/miccai2026.yaml

# 2. Evaluation
echo "Running Evaluation Phase..."
python test.py --config configs/miccai2026.yaml

echo "Pipeline execution completed successfully."
