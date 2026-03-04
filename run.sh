#!/bin/bash

# MICCAI 2026 Anonymous Release - Full Pipeline Execution Script
# This script reproduces the training and evaluation steps described in the manuscript.

set -e

echo "Starting full pipeline reproduction..."

# 1. Training
echo "Running Training Phase..."
python train.py --config configs/miccai2026.yaml

# 2. Evaluation
echo "Running Evaluation Phase..."
python test.py --config configs/miccai2026.yaml

echo "Pipeline execution completed successfully."
