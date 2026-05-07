#!/bin/bash
# EG vs EG+LIG: ResNet50, 50 images, 5-baseline pool.
# EG     = mean of IG attributions across {black, white, noise, blur, mean_corners}
# EG+LIG = mean of LIG attributions across the same pool
# Insertion/deletion reference: black (same for both methods).
source "$(dirname "$0")/_common.sh"

MODEL="${MODEL:-resnet50}"
OUTJSON="${OUTJSON:-results/eg_vs_eglig/${MODEL}.json}"
OUTMD="${OUTMD:-benchmark_eg_vs_eglig.md}"

python -u run_eg_benchmark.py \
  --model "$MODEL" \
  --n-test 50 --steps 50 --seed 42 --min-conf 0.70 \
  --device "$DEVICE" \
  --image-dir benchmark_50 \
  --json "$OUTJSON" \
  --markdown "$OUTMD"
