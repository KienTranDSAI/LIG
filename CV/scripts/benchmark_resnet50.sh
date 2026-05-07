#!/bin/bash
# Quick benchmark: ResNet50 only, zeros baseline, 6 methods, 50 images.
source "$(dirname "$0")/_common.sh"

OUTDIR="${OUTDIR:-results/benchmark_resnet50}"

python -u run_benchmark.py \
  --models resnet50 \
  --methods ig idgi guided_ig lig \
  --metrics insertion deletion ins-del \
  --n-test 50 --steps 50 --seed 42 --min-conf 0.70 \
  --device "$DEVICE" \
  --image-dir benchmark_50 \
  --outdir "$OUTDIR"
