#!/bin/bash
# Full benchmark: 9 backbones × 6 methods × 50 fixed images (zeros baseline).
source "$(dirname "$0")/_common.sh"

OUTDIR="${OUTDIR:-results/benchmark_all}"

python -u run_benchmark.py \
  --methods ig idgi guided_ig lig \
  --metrics insertion deletion ins-del \
  --n-test 50 --steps 50 --seed 42 --min-conf 0.70 \
  --device "$DEVICE" \
  --image-dir benchmark_50 \
  --outdir "$OUTDIR"
