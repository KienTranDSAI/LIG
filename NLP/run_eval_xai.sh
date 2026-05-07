#!/bin/bash

DATASET="sst2"
N=200
BACKBONES=("roberta" "bert" "distilbert")
LOG_FILE="eval_xai_results_$(date +%Y%m%d_%H%M%S).log"

echo "Starting eval_xai_metrics.py runs" | tee "$LOG_FILE"
echo "Dataset: $DATASET | N: $N | Backbones: ${BACKBONES[*]}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

for backbone in "${BACKBONES[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running backbone: $backbone" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    python eval_xai_metrics.py \
        --backbone "$backbone" \
        --dataset "$DATASET" \
        --n "$N" \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: backbone=$backbone exited with code $EXIT_CODE" | tee -a "$LOG_FILE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: backbone=$backbone" | tee -a "$LOG_FILE"
    fi

    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "All runs complete. Log saved to: $LOG_FILE"