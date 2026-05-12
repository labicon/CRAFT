#!/bin/bash

# Number of runs
NUM_RUNS=5
ENV_NAME="go2gate"
MANUAL=false
GPU=5

# Parse flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--manual) MANUAL=true ;;
        -g|--gpu) GPU="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

if $MANUAL; then
    MODULE="manual"
else
    MODULE="curriculum"
fi

echo "Starting $NUM_RUNS experimental runs using curriculum.main..."
echo "Task: $ENV_NAME | Module: $MODULE | GPU: $GPU"

# Run the selected script multiple times
for i in $(seq 1 $NUM_RUNS); do
    echo "========================================="
    echo "Starting run $i of $NUM_RUNS"
    echo "========================================="

    python -m curriculum.main --task $ENV_NAME --module $MODULE --seed $i --gpu $GPU

    echo "Run $i finished at $(date)"
    echo ""
done

echo "All $NUM_RUNS runs completed!"
