#!/bin/bash

# Number of runs
NUM_RUNS=5

echo "Starting $NUM_RUNS experimental runs..."

# Run main.py multiple times
for i in $(seq 1 $NUM_RUNS); do
    echo "========================================="
    echo "Starting run $i of $NUM_RUNS"
    echo "========================================="
    
    # Run main.py
    python curriculum/main.py
    
    echo "Run $i finished at $(date)"
    echo ""
done

echo "All $NUM_RUNS runs completed!"
