#!/bin/bash
export PYTHONPATH=$PWD

# Default script to run
SCRIPT="curriculum/main.py"

# Number of runs
NUM_RUNS=5
ENV_NAME="go2pushbox"

echo "Starting $NUM_RUNS experimental runs using $SCRIPT..."

# Run the selected script multiple times
for i in $(seq 1 $NUM_RUNS); do
    echo "========================================="
    echo "Starting run $i of $NUM_RUNS"
    echo "========================================="
    
    # Run the selected python script
    python $SCRIPT --task $ENV_NAME --seed $i
    
    echo "Run $i finished at $(date)"
    echo ""
done

echo "All $NUM_RUNS runs completed!"
