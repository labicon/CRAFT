#!/bin/bash

# Modality ablation runner.
#
#   full        : snapshots + trajectory text in evaluation, reward-curve plots in advice (= CRAFT)
#   state_only  : trajectory text only, no images anywhere (numeric reward lists in advice)
#   vision_only : snapshots only, no trajectory text; advice keeps reward-curve plots
#
# Logs land in logs/ablation/<modality>/<task>/<MM-DD_HH-MM>/ so analysis/craft_eval.py
# can be pointed at one arm at a time.
#
# Usage: ./run_ablation.sh -M state_only -t go2gate -g 5 -s 1 -e 5

MODALITY="state_only"
TASK="go2seesaw"
GPU=0
SEED_START=1
SEED_END=5

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -M|--modality) MODALITY="$2"; shift ;;
        -t|--task) TASK="$2"; shift ;;
        -g|--gpu) GPU="$2"; shift ;;
        -s|--seed_start) SEED_START="$2"; shift ;;
        -e|--seed_end) SEED_END="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

case $MODALITY in
    full|state_only|vision_only) ;;
    *) echo "Unknown modality: $MODALITY (expected full|state_only|vision_only)"; exit 1 ;;
esac

# The eval subprocess renders camera frames on the sim GPU (base_task.py ignores
# --graphics_device_id and follows sim_device_id). On this machine cuda:1 segfaults in
# get_camera_image while every other GPU renders fine; training on it works, so the
# failure only shows up hours later as a missing traj_dict.pkl.
if [ "$GPU" = "1" ]; then
    echo "ERROR: GPU 1 cannot render camera frames on this machine (IsaacGym segfaults"
    echo "in the eval subprocess, though training works). Pick another GPU with -g."
    exit 1
fi

LOGDIR="logs/ablation/$MODALITY/$TASK"

echo "Ablation | Task: $TASK | Modality: $MODALITY | GPU: $GPU | Seeds: $SEED_START..$SEED_END"
echo "Log root: $LOGDIR"

for i in $(seq $SEED_START $SEED_END); do
    echo "========================================="
    echo "Starting seed $i ($MODALITY / $TASK)"
    echo "========================================="

    python -m curriculum.main --task $TASK --module curriculum --modality $MODALITY \
        --seed $i --gpu $GPU --logdir $LOGDIR

    echo "Seed $i finished at $(date)"

    # Run directories are named %m-%d_%H-%M and CurriculumAPI.get_advice globs TensorBoard
    # events by that name, so consecutive runs must not start within the same minute.
    if [ "$i" -lt "$SEED_END" ]; then
        sleep 61
    fi
    echo ""
done

echo "All seeds completed for $MODALITY / $TASK."
