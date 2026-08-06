export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD
# task="go1football-defender"
task="go2gate"
# task="go2switch"
# task="go1seesaw"
# task="go1sheep-easy"
# task="go1sheep-hard"
# task="go1pushbox-plane"
# task="go1pushbox-gate"
# random_seed=0
device=0
num_envs=500
num_steps=80000000

# algo="jrpo"
# cfg=./openrl_ws/cfgs/jrpo.yaml
algo="ppo"
# cfg=./openrl_ws/cfgs/ppo.yaml
# algo="mat"
# cfg=./openrl_ws/cfgs/mat.yaml
# algo="sppo"
# cfg=./openrl_ws/cfgs/ppo.yaml
# algo="dppo"
# cfg=./openrl_ws/cfgs/dppo.yaml

# Number of runs
NUM_RUNS=1

for run in $(seq 1 $NUM_RUNS); do
    echo "========================================="
    echo "Starting run $run of $NUM_RUNS"
    echo "========================================="

    python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
        --task $task \
        --algo $algo \
        --sim_device cuda:$device \
        --rl_device cuda:$device \
        --seed 0 \
        --exp_name go2gate-from-scratch-05-13_01-02 \
        --use_wandb

    echo "Run $run finished at $(date)"
    echo ""
done
echo "All $NUM_RUNS runs completed!"