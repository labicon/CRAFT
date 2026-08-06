python ./openrl_ws/eval.py \
    --task go2gate \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint "/home/kanghyun/mqe-curriculum/baselines/checkpoints/go2gate/go2gate_05-11_19-42_scratch/rl_model_80000000_steps" \
    --record_video \
    # --headless \