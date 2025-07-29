python ./openrl_ws/test.py \
    --task go2gate \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint "/home/kang/multiagent-quadruped-environment/checkpoints/go2gate07/27/2025-15:16:10/module.pt" \
    # --record_video 