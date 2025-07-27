python ./openrl_ws/test.py \
    --task go2gate \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint "/home/kang/multiagent-quadruped-environment/logs/07-25_01-57/3_Reach the Final Target(07-25_01-57)/sample_2/model/module.pt" \
    # --record_video 