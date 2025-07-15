python ./openrl_ws/test.py \
    --task go1gate \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint "/home/kang/multiagent-quadruped-environment/runs/07-15_15-22/3_Follower gate passing(07-15_15-22)/sample_0/module.pt" \
    # --record_video