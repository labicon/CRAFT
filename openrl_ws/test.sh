python ./openrl_ws/test.py \
    --task go1gate \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint "/home/kang/multiagent-quadruped-environment/runs/07-15_16-13/1_Leader gate passing(07-15_16-13)/sample_0/rl_model_7000000_steps/module.pt" \
    --record_video