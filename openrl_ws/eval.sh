python ./openrl_ws/eval.py \
    --task go2gate \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint "/home/kang/multiagent-quadruped-environment/logs/07-30_18-12/3_Target Navigation Post-Gate(07-30_18-12)/sample_2/model/module.pt" \
    # --headless \
    # --record_video 