python ./openrl_ws/go2gate_eval.py \
    --task go2gate \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint "/home/kang/mqe-curriculum/logs/go2gate/08-14_01-33_randomized/3_Navigate to Goal Position(08-14_01-33)/sample_1/model" \
    # --headless \
    # --record_video 