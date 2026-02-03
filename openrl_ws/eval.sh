python ./openrl_ws/go2seesaw_eval.py \
    --task go2seesaw \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint "/home/kanghyun/mqe-curriculum/logs/go2seesaw/08-19_02-44/2_Seesaw Entry and Stability Task(08-19_02-44)/sample_0/model" \
    --record_video \
    # --headless \