export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD

python ./openrl_ws/test.py \
    --task go2pushbox \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint "/home/kang/CRAFT/logs/02-22_03-42/1_Head Contact Establishment(02-22_03-42)/sample_0/model/module.pt" \
    # --record_video 