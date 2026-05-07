export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD

python ./openrl_ws/test.py \
    --task go2pushbox \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --seed 9 \
    --num_envs 1 --checkpoint "/home/kang/CRAFT/logs/03-21_21-51/4_Final task(03-21_21-51)/sample_1/model/module.pt" \
    --record_video 