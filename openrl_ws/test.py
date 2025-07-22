import isaacgym
from openrl_ws.utils import make_env, get_args, MATWrapper
from mqe.envs.utils import custom_cfg

from openrl.envs.common import make
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import cv2
import imageio
import numpy as np
import os

def save_video(frames, fps, output_path='output_video.mp4'):
    # Assuming your ndarray is named 'frames'
    # frames.shape = (134, 4, 240, 360)

    # Define the video codec and frame rate
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    # Get the shape of a single frame
    frame_shape = frames.shape[2], frames.shape[3]

    frames = frames[:, :3, :, :]
    frames = np.transpose(frames, (0, 2, 3, 1))

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_path, codec, fps, frame_shape)

    # Iterate through each frame
    for i in range(len(frames)):
        # Convert frame to uint8 (assuming it's in range 0-255)
        frame = frames[i]
        frame = frame.astype(np.uint8)
        
        # Transpose frame from (4, 240, 360) to (240, 360, 3) if needed
        # frame = np.transpose(frame, (1, 2, 0))
        
        # Write the frame to the video file
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print("Video created successfully.")

def save_gif(frames, fps, output_path='output_animation.gif'):

    # Assuming your ndarray is named 'frames'
    # frames.shape = (134, 4, 240, 360)

    # Convert the frames to uint8 (assuming it's in range 0-1)
    frames = np.transpose(frames, (0, 2, 3, 1))
    frames_uint8 = frames.astype(np.uint8)

    frames = [frames_uint8[i] for i in range(len(frames_uint8))]

    # Save frames as GIF
    imageio.mimsave(output_path, frames, duration=2)

    print(f"GIF saved successfully to '{output_path}'.")

def save_images(frames, output_dir='output_images'):
    """Saves a sequence of frames as individual image files."""
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Transpose frames from (N, C, H, W) to (N, H, W, C)
    frames = np.transpose(frames, (0, 2, 3, 1))
    frames_uint8 = frames.astype(np.uint8)

    # Skip the first 5 frames (they are usually wrong)
    skip_frames = 5
    if len(frames_uint8) > skip_frames:
        frames_uint8 = frames_uint8[skip_frames:]
    
    # Calculate indices for 5 evenly spaced frames from the remaining frames
    num_frames = len(frames_uint8)
    if num_frames <= 5:
        # Save all remaining frames if there are 5 or fewer
        selected_indices = list(range(num_frames))
    else:
        # Select 5 evenly spaced frames from the remaining frames
        selected_indices = [int(i * (num_frames - 1) / 4) for i in range(5)]

    # Save selected frames as PNG images
    for i, frame_idx in enumerate(selected_indices):
        frame = frames_uint8[frame_idx]
        # Adjust frame index to account for skipped frames
        actual_frame_idx = frame_idx + skip_frames
        output_path = os.path.join(output_dir, f'snapshot_{actual_frame_idx:04d}.png')
        imageio.imwrite(output_path, frame)

    print(f"Successfully saved {len(selected_indices)} images to '{output_dir}'.")


if __name__ == "__main__":
    args = get_args()
    env, _ = make_env(args, custom_cfg(args))

    # args.config = "./openrl_ws/cfgs/ppo.yaml" # Somehow this does not work
    args.lr = 7e-4
    args.critic_lr = 7e-4
    args.log_interval = 5
    args.use_recurrent_policy = True
    args.use_joint_action_loss = False
    args.use_valuenorm = True
    args.use_adv_normalize = True

    net = PPONet(env, device="cuda")  # Create neural network.
    agent = PPOAgent(net)  # Initialize the agent.

    if args.algo == "jrpo" or args.algo == "ppo":
        from openrl.modules.common import PPONet
        from openrl.runners.common import PPOAgent
        net = PPONet(env, cfg=args, device=args.rl_device)
        agent = PPOAgent(net)
    else:
        from openrl.modules.common import MATNet
        from openrl.runners.common import MATAgent
        env = MATWrapper(env)
        net = MATNet(env, cfg=args, device=args.rl_device)
        agent = MATAgent(net, use_wandb=args.use_wandb)

    if getattr(args, "checkpoint") is not None:
        agent.load(args.checkpoint)

    # env.start_recording()
    agent.set_env(env)  # The agent requires an interactive environment.
    obs = env.reset()  # Initialize the environment to obtain initial observations and environmental information.
    while True:
        action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
        # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
        obs, r, done, info = env.step(action)
        print("Obs:", obs)
        print("Action:", action)
        print("Target pos:", env.target_pos)
        print("Gate pos:", env.gate_pos)
        # print("Info:", info)
        if done[0, 0] and env.cfg.env.record_video:
            frames = env.get_complete_frames()
            video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
            # print(video_array.shape)
            save_gif(video_array, 200)
            save_images(video_array)
