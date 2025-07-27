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
    # Get the shape info for debugging
    height, width = frames.shape[2], frames.shape[3]
    
    # Process frames for video creation
    frames = frames[:, :3, :, :]  # Take only RGB channels
    frames = np.transpose(frames, (0, 2, 3, 1))  # Convert to (N, H, W, C)

    # Debug: Print frame info
    print(f"Frame shape: {frames.shape}")
    print(f"Frame dtype: {frames.dtype}")
    print(f"Frame min/max: {frames.min():.3f}/{frames.max():.3f}")

    # Prepare frames for imageio
    imageio_frames = []
    for i in range(len(frames)):
        frame = frames[i]
        
        # Ensure frame values are in 0-255 range
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        
        # Ensure frame has exactly 3 channels
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        
        imageio_frames.append(frame)
    
    try:
        # Save as MP4 using imageio
        imageio.mimsave(output_path, imageio_frames, fps=fps, codec='libx264')
        print(f"Video created successfully at '{output_path}'")
    except Exception as e:
        print(f"Video creation failed: {e}")
        # Fallback to saving as image sequence
        print("Saving as image sequence instead...")
        # Convert back to original format for save_images
        original_frames = np.transpose(np.array(imageio_frames), (0, 3, 1, 2))
        save_images(original_frames, output_dir='failed_video_frames')

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
        output_path = os.path.join(output_dir, f'snapshot_{i}.png')
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
    seed = np.random.randint(0, 10000)
    obs = env.reset(seed=seed)  # Initialize the environment to obtain initial observations and environmental information.
    while True:
        action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
        # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
        obs, r, done, info = env.step(action)
        # print("Obs:", obs)
        # print("Action:", action)
        # print("Target pos:", env.target_pos)
        # print("Gate pos:", env.gate_pos)
        print("agent distance: ", np.linalg.norm(obs[0, 0, 2:4] - obs[0, 1, 2:4]))
        # for key, value in info[0].items():
        #     print(f"{key}: {value}")
        if done[0, 0]:
            if np.linalg.norm(obs[0, 0, 2:4] - obs[0, 1, 2:4]) < 0.51:
                print("Terminated due to close proximity")
            else:
                print("Terminated due to other reasons")
        if done[0, 0] and env.cfg.env.record_video:
            frames = env.get_complete_frames()
            video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
            # print(video_array.shape)
            # save_gif(video_array, 200)
            save_video(video_array, 50)
            save_images(video_array)
