import gymnasium as gym
import numpy as np
import sapien
import os
import datetime
import imageio
from pathlib import Path

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = "panda"
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

    save_dir: Annotated[Optional[str], tyro.conf.arg(aliases=["-d"])] = "saved_observations"
    """Directory to save RGBD observations and camera poses when pressing the V key"""


def save_observation(obs, save_dir: Path, args: Args):
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Saving RGBD observation to {save_dir}")
    
    # Save observation data if in visual mode
    if args.obs_mode in ["rgb", "rgbd", "sensor_data"]:
        # Save sensor data (RGB, depth, and camera parameters)
        for cam_name, cam_data in obs["sensor_data"].items():
            # Save RGB image
            if "rgb" in cam_data:
                rgb_img = cam_data["rgb"][0].cpu().numpy()  # Shape: (H, W, 3)
                rgb_path = save_dir / f"{cam_name}_rgb.png"
                imageio.imwrite(rgb_path, rgb_img)
            
            # Save depth image
            if "depth" in cam_data:
                depth_img = cam_data["depth"][0].cpu().numpy()  # Shape: (H, W, 1)
                depth_path = save_dir / f"{cam_name}_depth.npy"
                np.save(depth_path, depth_img)

                depth_norm_path = save_dir / f"{cam_name}_depth_norm.png"
                # Normalize depth for visualization
                depth_norm = (depth_img / np.max(depth_img) * 255).astype(np.uint8)
                depth_norm = np.repeat(depth_norm, 3, axis=2)
                imageio.imwrite(depth_norm_path, depth_norm)

            # Save camera parameters
            if cam_name in obs["sensor_param"]:
                cam_params = obs["sensor_param"][cam_name]
                # Save extrinsic matrix
                if "extrinsic_cv" in cam_params:
                    np.save(
                        save_dir / f"{cam_name}_extrinsic.npy", 
                        cam_params["extrinsic_cv"][0].cpu().numpy()
                    )
                # Save intrinsic matrix
                if "intrinsic_cv" in cam_params:
                    np.save(
                        save_dir / f"{cam_name}_intrinsic.npy", 
                        cam_params["intrinsic_cv"][0].cpu().numpy()
                    )
                # Save camera-to-world matrix
                if "cam2world_gl" in cam_params:
                    np.save(
                        save_dir / f"{cam_name}_cam2world.npy", 
                        cam_params["cam2world_gl"][0].cpu().numpy()
                    )
        
        print(f"Saved observation data to {save_dir}")
    else:
        print(f"Cannot save visual data: observation mode '{args.obs_mode}' doesn't provide image data")
        print(f"Use 'rgbd' or 'sensor_data' observation mode to save images")


def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        robot_uids=args.robot_uids,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    
    # Create save directory for RGBD observations
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    else:
        save_dir = None
    
    observation_count = 0
    
    while True:
        action = env.action_space.sample() if env.action_space is not None else None
        # action *= 0
        obs, reward, terminated, truncated, info = env.step(action)
        if verbose:
            print("reward", reward)
            print("terminated", terminated)
            print("truncated", truncated)
            print("info", info)
        if args.render_mode is not None:
            env.render()
        
        # Check for V key press to save observation
        if args.render_mode == "human" and isinstance(viewer, sapien.utils.Viewer) and viewer.window.key_press("v"):
            observation_count += 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            obs_dir = save_dir / f"observation_{timestamp}_{observation_count}"
            save_observation(obs, obs_dir, args)
        
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)