from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, XArm6Robotiq, XArm6AllegroLeft, XArm6AllegroRight, FloatingRobotiq2F85Gripper, XArm6PandaGripper
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("XembCalibration-v1", max_episode_steps=50)
class XembCalibrationEnv(BaseEnv):
    """
    **Task Description:**
    A calibration task where the objective is to reach each goal position in order of x, y, z and go back to the original position.

    **Success Conditions:**
    - If the tcp reaches within `goal_thresh` (default 0.04m) euclidean distance of the each goal position in the right order, the task is considered successful.
    - the robot is static at the end of the episode (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "xarm6_allegro_left",
        "xarm6_allegro_right",
        "floating_robotiq_2f_85_gripper",
        "xarm6_pandagripper"
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, XArm6AllegroLeft, XArm6AllegroRight, FloatingRobotiq2F85Gripper, XArm6PandaGripper]
    goal_thresh = 0.04
    calibration_delta = 0.1
    axes = ['x', 'y', 'z']

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.goal_positions = {}
        for i, axis in enumerate(self.axes):
            positions = [np.array([0, 0, 0.2]) for _ in range(2)]
            positions[0][i] += self.calibration_delta
            positions[1][i] -= self.calibration_delta
            self.goal_positions[axis] = positions

        self.goal_sites = []
        for axis in self.axes:
            for i, pos in enumerate(self.goal_positions[axis]):
                goal_site = actors.build_sphere(
                    self.scene,
                    radius=self.goal_thresh,
                    color=[0, 1, 0, 0.5],
                    name=f"goal_site_{axis}_{i+1}",
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose(p=pos),
                )
                self.goal_sites.append(goal_site)

        self._hidden_objects.extend(self.goal_sites)
        # Dictionary to store goal indexing
        self.goal_name_to_idx = {site: i for i, site in enumerate(self.goal_sites)}
        # This will be initialized properly in _initialize_episode

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            # Initialize tensor to track which goals have been reached
            # Shape: [n_envs, n_goals]
            self.reached_goal_tracker = torch.zeros(
                (self.num_envs, len(self.goal_sites)), 
                dtype=torch.bool, 
                device=self.device
            )
            # Initialize tensor to track current goal index for each env
            self.current_goal_idx = torch.zeros(
                self.num_envs, 
                dtype=torch.long, 
                device=self.device
            )
            # Create a tensor to track the stage rewards (reset at the beginning of each episode)
            # This stores the accumulated reward per env to add to the total
            self.stage_rewards = torch.zeros(
                (self.num_envs),
                dtype=torch.float32,
                device=self.device
            )

    def _get_obs_extra(self, info: Dict):
        return {
            "current_goal_idx": self.current_goal_idx,
        }

    def evaluate(self):
        # Check if all goals have been reached in sequence
        # We need to make sure the robot is also static at the end
        qvel = self.agent.robot.get_qvel()
        is_robot_static = torch.max(torch.abs(qvel), dim=1)[0] <= 0.2
        
        # Check if all goals have been reached for each environment
        all_goals_reached = torch.all(self.reached_goal_tracker, dim=1)
        
        return {
            "success": all_goals_reached & is_robot_static,
            "is_robot_static": is_robot_static,
            "all_goals_reached": all_goals_reached,
        }

    def _get_goal_poses_tensor(self):
        """Convert goal sites' positions to a tensor for easier processing"""
        if not hasattr(self, 'goal_poses_tensor'):
            # Create a tensor of shape [num_goals, 3] with all goal positions
            self.goal_poses_tensor = torch.stack([
                torch.tensor(goal.pose.p, device=self.device) 
                for goal in self.goal_sites
            ]).squeeze(1)
        return self.goal_poses_tensor

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Get TCP position
        tcp_pose = self.agent.tcp.pose.p
        
        # Base reward for all environments
        # Start with stage rewards from previously completed stages
        rewards = self.stage_rewards.clone()
        
        # Create masks for different conditions
        all_goals_reached_mask = self.current_goal_idx >= len(self.goal_sites)
        goals_in_progress_mask = ~all_goals_reached_mask
        
        # ===== PROCESS ENVIRONMENTS WHERE ALL GOALS ARE REACHED =====
        if torch.any(all_goals_reached_mask):
            # Calculate robot static reward
            qvel = self.agent.robot.get_qvel()
            qvel_norm = torch.linalg.norm(qvel, axis=1)
            static_reward = 1 - torch.tanh(5 * qvel_norm)
            
            # Apply static reward to environments with all goals reached
            robot_static_mask = info["is_robot_static"] & all_goals_reached_mask
            robot_not_static_mask = ~info["is_robot_static"] & all_goals_reached_mask
            
            # Add partial static reward for robots not yet static
            if torch.any(robot_not_static_mask):
                rewards[robot_not_static_mask] += static_reward[robot_not_static_mask]
                
            # Add full static reward (1.0) for robots that are static
            if torch.any(robot_static_mask):
                rewards[robot_static_mask] += 1.0
        
        # ===== PROCESS ENVIRONMENTS STILL REACHING GOALS =====
        if torch.any(goals_in_progress_mask):
            # Get goal positions as a tensor [num_goals, 3]
            goal_poses = self._get_goal_poses_tensor()
            
            # Identify environments still in progress
            in_progress_indices = torch.where(goals_in_progress_mask)[0]
            
            # Get next goal index for each environment in a tensor
            next_goal_indices = self.current_goal_idx[in_progress_indices].long()
            
            # Get the corresponding goal position for each environment
            # by indexing into the goal_poses tensor
            next_goal_positions = goal_poses[next_goal_indices]
            
            # Calculate distances between TCPs and their respective goals
            tcp_positions = tcp_pose[in_progress_indices]
            tcp_to_goal_dists = torch.linalg.norm(
                next_goal_positions - tcp_positions, axis=1
            )
            
            # Check which environments have reached their goals
            goals_reached_mask = tcp_to_goal_dists <= self.goal_thresh
            
            # For environments that reached their goals:
            if torch.any(goals_reached_mask):
                # Get the indices of environments that reached their goals
                reached_env_indices = in_progress_indices[goals_reached_mask]
                reached_goal_indices = next_goal_indices[goals_reached_mask]
                
                # Update reached_goal_tracker for these environments
                # Using advanced indexing to directly update the appropriate elements
                self.reached_goal_tracker[reached_env_indices, reached_goal_indices] = True
                
                # Add fixed reward for completing this stage (1.0 per goal)
                self.stage_rewards[reached_env_indices] += 1.0
                
                # Move to the next goal
                self.current_goal_idx[reached_env_indices] += 1
            
            # For environments that haven't reached their goals:
            if torch.any(~goals_reached_mask):
                # Calculate reaching reward (partial progress toward the current goal)
                reaching_reward = 1 - torch.tanh(5 * tcp_to_goal_dists[~goals_reached_mask])
                
                # Apply reaching reward to the appropriate environments
                rewards[in_progress_indices[~goals_reached_mask]] += reaching_reward
        
        return rewards

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # Normalize between 0 and 1 by dividing by maximum possible reward (7.0)
        max_reward = 7.0  # 6 goals + 1 for static
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
