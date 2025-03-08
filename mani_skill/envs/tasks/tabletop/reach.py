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


@register_env("Reach-v1", max_episode_steps=50)
class ReachEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to bring the tcp to a goal position.

    **Randomizations:**
    - the target goal (marked by a green sphere) has its position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.2]

    **Success Conditions:**
    - the tcp position is within `goal_thresh` (default 0.005m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "xarm6_allegro_left",
        "xarm6_allegro_right",
        "floating_robotiq_2f_85_gripper",
        "xarm6_pandagripper",
        "xarm6_nogripper"
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, XArm6AllegroLeft, XArm6AllegroRight, FloatingRobotiq2F85Gripper, XArm6PandaGripper]
    goal_thresh = 0.01

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        default_sensor_configs = []

        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        default_sensor_configs.append(CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100))

        pose = sapien_utils.look_at(eye=[0.212, 0.212, 0.6], target=[-0.1, 0, 0.1])
        default_sensor_configs.append(CameraConfig("third_person", pose, 128, 128, np.pi / 2, 0.01, 100))

        return default_sensor_configs

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
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        # self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz = torch.rand((b, 3)) * 0.3 - 0.15
            xyz[:, 2] += 0.02 + 0.15
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            self.goal_site.set_pose(Pose.create_from_pq(xyz, qs))

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_to_goal_dist=self.agent.tcp.pose.p - self.goal_site.pose.p
        )
        return obs

    def evaluate(self):
        # Check if the robot eef has reached the goal
        has_reached = (
            torch.linalg.norm(self.goal_site.pose.p - self.agent.tcp.pose.p, axis=1)
            <= self.goal_thresh
        )
        qvel = self.agent.robot.get_qvel()
        is_robot_static = torch.max(torch.abs(qvel), dim=1)[0] <= 0.2
        return {
            "success": has_reached & is_robot_static,
            "has_reached": has_reached,
            "is_robot_static": is_robot_static,
        }

    def staged_rewards(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_goal_dist)

        qvel = self.agent.robot.get_qvel()
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel, axis=1)
        )
        static_reward *= info["has_reached"]

        return reaching_reward.mean(), static_reward.mean()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_goal_dist)
        reward = reaching_reward

        qvel = self.agent.robot.get_qvel()
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(10 * qvel, axis=1)
        )
        reward += static_reward * info["has_reached"]

        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3
