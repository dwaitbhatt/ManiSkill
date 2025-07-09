from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link


@register_agent()
class Reachy2StaticOnehand(BaseAgent):
    uid = "reachy2_static_onehand"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/reachy/reachy2_static.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            r_gripper_finger_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            l_gripper_finger_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            qpos=np.zeros(21),
        ),
        right_hand_out=Keyframe(
            pose=sapien.Pose(),
            qpos=([
                -1.6, -1.2,   # r_shoulder_pitch, r_shoulder_roll
                0,            # neck_roll
                0, 0,         # r_shoulder_roll, l_shoulder_roll
                0,            # neck_pitch
                0, 0,         # r_arm_yaw, l_arm_yaw
                0,            # neck_yaw
                0, 0,         # r_elbow_pitch, l_elbow_pitch
                0, 0,         # r_forearm_yaw, l_forearm_yaw
                0, 0,         # l_antenna, r_antenna
                0, 0,         # r_wrist_pitch, l_wrist_pitch
                0, 0,         # r_wrist_roll, l_wrist_roll
                -0.35, 1.2,   # r_gripper, l_gripper
            ]),
        ),
    )

    def __init__(self, *args, **kwargs):
        self.r_arm_joint_names = [
            "r_shoulder_pitch",
            "r_shoulder_roll",
            "r_arm_yaw",
            "r_elbow_pitch",
            "r_forearm_yaw",
            "r_wrist_pitch",
            "r_wrist_roll",
        ]
        self.l_arm_joint_names = []
        self.all_arm_joint_names = self.r_arm_joint_names

        self.gripper_joint_names = [
            "r_gripper",
            "l_gripper",
        ]

        self.head_joint_names = []

        self.l_ee_link_name = "l_arm_tip"
        self.r_ee_link_name = "r_arm_tip"
        self.camera_link_name = "camera_link"

        self.arm_stiffness = 1e7
        self.arm_damping = 1e6 # 0.1
        self.arm_force_limit = 1e9

        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2 # 0.1
        self.gripper_force_limit = 100

        self.head_stiffness = 1e3
        self.head_damping = 1e2 # 0.1
        self.head_force_limit = 100

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
       # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arms_pd_joint_pos = PDJointPosControllerConfig(
            self.all_arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arms_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.all_arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arms_pd_joint_target_delta_pos = deepcopy(arms_pd_joint_delta_pos)
        arms_pd_joint_target_delta_pos.use_target = True


        # PD ee position for right arm
        r_arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.r_arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.r_ee_link_name,
            urdf_path=self.urdf_path,
        )
        r_arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.r_arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.r_ee_link_name,
            urdf_path=self.urdf_path,
        )

        r_arm_pd_ee_target_delta_pos = deepcopy(r_arm_pd_ee_delta_pos)
        r_arm_pd_ee_target_delta_pos.use_target = True
        r_arm_pd_ee_target_delta_pose = deepcopy(r_arm_pd_ee_delta_pose)
        r_arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        r_arm_pd_ee_delta_pose_align = deepcopy(r_arm_pd_ee_delta_pose)
        r_arm_pd_ee_delta_pose_align.frame = "r_ee_align"


        # PD ee position for left arm
        l_arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.l_arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.l_ee_link_name,
            urdf_path=self.urdf_path,
        )
        l_arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.l_arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.l_ee_link_name,
            urdf_path=self.urdf_path,
        )

        l_arm_pd_ee_target_delta_pos = deepcopy(l_arm_pd_ee_delta_pos)
        l_arm_pd_ee_target_delta_pos.use_target = True
        l_arm_pd_ee_target_delta_pose = deepcopy(l_arm_pd_ee_delta_pose)
        l_arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        l_arm_pd_ee_delta_pose_align = deepcopy(l_arm_pd_ee_delta_pose)
        l_arm_pd_ee_delta_pose_align.frame = "l_ee_align"


        # PD joint velocity
        arms_pd_joint_vel = PDJointVelControllerConfig(
            self.all_arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arms_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.all_arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=True,
        )
        arms_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.all_arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.01,  # a trick to have force when the object is thin
            0.05,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arms=arms_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_pos=dict(
                arms=arms_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pos=dict(
                r_arm=r_arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pose=dict(
                r_arm=r_arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pose_align=dict(
                r_arm=r_arm_pd_ee_delta_pose_align,
                gripper=gripper_pd_joint_pos,
            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arms=arms_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_target_delta_pos=dict(
                r_arm=r_arm_pd_ee_target_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_target_delta_pose=dict(
                r_arm=r_arm_pd_ee_target_delta_pose,
                gripper=gripper_pd_joint_pos,
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arms=arms_pd_joint_vel,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_pos_vel=dict(
                arms=arms_pd_joint_pos_vel,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_delta_pos_vel=dict(
                arms=arms_pd_joint_delta_pos_vel,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_delta_pos_stiff_body=dict(
                arms=arms_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.l_finger_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_gripper_finger"
        )
        self.l_thumb_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_gripper_thumb"
        )
        self.l_tcp: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_arm_tip"
        )

        self.r_finger_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_gripper_finger"
        )
        self.r_thumb_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_gripper_thumb"
        )
        self.r_tcp: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_arm_tip"
        )
        self.tcp = self.r_tcp

        self.head_camera_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.camera_link_name
        )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        # Check if the left hand is grasping
        l_finger_contact_forces = self.scene.get_pairwise_contact_forces(
            self.l_finger_link, object
        )
        l_thumb_contact_forces = self.scene.get_pairwise_contact_forces(
            self.l_thumb_link, object
        )
        l_finger_force = torch.linalg.norm(l_finger_contact_forces, axis=1)
        l_thumb_force = torch.linalg.norm(l_thumb_contact_forces, axis=1)

        # direction to open the gripper
        lfdirection = -self.l_finger_link.pose.to_transformation_matrix()[..., :3, 1]
        ltdirection = self.l_thumb_link.pose.to_transformation_matrix()[..., :3, 1]
        lfangle = common.compute_angle_between(lfdirection, l_finger_contact_forces)
        ltangle = common.compute_angle_between(ltdirection, l_thumb_contact_forces)
        lf_flag = torch.logical_and(
            l_finger_force >= min_force, torch.rad2deg(lfangle) <= max_angle
        )
        lt_flag = torch.logical_and(
            l_thumb_force >= min_force, torch.rad2deg(ltangle) <= max_angle
        )

        is_left_grasping = torch.logical_and(lf_flag, lt_flag)

        # Check if the right hand is grasping
        r_finger_contact_forces = self.scene.get_pairwise_contact_forces(
            self.r_finger_link, object
        )
        r_thumb_contact_forces = self.scene.get_pairwise_contact_forces(
            self.r_thumb_link, object
        )
        r_finger_force = torch.linalg.norm(r_finger_contact_forces, axis=1)
        r_thumb_force = torch.linalg.norm(r_thumb_contact_forces, axis=1)

        rfdirection = -self.r_finger_link.pose.to_transformation_matrix()[..., :3, 1]
        rtdirection = self.r_thumb_link.pose.to_transformation_matrix()[..., :3, 1]
        rfangle = common.compute_angle_between(rfdirection, r_finger_contact_forces)
        rtangle = common.compute_angle_between(rtdirection, r_thumb_contact_forces)
        rf_flag = torch.logical_and(
            r_finger_force >= min_force, torch.rad2deg(rfangle) <= max_angle
        )
        rt_flag = torch.logical_and(
            r_thumb_force >= min_force, torch.rad2deg(rtangle) <= max_angle
        )

        is_right_grasping = torch.logical_and(rf_flag, rt_flag)

        return torch.logical_or(is_left_grasping, is_right_grasping)

    def is_static(self, threshold: float = 0.2):
        body_qvel = self.robot.get_qvel()[..., 3:-2]
        return torch.all(body_qvel <= threshold, dim=1)

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    @property
    def tcp_pose(self) -> Tuple[Pose, Pose]:
        return (self.l_tcp.pose, self.r_tcp.pose)
