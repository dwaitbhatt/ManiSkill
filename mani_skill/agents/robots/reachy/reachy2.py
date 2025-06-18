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

REACHY_ARM_COLLISION_BIT = 28
"""Collision bit of the reachy hand links"""
REACHY_TORSO_COLLISION_BIT = 29
"""Collision bit of the reachy torso"""
REACHY_WHEELS_COLLISION_BIT = 30
"""Collision bit of the reachy robot wheel links"""
REACHY_BASE_COLLISION_BIT = 31
"""Collision bit of the reachy base"""


@register_agent()
class Reachy2(BaseAgent):
    uid = "reachy2"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/reachy/reachy2_mod.urdf"
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
            qpos=([
                0, 0, 0,                         # Base (3 - x, y, theta)
                0,                               # Tripod (1)
                -0.191, -0.191, 0.234, 0.234,    # Tripod bars (4 - dummies)
                0, 0, 0,                         # Neck (3 - roll, pitch, yaw)
                0, 0, 0, 0,                      # Shoulders (4 - r_pitch, l_pitch, r_roll, l_roll)
                0, 0,                            # Antennas (2 - left, right)
                0, 0, 0, 0,                      # Elbows (4 - r_yaw, l_yaw, r_pitch, l_pitch)
                0, 0, 0.008, 0.008, 0, 0,        # Wrists (6 - r_roll, l_roll, r_pitch, l_pitch, r_yaw, l_yaw)
                0.3,                             # R Gripper (1)
                0.528, 0.528,                    # R Proximal (2 - dummies)
                0.3,                             # L Gripper (1)                                     
                0.528, 0.528,                    # L Proximal (2 - dummies)
                -0.51, -0.51,                    # R Distal (2 - dummies)
                -0.51, -0.51]),                  # L Distal (2 - dummies)
        ),
        right_hand_out=Keyframe(
            pose=sapien.Pose(),
            qpos=([
                0, 0, 0,                         # Base (3 - x, y, theta)
                0,                               # Tripod (1)
                -0.191, -0.191, 0.234, 0.234,    # Tripod bars (4 - dummies)
                0, 0, 0,                         # Neck (3 - roll, pitch, yaw)
                -1.6, 0, 0, 0,                   # ----- Shoulders (4 - r_pitch, l_pitch, r_roll, l_roll)
                0, 0,                            # Antennas (2 - left, right)
                0, 0, 0, 0,                      # Elbows (4 - r_yaw, l_yaw, r_pitch, l_pitch)
                0, 0, 0.008, 0.008, 0, 0,        # Wrists (6 - r_roll, l_roll, r_pitch, l_pitch, r_yaw, l_yaw)
                0.3,                             # R Gripper (1)
                0.528, 0.528,                    # R Proximal (2 - dummies)
                0.3,                             # L Gripper (1)                                     
                0.528, 0.528,                    # L Proximal (2 - dummies)
                -0.51, -0.51,                    # R Distal (2 - dummies)
                -0.51, -0.51]),                  # L Distal (2 - dummies)
        ),
        right_hand_out_reversed=Keyframe(
            pose=sapien.Pose(),
            qpos=([
                0, 0, np.pi,                     # -----Base (3 - x, y, theta)
                0,                               # Tripod (1)
                -0.191, -0.191, 0.234, 0.234,    # Tripod bars (4 - dummies)
                0, 0, 0,                         # Neck (3 - roll, pitch, yaw)
                -1.6, 0, 0, 0,                   # ----- Shoulders (4 - r_pitch, l_pitch, r_roll, l_roll)
                0, 0,                            # Antennas (2 - left, right)
                0, 0, 0, 0,                      # Elbows (4 - r_yaw, l_yaw, r_pitch, l_pitch)
                0, 0, 0.008, 0.008, 0, 0,        # Wrists (6 - r_roll, l_roll, r_pitch, l_pitch, r_yaw, l_yaw)
                0.3,                             # R Gripper (1)
                0.528, 0.528,                    # R Proximal (2 - dummies)
                0.3,                             # L Gripper (1)                                     
                0.528, 0.528,                    # L Proximal (2 - dummies)
                -0.51, -0.51,                    # R Distal (2 - dummies)
                -0.51, -0.51]),                  # L Distal (2 - dummies)
        )
    )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="reachy_left_camera",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid=self.l_camera_link_name,
            ),
            CameraConfig(
                uid="reachy_right_camera",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid=self.r_camera_link_name,
            ),
            CameraConfig(
                uid="reachy_tof_camera",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid=self.tof_camera_link_name,
            ),
        ]
    disable_self_collisions = False

    def __init__(self, *args, **kwargs):
        self.r_arm_joint_names = [
            "r_shoulder_pitch",
            "r_shoulder_roll",
            "r_elbow_yaw",
            "r_elbow_pitch",
            "r_wrist_roll",
            "r_wrist_pitch",
            "r_wrist_yaw"
        ]
        self.l_arm_joint_names = [
            "l_shoulder_pitch",
            "l_shoulder_roll",
            "l_elbow_yaw",
            "l_elbow_pitch",
            "l_wrist_roll",
            "l_wrist_pitch",
            "l_wrist_yaw"

        ]
        self.all_arm_joint_names = self.r_arm_joint_names + self.l_arm_joint_names

        self.r_hand_active_joint_names = ["r_hand_finger"]
        # self.r_hand_joint_names = ["r_hand_finger_proximal", "r_hand_finger_distal"]
        self.r_hand_mimic_joint_names = [
            "r_hand_finger_proximal",
            "r_hand_finger_distal",
            "r_hand_finger_proximal_mimic",
            "r_hand_finger_distal_mimic"
        ]

        self.l_hand_active_joint_names = ["l_hand_finger"]
        # self.l_hand_joint_names = ["l_hand_finger_proximal", "l_hand_finger_distal"]
        self.l_hand_mimic_joint_names = [
            "l_hand_finger_proximal",
            "l_hand_finger_distal",
            "l_hand_finger_proximal_mimic",
            "l_hand_finger_distal_mimic"
        ]

        self.all_hand_active_joint_names = self.l_hand_active_joint_names + self.r_hand_active_joint_names
        self.all_hand_passive_joint_names = self.l_hand_mimic_joint_names + self.r_hand_mimic_joint_names

        self.head_joint_names = [
            "neck_roll",
            "neck_pitch",
            "neck_yaw",
            "antenna_left",
            "antenna_right"
        ]

        self.tripod_joint_names = ["tripod_joint"]
        self.tripod_mimic_joint_names = [
            "left_bar_joint_mimic",
            "left_bar_prism_joint_mimic",
            "right_bar_joint_mimic",
            "right_bar_prism_joint_mimic"
        ]

        self.base_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint",
        ]

        self.l_ee_link_name = "l_arm_tip"
        self.r_ee_link_name = "r_arm_tip"
        self.l_camera_link_name = "left_camera"
        self.r_camera_link_name = "right_camera"
        self.tof_camera_link_name = "tof_camera"

        self.arm_stiffness = 1e3
        self.arm_damping = 1e2 # 0.1
        self.arm_force_limit = 100

        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2 # 0.1
        self.gripper_force_limit = 100
        self.gripper_friction = 1

        self.head_stiffness = 1e3
        self.head_damping = 1e2 # 0.1
        self.head_force_limit = 100

        self.tripod_stiffness = 1e4
        self.tripod_damping = 1e3 # 0.1
        self.tripod_force_limit = 100

        self.base_stiffness = 1e3
        self.base_damping = 1e2 # 0.5
        self.base_force_limit = 100

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

        gripper_l_passive_joints_controller = PassiveControllerConfig(
            self.l_hand_mimic_joint_names,
            damping=0.01,
            friction=0.1,
            force_limit=100,
        )

        gripper_l_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.l_hand_active_joint_names,
            lower=0.3,
            upper=0.75,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=False,
        )

        gripper_l_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            self.l_hand_active_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=True,
            use_delta=True,
        )

        gripper_r_passive_joints_controller = PassiveControllerConfig(
            self.r_hand_mimic_joint_names,
            damping=0.01,
            friction=0.1,
            force_limit=100,
        )

        gripper_r_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.r_hand_active_joint_names,
            lower=0.3,
            upper=0.75,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=False,
        )

        gripper_r_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            self.r_hand_active_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=True,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Head
        # -------------------------------------------------------------------------- #
        head_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.head_joint_names,
            -0.1,
            0.1,
            self.head_stiffness,
            self.head_damping,
            self.head_force_limit,
            use_delta=True,
        )

        # useful to keep body unmoving from passed position
        stiff_head_pd_joint_pos = PDJointPosControllerConfig(
            self.head_joint_names,
            None,
            None,
            1e5,
            1e5,
            1e5,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Tripod
        # -------------------------------------------------------------------------- #
        tripod_passive_joints_controller = PassiveControllerConfig(
            self.tripod_mimic_joint_names,
            damping=0,
            friction=0,
        )

        tripod_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.tripod_joint_names,
            lower=0,
            upper=0.21,
            stiffness=self.tripod_stiffness,
            damping=self.tripod_damping,
            force_limit=self.tripod_force_limit,
            normalize_action=False,
        )

        tripod_mimic_joint_delta_pos = PDJointPosMimicControllerConfig(
            self.tripod_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.tripod_stiffness,
            damping=self.tripod_damping,
            force_limit=self.tripod_force_limit,
            normalize_action=True,
            use_delta=True,
        )

        stiff_tripod_pd_joint_pos = PDJointPosControllerConfig(
            self.tripod_joint_names,
            None,
            None,
            1e5,
            1e5,
            1e5,
        )

        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        base_pd_joint_vel = PDBaseForwardVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -3.14],
            upper=[1, 3.14],
            damping=1000,
            force_limit=500,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arms=arms_pd_joint_delta_pos,
                gripper_r_active=gripper_r_mimic_pd_joint_delta_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_delta_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,
                head=head_pd_joint_delta_pos,
                tripod_active=tripod_mimic_joint_delta_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel
            ),
            pd_joint_pos=dict(
                arms=arms_pd_joint_pos,
                gripper_r_active=gripper_r_mimic_pd_joint_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,                
                head=head_pd_joint_delta_pos,
                tripod_active=tripod_mimic_pd_joint_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel
            ),
            pd_ee_delta_pos=dict(
                l_arm=l_arm_pd_ee_delta_pos,
                r_arm=r_arm_pd_ee_delta_pos,
                gripper_r_active=gripper_r_mimic_pd_joint_delta_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_delta_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,                
                head=head_pd_joint_delta_pos,
                tripod_active=tripod_mimic_pd_joint_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pose=dict(
                l_arm=l_arm_pd_ee_delta_pose,
                r_arm=r_arm_pd_ee_delta_pose,
                gripper_r_active=gripper_r_mimic_pd_joint_delta_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_delta_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,                
                head=head_pd_joint_delta_pos,
                tripod_active=tripod_mimic_pd_joint_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pose_align=dict(
                l_arm=l_arm_pd_ee_delta_pose_align,
                r_arm=r_arm_pd_ee_delta_pose_align,
                gripper_r_active=gripper_r_mimic_pd_joint_delta_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_delta_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,
                head=head_pd_joint_delta_pos,
                tripod_active=tripod_mimic_pd_joint_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel,
            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arms=arms_pd_joint_target_delta_pos,
                gripper_r_active=gripper_r_mimic_pd_joint_delta_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_delta_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,
                head=head_pd_joint_delta_pos,
                tripod_active=tripod_mimic_pd_joint_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pos=dict(
                l_arm=l_arm_pd_ee_target_delta_pos,
                r_arm=r_arm_pd_ee_target_delta_pos,
                gripper_r_active=gripper_r_mimic_pd_joint_delta_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_delta_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,
                head=head_pd_joint_delta_pos,
                tripod_active=tripod_mimic_pd_joint_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pose=dict(
                l_arm=l_arm_pd_ee_target_delta_pose,
                r_arm=r_arm_pd_ee_target_delta_pose,
                gripper_r_active=gripper_r_mimic_pd_joint_delta_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_delta_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,
                head=head_pd_joint_delta_pos,
                tripod_active=tripod_mimic_pd_joint_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel,
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arms=arms_pd_joint_vel,
                gripper_r_active=gripper_r_mimic_pd_joint_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,
                head=head_pd_joint_delta_pos,
                tripod_active=tripod_mimic_pd_joint_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos_vel=dict(
                arms=arms_pd_joint_pos_vel,
                gripper_r_active=gripper_r_mimic_pd_joint_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,  
                head=head_pd_joint_delta_pos,
                tripod_active=tripod_mimic_pd_joint_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_vel=dict(
                arms=arms_pd_joint_delta_pos_vel,
                gripper_r_active=gripper_r_mimic_pd_joint_delta_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_delta_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,
                head=head_pd_joint_delta_pos,
                tripod_active=tripod_mimic_joint_delta_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_stiff_body=dict(
                arms=arms_pd_joint_delta_pos,
                gripper_r_active=gripper_r_mimic_pd_joint_delta_pos,
                gripper_r_passive=gripper_r_passive_joints_controller,
                gripper_l_active=gripper_l_mimic_pd_joint_delta_pos,
                gripper_l_passive=gripper_l_passive_joints_controller,
                head=stiff_head_pd_joint_pos,
                tripod_active=stiff_tripod_pd_joint_pos,
                tripod_passive=tripod_passive_joints_controller,
                base=base_pd_joint_vel
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        # TCPs
        self.l_tcp_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.l_ee_link_name
        )
        self.r_tcp_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.r_ee_link_name
        )
        self.tcp = self.r_tcp_link

        # Base
        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "base_link"
        )
        self.base_link.set_collision_group_bit(
            group=2, bit_idx=REACHY_BASE_COLLISION_BIT, bit=1
        )

        # Tripod Bars
        # Kinematic chains are: 
        #     base_link -> back_bar_base -> back_bar_inner -> torso
        #     torso -> left_bar_inner -> left_bar_base
        #     torso -> right_bar_inner -> right_bar_base
        #     base_link -> left_bar_anchor
        #     base_link -> right_bar_anchor
        self.back_bar_base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "back_bar_base"
        )
        self.back_bar_inner_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "back_bar_inner"
        )

        self.left_bar_base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_bar_base"
        )
        self.left_bar_inner_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_bar_inner"
        )
        self.left_bar_anchor_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_bar_anchor"
        )

        self.right_bar_base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_bar_base"
        )
        self.right_bar_inner_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_bar_inner"
        )
        self.right_bar_anchor_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_bar_anchor"
        )
        for link in [self.left_bar_base_link, self.right_bar_base_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=REACHY_BASE_COLLISION_BIT, bit=1
            )

        # Torso and arms
        # Kinematic chains are: 
        #     torso -> ... -> l_elbow_arm_link -> ... -> l_elbow_forearm_link -> ... -> l_hand_palm_link
        #     torso -> ... -> r_elbow_arm_link -> ... -> r_elbow_forearm_link -> ... -> r_hand_palm_link
        self.torso_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "torso"
        )
        self.l_elbow_arm_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_elbow_arm_link"
        )
        self.l_elbow_forearm_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_elbow_forearm_link"
        )
        self.r_elbow_arm_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_elbow_arm_link"
        )
        self.r_elbow_forearm_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_elbow_forearm_link"
        )
        for link in [self.torso_link, self.l_elbow_arm_link, self.r_elbow_arm_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=REACHY_TORSO_COLLISION_BIT, bit=1
            )
        for link in [self.l_elbow_forearm_link, self.l_elbow_arm_link,
                     self.r_elbow_forearm_link, self.r_elbow_arm_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=REACHY_ARM_COLLISION_BIT, bit=1
            )

        # Hands
        # Kinematic chains are: 
        #     l_hand_palm_link -> l_hand_proximal_link -> l_hand_distal_link
        #     l_hand_palm_link -> l_hand_proximal_mimic_link -> l_hand_distal_mimic_link
        #     r_hand_palm_link -> r_hand_proximal_link -> r_hand_distal_link
        #     r_hand_palm_link -> r_hand_proximal_mimic_link -> r_hand_distal_mimic_link
        self.l_hand_palm_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_hand_palm_link"
        )
        self.l_hand_proximal_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_hand_proximal_link"
        )
        self.l_hand_proximal_mimic_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_hand_proximal_mimic_link"
        )
        self.l_hand_distal_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_hand_distal_link"
        )
        self.l_hand_distal_mimic_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_hand_distal_mimic_link"
        )

        self.r_hand_palm_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_hand_palm_link"
        )
        self.r_hand_proximal_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_hand_proximal_link"
        )
        self.r_hand_proximal_mimic_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_hand_proximal_mimic_link"
        )
        self.r_hand_distal_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_hand_distal_link"
        )
        self.r_hand_distal_mimic_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_hand_distal_mimic_link"
        )
        for link in [self.l_elbow_forearm_link, self.l_hand_palm_link,
                     self.r_elbow_forearm_link, self.r_hand_palm_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=REACHY_ARM_COLLISION_BIT, bit=1
            )

        # Wheels
        self.wheel1_link: Link = self.robot.links_map["drivewhl1_link"]
        self.wheel2_link: Link = self.robot.links_map["drivewhl2_link"]
        self.wheel3_link: Link = self.robot.links_map["drivewhl3_link"]
        for link in [self.wheel1_link, self.wheel2_link, self.wheel3_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=REACHY_WHEELS_COLLISION_BIT, bit=1
            )

        self.left_camera_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.l_camera_link_name
        )
        self.right_camera_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.r_camera_link_name
        )
        self.tof_camera_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.tof_camera_link_name
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
            self.l_hand_distal_link, object
        )
        l_thumb_contact_forces = self.scene.get_pairwise_contact_forces(
            self.l_hand_distal_mimic_link, object
        )
        l_finger_force = torch.linalg.norm(l_finger_contact_forces, axis=1)
        l_thumb_force = torch.linalg.norm(l_thumb_contact_forces, axis=1)

        # direction to open the gripper
        lfdirection = -self.l_hand_distal_link.pose.to_transformation_matrix()[..., :3, 1]
        ltdirection = self.l_hand_distal_mimic_link.pose.to_transformation_matrix()[..., :3, 1]
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
            self.r_hand_distal_link, object
        )
        r_thumb_contact_forces = self.scene.get_pairwise_contact_forces(
            self.r_hand_distal_mimic_link, object
        )
        r_finger_force = torch.linalg.norm(r_finger_contact_forces, axis=1)
        r_thumb_force = torch.linalg.norm(r_thumb_contact_forces, axis=1)

        rfdirection = -self.r_hand_distal_link.pose.to_transformation_matrix()[..., :3, 1]
        rtdirection = self.r_hand_distal_mimic_link.pose.to_transformation_matrix()[..., :3, 1]
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

    def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
        body_qvel = self.robot.get_qvel()[..., 3:-2]
        base_qvel = self.robot.get_qvel()[..., :3]
        return torch.all(body_qvel <= threshold, dim=1) & torch.all(
            base_qvel <= base_threshold, dim=1
        )

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
