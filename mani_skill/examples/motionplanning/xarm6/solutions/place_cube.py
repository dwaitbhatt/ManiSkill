import numpy as np
import sapien

from mani_skill.envs.tasks import PlaceCubeEnv
from mani_skill.examples.motionplanning.xarm6.motionplanner import \
    XArm6RobotiqMotionPlanningSolver, XArm6PandaGripperMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: PlaceCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)

    if env.unwrapped.robot_uids == "xarm6_robotiq":
        planner_cls = XArm6RobotiqMotionPlanningSolver
    elif env.unwrapped.robot_uids == "xarm6_pandagripper":
        planner_cls = XArm6PandaGripperMotionPlanningSolver
    else:
        raise ValueError(f"Unsupported robot uid: {env.robot_uid}")

    planner = planner_cls(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.obj)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.obj.pose.sp.p)

    # Using RRT* for harder stages of the task
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_RRTStar(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # # -------------------------------------------------------------------------- #
    # # Move above bin
    # # -------------------------------------------------------------------------- #
    position_above_bin = env.bin.pose.sp.p + np.array([0, 0, env.short_side_half_size + 4 * env.cube_half_length + 4 * env.short_side_half_size])
    above_bin_pose = sapien.Pose(position_above_bin, grasp_pose.q)
    planner.move_to_pose_with_RRTStar(above_bin_pose)

    # # -------------------------------------------------------------------------- #
    # # Move down
    # # # -------------------------------------------------------------------------- #
    position_inside_bin = env.bin.pose.sp.p + np.array([0, 0, env.short_side_half_size + env.cube_half_length + 0.01])
    inside_bin_pose = sapien.Pose(position_inside_bin, grasp_pose.q)
    planner.move_to_pose_with_screw(inside_bin_pose)

    # # -------------------------------------------------------------------------- #
    # # Open gripper
    # # -------------------------------------------------------------------------- #
    res = planner.open_gripper()

    planner.close()
    return res

