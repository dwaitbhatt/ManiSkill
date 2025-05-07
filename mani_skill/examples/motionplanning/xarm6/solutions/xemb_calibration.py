import numpy as np
import sapien

from mani_skill.envs.tasks import XembCalibrationEnv
from mani_skill.examples.motionplanning.xarm6.motionplanner import \
    XArm6RobotiqMotionPlanningSolver, XArm6PandaGripperMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.structs import Pose

def solve(env: XembCalibrationEnv, seed=None, debug=False, vis=False):
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
    env = env.unwrapped

    delta = 0.1
    original_p = env.agent.tcp.pose.p.numpy()
    original_q = env.agent.tcp.pose.q.numpy()

    # -------------------------------------------------------------------------- #
    # Move in each axis and go back
    # -------------------------------------------------------------------------- #
    axes = ['x', 'y', 'z']
    for i, axis in enumerate(axes):
        # Create displacement vector for current axis
        displacement = np.zeros(3, dtype=np.float32)
        displacement[i] = delta
        
        # Move forward
        reach_pose = Pose.create_from_pq(p=original_p + displacement, q=original_q)
        planner.move_to_pose_with_screw(reach_pose)
        
        # Move backward (twice the distance)
        reach_pose = Pose.create_from_pq(p=original_p - displacement, q=original_q)
        planner.move_to_pose_with_screw(reach_pose)
        
        # Return to middle position
        reach_pose = Pose.create_from_pq(p=original_p, q=original_q)
        planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Close gripper 
    # -------------------------------------------------------------------------- #
    res = planner.close_gripper()

    return res
