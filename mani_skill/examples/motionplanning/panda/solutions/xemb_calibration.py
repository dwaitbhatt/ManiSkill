import numpy as np
import sapien

from mani_skill.envs.tasks import XembCalibrationEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.structs import Pose

def solve(env: XembCalibrationEnv, seed=None, debug=False, vis=False, eps=0.0):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    original_p = env.agent.tcp.pose.p.numpy()
    original_q = env.agent.tcp.pose.q.numpy()

    # -------------------------------------------------------------------------- #
    # Move in each axis and go back
    # -------------------------------------------------------------------------- #
    for axis, delta in env.calibration_deltas.items():
        delta_pos = np.zeros(3, dtype=np.float32)
        if axis[1] == 'x':
            j = 0
        elif axis[1] == 'y':
            j = 1
        elif axis[1] == 'z':
            j = 2

        delta_pos[j] = delta

        # Move to goal position
        reach_pose = Pose.create_from_pq(p=original_p + delta_pos, q=original_q)
        planner.move_to_pose_with_screw(reach_pose, eps=eps)
        
        # Return to middle position
        reach_pose = Pose.create_from_pq(p=original_p, q=original_q)
        planner.move_to_pose_with_screw(reach_pose, eps=eps)

    # -------------------------------------------------------------------------- #
    # Close gripper 
    # -------------------------------------------------------------------------- #
    res = planner.close_gripper()

    return res
