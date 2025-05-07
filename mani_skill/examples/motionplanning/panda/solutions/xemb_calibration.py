import numpy as np
import sapien

from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.structs import Pose

def solve(env: PickCubeEnv, seed=None, debug=False, vis=False):
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
