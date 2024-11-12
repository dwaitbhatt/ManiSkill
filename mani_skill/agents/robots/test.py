# import mani_skill.envs
# import xarm6
# import gymnasium as gym

# env = gym.make("EmptyEnv-v1", robot_uids="xarm6")
# env.reset()
# env.step(env.action_space.sample())

import mani_skill.examples.demo_robot as demo_robot_script

demo_robot_script.main()
