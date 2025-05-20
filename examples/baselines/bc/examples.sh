# Example scripts for training Behavior Cloning that have some results

# Learning from motion planning generated demonstrations

# PushCube-v1
# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
#   --use-first-env-state -c pd_ee_delta_pos -o state \
#   --save-traj --num-procs 10 -b cpu

# python bc.py --env-id "PushCube-v1" \
#   --demo-path /home/dwait/dwait_ws/ManiSkill/mani_skill/examples/motionplanning/xarm6/demos/PlaceCube-v1/motionplanning/trajectory.state.pd_joint_vel.physx_cpu.h5 \
#   --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max-episode-steps 100 \
#   --total-iters 10000

# # PickCube-v1
# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path /home/dwait/ManiSkill/mani_skill/examples/motionplanning/xarm6/demos/PickCube-v1/motionplanning/trajectory_simplest.h5 \
#   --use-first-env-state -c pd_joint_vel -o state \
#   --save-traj --num-procs 10 -b cpu

# python bc.py --env-id "PickCube-v1" \
#   --demo-path /home/dwait/ManiSkill/mani_skill/examples/motionplanning/xarm6/demos/PickCube-v1/motionplanning/trajectory_simplest.state.pd_joint_vel.cpu.h5 \
#   --control-mode "pd_joint_vel" --sim-backend "cpu" --max-episode-steps 100 \
#   --total-iters 10000


# # Learning from neural network / RL generated demonstrations

# # PickCube-v1
# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path ~/.maniskill/demos/PickCube-v1/rl/trajectory.h5 \
#   --use-first-env-state -c pd_joint_delta_pos -o state \
#   --save-traj --num-procs 10 -b cpu

# python bc.py --env-id "PickCube-v1" \
#   --demo-path ~/.maniskill/demos/PickCube-v1/rl/trajectory.state.pd_joint_delta_pos.cpu.h5 \
#   --control-mode "pd_joint_delta_pos" --sim-backend "cpu" --max-episode-steps 100 \
#   --total-iters 10000


# Full BC pipeline for PlaceCube-v1
python -m mani_skill.examples.motionplanning.xarm6.run --env-id PlaceCube-v1 --traj-name="trajectory" -n 100 --only-count-success

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/PickCube-v1/motionplanning/20250519_193425.h5 \
  --use-first-env-state -c pd_joint_vel -o rgbd \
  --save-traj --num-envs 10 -b cpu

python bc_rgbd.py --env-id "PlaceCube-v1" \
  --demo-path demos/PickCube-v1/motionplanning/20250519_193425.rgbd.pd_joint_vel.physx_cpu.h5 \
  --control-mode "pd_joint_vel" --sim-backend "cpu" --max-episode-steps 150 \
  --total-iters 1000000 --batch-size 128 --num_eval_envs 1 \
  --track --wandb_project_name "maniskill_experiments" --wandb_entity "ucsd_erl"
