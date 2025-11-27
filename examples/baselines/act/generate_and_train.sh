XARM_UID="xarm6_robotiq_custom"
CONTROL_MODE="pd_joint_vel"

for env_id in PickCube-v1 PickCubeCustom-v1 # PlaceCube-v1
do
  # Generate motion planning demos
  python -m mani_skill.examples.motionplanning.xarm6.run --env-id $env_id --robot-uid $XARM_UID \
    --traj-name="trajectory" --only-count-success --save-video -n 1 \
    --shader="rt" # generate sample videos
  mv demos/$env_id/motionplanning/0.mp4 demos/$env_id/motionplanning/sample.mp4
  python -m mani_skill.examples.motionplanning.xarm6.run --env-id $env_id --traj-name="trajectory" -n 1000 --only-count-success --num-procs 10
  mv demos/$env_id/motionplanning/trajectory.h5 demos/$env_id/motionplanning/$XARM_UID/trajectory.h5

  # Replay to desired observation and control mode
  python -m mani_skill.trajectory.replay_trajectory \
    --traj-path demos/$env_id/motionplanning/$XARM_UID/trajectory.h5 \
    --use-first-env-state -c $CONTROL_MODE -o state \
    --save-traj --num-envs 10 -b cpu

  # Train ACT policy
  python train.py --env-id $env_id --robot_uid $XARM_UID \
    --demo-path demos/$env_id/motionplanning/$XARM_UID/ \
    --control-mode $CONTROL_MODE --sim-backend "gpu" --num_demos 1000 --max_episode_steps 100 --num_eval_envs 30 \
    --total_iters 200000 --log_freq 100 --eval_freq 5000 --wandb_video_freq 1 \
    --exp-name=act-PickCube-v1-xarm6_robotiq-state-1000_motionplanning_demos-1 \
    --demo_type motionplanning --track
done

# # RGB based
# python train_rgbd.py --env-id PickCube-v1 --no_include_depth \
#   --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
#   --control-mode "pd_joint_vel" --sim-backend "gpu" --num_demos 1000 --max_episode_steps 100 --num_eval_envs 30 \
#   --total_iters 200000 --log_freq 100 --eval_freq 5000 --wandb_video_freq 1 \
#   --exp-name=act-PickCube-v1-xarm6_robotiq-rgb-1000_motionplanning_demos-1 \
#   --demo_type motionplanning --track