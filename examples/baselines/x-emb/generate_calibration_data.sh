# Sample videos for each robot:
printf "\nGenerating sample video for Panda..."
python -m mani_skill.examples.motionplanning.panda.run --env-id XembCalibration-v1 --eps 3e-2 --traj-name="panda_calibration_noisy_traj" --only-count-success --save-video -n 1 --shader="rt" > /dev/null
mv demos/XembCalibration-v1/motionplanning/0.mp4 demos/XembCalibration-v1/motionplanning/panda_calibration_noisy_traj.mp4

printf "\nGenerating sample video for XArm6..."
python -m mani_skill.examples.motionplanning.xarm6.run --env-id XembCalibration-v1 --eps 3e-2 --traj-name="xarm6_calibration_noisy_traj" --only-count-success --save-video -n 1 --shader="rt" > /dev/null
mv demos/XembCalibration-v1/motionplanning/0.mp4 demos/XembCalibration-v1/motionplanning/xarm6_calibration_noisy_traj.mp4

# Generate 10 trajectories of calibration data for each robot
printf "\nGenerating calibration data for Panda..."
python -m mani_skill.examples.motionplanning.panda.run --env-id XembCalibration-v1 --eps 3e-2 --traj-name="panda_calibration_noisy_traj_n10" --only-count-success -n 10

printf "\nGenerating calibration data for XArm6... (output suppressed)"
python -m mani_skill.examples.motionplanning.xarm6.run --env-id XembCalibration-v1 --eps 3e-2 --traj-name="xarm6_calibration_noisy_traj_n10" --only-count-success -n 10 > /dev/null

printf "\nFinished generating calibration data for both robots."

# Replay trajectories for each robot
python -m mani_skill.trajectory.replay_trajectory --traj-path demos/XembCalibration-v1/motionplanning/panda_calibration_noisy_traj_n10.h5 --use-first-env-state -c pd_joint_vel -o state_dict --save-traj --num-envs 10 -b cpu

python -m mani_skill.trajectory.replay_trajectory --traj-path demos/XembCalibration-v1/motionplanning/xarm6_calibration_noisy_traj_n10.h5 --use-first-env-state -c pd_joint_vel -o state_dict --save-traj --num-envs 10 -b cpu