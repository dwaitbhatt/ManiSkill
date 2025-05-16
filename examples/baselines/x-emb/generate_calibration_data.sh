SAVE_PATH=demos/XembCalibration-v1/motionplanning

################################ Sample videos for each robot ################################
printf "\nGenerating 1 normal trajectory video for Panda, and saving traj data...\n"
python -m mani_skill.examples.motionplanning.panda.run --env-id XembCalibration-v1 --eps 0 --traj-name="panda_calibration_traj" --only-count-success --save-video -n 1 --shader="default" > /dev/null
mv $SAVE_PATH/0.mp4 $SAVE_PATH/panda_calibration_traj.mp4

printf "\nGenerating 1 noisy trajectory video for Panda...\n"
python -m mani_skill.examples.motionplanning.panda.run --env-id XembCalibration-v1 --eps 3e-2 --traj-name="panda_calibration_noisy_traj" --only-count-success --save-video -n 1 --shader="default" > /dev/null
mv $SAVE_PATH/0.mp4 $SAVE_PATH/panda_calibration_noisy_traj.mp4
rm $SAVE_PATH/panda_calibration_noisy_traj.h5
rm $SAVE_PATH/panda_calibration_noisy_traj.json

printf "\nGenerating 1 normal trajectory video for XArm6, and saving traj data...\n"
python -m mani_skill.examples.motionplanning.xarm6.run --env-id XembCalibration-v1 --eps 0 --traj-name="xarm6_calibration_traj" --only-count-success --save-video -n 1 --shader="default" > /dev/null
mv $SAVE_PATH/0.mp4 $SAVE_PATH/xarm6_calibration_traj.mp4

printf "\nGenerating 1 noisy trajectory video for XArm6...\n"
python -m mani_skill.examples.motionplanning.xarm6.run --env-id XembCalibration-v1 --eps 3e-2 --traj-name="xarm6_calibration_noisy_traj" --only-count-success --save-video -n 1 --shader="default" > /dev/null
mv $SAVE_PATH/0.mp4 $SAVE_PATH/xarm6_calibration_noisy_traj.mp4
rm $SAVE_PATH/xarm6_calibration_noisy_traj.h5
rm $SAVE_PATH/xarm6_calibration_noisy_traj.json

################################ Generate 10 noisy trajectories of calibration data for each robot ################################
printf "\nGenerating 10 noisy trajectories of calibration data for Panda...\n"
python -m mani_skill.examples.motionplanning.panda.run --env-id XembCalibration-v1 --eps 3e-2 --traj-name="panda_calibration_noisy_traj_n10" --only-count-success -n 10

printf "\nGenerating 10 noisy trajectories of calibration data for XArm6... (output suppressed)\n"
python -m mani_skill.examples.motionplanning.xarm6.run --env-id XembCalibration-v1 --eps 3e-2 --traj-name="xarm6_calibration_noisy_traj_n10" --only-count-success -n 10 > /dev/null

printf "\nFinished generating calibration data for both robots.\n"

################################ Merge 10 noisy with 1 normal trajectory for each robot ################################
printf "\nMerging 10 noisy with 1 normal trajectory for each robot...\n"

mkdir -p $SAVE_PATH/to_merge/panda_calibration_traj
mkdir -p $SAVE_PATH/to_merge/panda_calibration_noisy_traj_n10
mkdir -p $SAVE_PATH/to_merge/xarm6_calibration_traj
mkdir -p $SAVE_PATH/to_merge/xarm6_calibration_noisy_traj_n10
mkdir -p $SAVE_PATH/merged

# Copy trajectories to folders as required by merge_trajectory.py
mv $SAVE_PATH/panda_calibration_traj.h5 $SAVE_PATH/to_merge/panda_calibration_traj/trajectory.h5
mv $SAVE_PATH/panda_calibration_traj.json $SAVE_PATH/to_merge/panda_calibration_traj/trajectory.json
mv $SAVE_PATH/panda_calibration_noisy_traj_n10.h5 $SAVE_PATH/to_merge/panda_calibration_noisy_traj_n10/trajectory.h5
mv $SAVE_PATH/panda_calibration_noisy_traj_n10.json $SAVE_PATH/to_merge/panda_calibration_noisy_traj_n10/trajectory.json

mv $SAVE_PATH/xarm6_calibration_traj.h5 $SAVE_PATH/to_merge/xarm6_calibration_traj/trajectory.h5
mv $SAVE_PATH/xarm6_calibration_traj.json $SAVE_PATH/to_merge/xarm6_calibration_traj/trajectory.json
mv $SAVE_PATH/xarm6_calibration_noisy_traj_n10.h5 $SAVE_PATH/to_merge/xarm6_calibration_noisy_traj_n10/trajectory.h5
mv $SAVE_PATH/xarm6_calibration_noisy_traj_n10.json $SAVE_PATH/to_merge/xarm6_calibration_noisy_traj_n10/trajectory.json

# Merge trajectories
python -m mani_skill.trajectory.merge_trajectory --input-dirs $SAVE_PATH/to_merge/panda_calibration_traj $SAVE_PATH/to_merge/panda_calibration_noisy_traj_n10 --output-path $SAVE_PATH/merged/panda_calibration_noisy_hybrid_traj_n10.h5
python -m mani_skill.trajectory.merge_trajectory --input-dirs $SAVE_PATH/to_merge/xarm6_calibration_traj $SAVE_PATH/to_merge/xarm6_calibration_noisy_traj_n10 --output-path $SAVE_PATH/merged/xarm6_calibration_noisy_hybrid_traj_n10.h5


################################ Replay trajectories for each robot ################################
printf "\nConverting merged trajectories to have obs_mode=state_dict and control_mode=pd_joint_vel...\n"

python -m mani_skill.trajectory.replay_trajectory --traj-path $SAVE_PATH/merged/panda_calibration_noisy_hybrid_traj_n10.h5 --use-first-env-state -c pd_joint_vel -o state_dict --save-traj --num-envs 10 -b cpu

python -m mani_skill.trajectory.replay_trajectory --traj-path $SAVE_PATH/merged/xarm6_calibration_noisy_hybrid_traj_n10.h5 --use-first-env-state -c pd_joint_vel -o state_dict --save-traj --num-envs 10 -b cpu