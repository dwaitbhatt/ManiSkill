CONTROL_MODE="pd_joint_vel"

for XARM_UID in xarm6_robotiq xarm6_robotiq_custom ; do
    for env_id in PickCubeCustom-v1 PlaceCube-v1 StackCubeCustom-v1 ; do
        mkdir -p demos/$env_id/motionplanning/$XARM_UID

        # Generate sample motion planning demo video
        python -m mani_skill.examples.motionplanning.xarm6.run --env-id $env_id --robot-uid $XARM_UID \
            --traj-name="trajectory" --only-count-success --save-video -n 1 \
            --shader="rt" # generate sample video
        mv demos/$env_id/motionplanning/0.mp4 demos/$env_id/motionplanning/$XARM_UID/sample.mp4

        # Generate motion planning demo trajectories
        python -m mani_skill.examples.motionplanning.xarm6.run --robot-uid $XARM_UID --env-id $env_id --traj-name="trajectory" -n 1000 --only-count-success --num-procs 10
        mv demos/$env_id/motionplanning/trajectory.{h5,json} demos/$env_id/motionplanning/$XARM_UID/

        # Replay to desired observation and control mode
        python -m mani_skill.trajectory.replay_trajectory \
            --traj-path demos/$env_id/motionplanning/$XARM_UID/trajectory.h5 \
            --use-first-env-state -c $CONTROL_MODE -o state \
            --save-traj --num-envs 10 -b cpu

        # Train ACT policy
        python train.py --env-id $env_id --robot_uid $XARM_UID \
            --demo-path demos/$env_id/motionplanning/$XARM_UID/trajectory.state.pd_joint_vel.physx_cpu.h5 \
            --control-mode $CONTROL_MODE --sim-backend "gpu" --num_demos 1000 --max_episode_steps 100 --num_eval_envs 30 \
            --total_iters 200000 --log_freq 100 --eval_freq 5000 --wandb_video_freq 1 \
            --exp-name=act-PickCube-v1-xarm6_robotiq-state-1000_motionplanning_demos-1 \
            --demo_type motionplanning --track
    done
done

# # RGB based
# python train_rgbd.py --env-id PickCube-v1 --no_include_depth \
#   --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
#   --control-mode "pd_joint_vel" --sim-backend "gpu" --num_demos 1000 --max_episode_steps 100 --num_eval_envs 30 \
#   --total_iters 200000 --log_freq 100 --eval_freq 5000 --wandb_video_freq 1 \
#   --exp-name=act-PickCube-v1-xarm6_robotiq-rgb-1000_motionplanning_demos-1 \
#   --demo_type motionplanning --track