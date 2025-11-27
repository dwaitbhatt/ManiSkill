# Generate all motion planning demos for the dataset
for robot in panda xarm6
do
  for env_id in PickCube-v1 PickCubeCustom-v1 # PickCube-v1 PushCube-v1 StackCube-v1 PlugCharger-v1
  do
      python -m mani_skill.examples.motionplanning.$robot.run --env-id $env_id \
        --traj-name="trajectory" --only-count-success --save-video -n 1 \
        --shader="rt" # generate sample videos
      mv demos/$env_id/motionplanning/0.mp4 demos/$env_id/motionplanning/sample.mp4
      python -m mani_skill.examples.motionplanning.$robot.run --env-id $env_id --traj-name="trajectory" -n 1000 --only-count-success --num-procs 10
  done
done