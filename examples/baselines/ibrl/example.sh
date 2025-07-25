# test command 
python3 ibrl.py \
    --env-id="PickCube-v1" \
    --num-envs=32 \
    --utd=0.5 \
    --seed=1 \
    --buffer-size=500_000 \
    --total-timesteps=1_000_000 \
    --eval-freq=50_000 \
    --control-mode="pd_joint_vel" \
    --warmup-steps=5_000 \
    --wu_demo \
    --sample_mode="all" \
    --track \
    --wandb_group="test"

# td3 command
# pickcube
for seed in 1 2 3 4 5; do
python3 ibrl.py \
    --env-id="PickCube-v1" \
    --num-envs=32 \
    --utd=0.5 \
    --seed=$seed \
    --buffer-size=500_000 \
    --total-timesteps=1_000_000 \
    --eval-freq=50_000 \
    --control-mode="pd_joint_vel" \
    --track 
    # project name will be fixed, group need change by the algo using
done