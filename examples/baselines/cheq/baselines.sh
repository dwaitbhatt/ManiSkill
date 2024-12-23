# Baseline results for CHEQ

seeds=(7298 6173 1450)

### CHEQ Baselines ###
for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="PushCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-PushCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-PickCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="PushT-v1" --seed=${seed} \
    --num_envs=4096 --update_epochs=8 --num_minibatches=32 --gamma=0.99 \
    --total_timesteps=50_000_000 --num-steps=16 --num_eval_steps=100 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-PushT-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="StackCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-StackCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="RollBall-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 --gamma=0.95 \
    --total_timesteps=50_000_000 --num-eval-steps=80 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-RollBall-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="PullCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-PullCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="PokeCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-PokeCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="LiftPegUpright-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-LiftPegUpright-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="AnymalC-Reach-v1" --seed=${seed} \
    --num_envs=4096 --update_epochs=8 --num_minibatches=32 --gamma=0.99 --gae_lambda=0.95 \
    --total_timesteps=50_000_000 --num-steps=16 --num-eval-steps=200 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-AnymalC-Reach-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="PegInsertionSide-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 --gamma=0.97 --gae_lambda=0.95 \
    --total_timesteps=150_000_000 --num-steps=100 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-PegInsertionSide-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="TwoRobotPickCube-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num-steps=100 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-TwoRobotPickCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="UnitreeG1PlaceAppleInBowl-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num-steps=32 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-UnitreeG1PlaceAppleInBowl-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python cheq_train.py --env_id="UnitreeG1TransportBox-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=100_000_000 --num-steps=32 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="cheq-UnitreeG1TransportBox-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done


