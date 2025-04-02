# Network hyperparam test
python nautilus_configs/launch_nautilus_pods.py --algo SAC_LATENT --total_timesteps 1_500_000 --num_pods 3 --jobs --exp_suffix ld32 --extra "--latent_robot_obs_dim 32 --latent_env_obs_dim 32 --latent_action_dim 32"

# Sanity/implementation check - basic SAC
python nautilus_configs/launch_nautilus_pods.py --algo SAC_LATENT --total_timesteps 500_000 --num_pods 1 --exp_suffix disableall_justsac --extra "--disable_latent_dynamics --disable_rew_predictor --disable_act_encoder --disable_act_decoder --disable_obs_encoders"

# Without latent dynamics and reward predictor
python nautilus_configs/launch_nautilus_pods.py --algo SAC_LATENT --total_timesteps 2_000_000 --jobs --exp_suffix obsenc_actenc_actdec_nold_norew --extra "--disable_latent_dynamics disable_rew_predictor"

# Only latent dynamics
python nautilus_configs/launch_nautilus_pods.py --algo SAC_LATENT --total_timesteps 2_000_000 --jobs --exp_suffix obsenc_actenc_actdec_ld_norew --extra "--disable_rew_predictor"

# Only reward predictor
python nautilus_configs/launch_nautilus_pods.py --algo SAC_LATENT --total_timesteps 2_000_000 --jobs --exp_suffix obsenc_actenc_actdec_nold_rew --extra "--disable_latent_dynamics"

# Without action encoder
python nautilus_configs/launch_nautilus_pods.py --algo SAC_LATENT --total_timesteps 2_000_000 --jobs --exp_suffix obsenc_noactenc_actdec_ld_rew --extra "--disable_act_encoder"

# Without action decoder
python nautilus_configs/launch_nautilus_pods.py --algo SAC_LATENT --total_timesteps 2_000_000 --jobs --exp_suffix obsenc_actenc_noactdec_ld_rew --extra "--disable_act_decoder"

# Without observation encoders
python nautilus_configs/launch_nautilus_pods.py --algo SAC_LATENT --total_timesteps 2_000_000 --jobs --exp_suffix noobsenc_actenc_actdec_ld_rew --extra "--disable_obs_encoders"