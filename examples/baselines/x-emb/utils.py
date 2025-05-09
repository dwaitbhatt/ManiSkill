import wandb
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from typing import Optional

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    algorithm: str = "SAC_LATENT"
    """the actor-critic algorithm to use for the experiment (TD3 or SAC)"""
    source_robot: str = "panda"
    """which robot to learnt the latent policy with"""
    target_robot: str = "xarm6_robotiq"
    """which robot to transfer the policy to"""
    control_mode: str = "pd_joint_pos"
    """which control mode to use for the experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "maniskill_experiments"
    """the wandb's project name"""
    wandb_entity: str = "ucsd_erl"
    """the entity (team) of wandb's project"""
    wandb_group: str = "xemb_transfer_learning"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    wandb_video_freq: int = 0
    """frequency to upload saved videos to wandb (every nth saved video will be uploaded)"""
    save_model: bool = True
    """whether to save the model checkpoints"""
    save_model_dir: Optional[str] = "runs"
    """the directory to save the model"""

    # Environment specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 5000
    """evaluation frequency in terms of timesteps"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of timesteps"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""

    # TD3 specific parameters
    noise_clip: float = 0.5
    """clip parameter of the target policy noise"""
    target_policy_noise: float = 0.1
    """noise added to target policy during critic update"""
    exploration_noise: float = 0.1
    """standard deviation of exploration noise"""

    # SAC specific parameters
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""

    # Neural network hyperparameters
    mlp_dim: int = 256
    """the hidden dimension of the networks"""
    num_layers: int = 3
    """the number of hidden layers in the networks"""
    simnorm_dim: int = 8
    """the dimension of the simplicial normalization"""
    enc_dim: int = 256
    """the hidden dimension of the encoder networks"""
    enc_lr_scale: float = 1
    """the learning rate scale of the encoder networks"""
    num_bins: int = 101
    """the number of bins for the Q/reward predictor"""
    vmin: float = -10
    """the minimum value of the Q/reward predictor"""
    vmax: float = 10
    """the maximum value of the Q/reward predictor"""

    # Transfer learning specific parameters
    latent_robot_obs_dim: int = 128
    """the dimension of the latent observation"""
    latent_env_obs_dim: int = 128
    """the dimension of the latent observation"""
    latent_action_dim: int = 128
    """the dimension of the latent action"""
    latent_dynamics_freq: int = 1000
    """the frequency of updating the latent dynamics"""

    # Ablation specific parameters
    disable_latent_dynamics: bool = False
    """whether to disable the latent dynamics"""
    disable_rew_predictor: bool = False
    """whether to disable the reward predictor"""
    disable_act_encoder: bool = False
    """whether to disable the action encoder"""
    disable_act_decoder: bool = False
    """whether to disable the action decoder"""
    disable_obs_encoders: bool = False
    """whether to disable the observation encoders"""

    # Alignment specific parameters
    source_trained_model_path: Optional[str] = "examples/baselines/x-emb/source_panda_final_ckpt.pt"
    """the path to the source trained model"""
    alignment_source_traj_path: Optional[str] = "demos/XembCalibration-v1/motionplanning/panda_calibration_traj_n10.state_dict.pd_joint_vel.physx_cpu.h5"
    """the path to the alignment source trajectories"""
    alignment_target_traj_path: Optional[str] = "demos/XembCalibration-v1/motionplanning/xarm6_calibration_traj_n10.state_dict.pd_joint_vel.physx_cpu.h5"
    """the path to the alignment target trajectories"""
    alignment_samples: int = 100_000
    """the number of samples to run alignment on"""
    alignment_batch_size: int = 128
    """the batch size for alignment"""
    adapted_target_nets: bool = True
    """whether to use adapters for target encoder networks"""
    adapter_layers: int = 1
    """the number of adapter layers at the start of the target encoder networks"""
    only_train_adapters: bool = False
    """whether to only train the adapter layers"""
    use_latent_adversary: bool = True
    """whether to use a latent adversary"""
    lambda_latent_gp: float = 10.0
    """the lambda for the latent gradient penalty"""
    lambda_latent_dynamics_loss: float = 10.0
    """the lambda for the latent dynamics loss"""

    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 0.5
    """update to data ratio"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""
    latent_obs_dim: int = 0
    """the dimension of the latent observation (including robot and env obs)"""
    bin_size: float = 0
    """the size of the bins for the Q/reward predictor"""
