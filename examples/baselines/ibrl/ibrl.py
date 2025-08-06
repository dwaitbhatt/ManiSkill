from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
import math
from typing import Optional

import tqdm
import h5py

from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from typing import Sequence
import tyro

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    # seed: int = 1
    # """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill-Rainbow"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: Optional[str] = None
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
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    robot_uids: str = "xarm6_robotiq"
    """the robot uids to use in the environment, make sure it match the uids use in the .pt file"""

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
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 1
    """update to data ratio"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""
    
    # target policy smoothing para
    exploration_noise: float = 0.1
    """the noise to add to the actions during data collection"""
    policy_noise: float = 0.1
    """"for target policy smoothing"""
    noise_clip: float = 0.5
    """"for target policy smoothing""" 

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""

    #wsrl
    warmup_steps: int = 5000
    """the number of warmup steps to take before starting training"""

    #wu_demo
    normalize_states: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""

    #jsrl
    algo_jsrl: bool = False
    """jsrl algo trigger"""

    #training sample para, all the prob is the env sample, for sigmoid function
    final_p: float = 0.5
    start_p: float = 0.5

    #sample tag
    sample_mode_original: bool = False

    #algo selector
    algo_num: int = 1

    #qfs number
    qfs_num: int = 2
    min_q_target: int = 2

    #cheq parameter 
    ulow: float = 0.005
    uhigh: float = 0.15
    lam_low: float = 0.2
    lam_high: float = 1.0
    p_masking: float =  0.8

    wandb_video_freq: int = 2

    action_mixing_trigger: bool = False

@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    sources: torch.Tensor
    # sources tag: 0 from env, 1 from wu_demo, 2 from wsrl

class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, demo_buffer_size: int, storage_device: torch.device, sample_device: torch.device, cheq_activate: bool):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.demo_pos = 0
        self.demo_full = False
        self.wu_active = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        self.demo_per_env_buffer_size = demo_buffer_size // num_envs
        if cheq_activate:
            orig_shape = env.single_observation_space.shape      # e.g. (48,)
            augmented_shape = (*orig_shape[:-1], orig_shape[-1] + 1)  # → (49,)
            self.obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + augmented_shape).to(storage_device)
            self.next_obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + augmented_shape).to(storage_device)
            self.demo_obs = torch.zeros((self.demo_per_env_buffer_size, self.num_envs) + augmented_shape).to(storage_device)
            self.demo_next_obs = torch.zeros((self.demo_per_env_buffer_size, self.num_envs) + augmented_shape).to(storage_device)
        else:
            self.obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
            self.next_obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
            self.demo_obs = torch.zeros((self.demo_per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
            self.demo_next_obs = torch.zeros((self.demo_per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_action_space.shape).to(storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.sources = torch.zeros((self.per_env_buffer_size, self.num_envs), dtype=torch.uint8).to(storage_device)

        self.demo_actions = torch.zeros((self.demo_per_env_buffer_size, self.num_envs) + env.single_action_space.shape).to(storage_device)
        self.demo_rewards = torch.zeros((self.demo_per_env_buffer_size, self.num_envs)).to(storage_device)
        self.demo_dones = torch.zeros((self.demo_per_env_buffer_size, self.num_envs)).to(storage_device)
        self.demo_sources = torch.zeros((self.demo_per_env_buffer_size, self.num_envs), dtype=torch.uint8).to(storage_device)

    def clean_samples(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        # Find row indices where any of the tensors are nan or too large
        nan_mask = torch.isnan(obs).any(dim=1) | torch.isnan(next_obs).any(dim=1) | torch.isnan(action).any(dim=1) | torch.isnan(reward)
        inf_mask = torch.isinf(reward)
        large_mask = (obs > 1e4).any(dim=1) | (next_obs > 1e4).any(dim=1) | (action > 1e4).any(dim=1)
        nan_or_large_indices = torch.where(nan_mask | large_mask | inf_mask)[0]
        if len(nan_or_large_indices) > 0:
            print(f"NaN or inf or large value detected: {len(nan_or_large_indices)} transitions converted to zero")
            obs[nan_or_large_indices] = 0
            next_obs[nan_or_large_indices] = 0
            action[nan_or_large_indices] = 0
            reward[nan_or_large_indices] = 0
            done[nan_or_large_indices] = 0
        return obs, next_obs, action, reward, done
    
    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, sources:torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        obs, next_obs, action, reward, done = self.clean_samples(obs, next_obs, action, reward, done)

        if args.sample_mode_original == True:
            self.obs[self.pos] = obs
            self.next_obs[self.pos] = next_obs
            self.actions[self.pos] = action
            self.rewards[self.pos] = reward
            self.dones[self.pos] = done
            self.sources[self.pos]  = torch.tensor(sources, dtype=torch.uint8, device=self.storage_device)

            self.pos += 1
            if self.pos == self.per_env_buffer_size:
                self.full = True
                self.pos = 0
        else:
            if sources != 0:
                self.demo_obs[self.demo_pos] = obs
                self.demo_next_obs[self.demo_pos] = next_obs
                self.demo_actions[self.demo_pos] = action
                self.demo_rewards[self.demo_pos] = reward
                self.demo_dones[self.demo_pos] = done
                self.demo_sources[self.demo_pos] = torch.tensor(sources, dtype=torch.uint8, device=self.storage_device)

                self.demo_pos += 1
                if self.demo_pos == self.demo_per_env_buffer_size:
                    self.demo_full = True
                    self.demo_pos = 0
            else:
                self.obs[self.pos] = obs
                self.next_obs[self.pos] = next_obs
                self.actions[self.pos] = action
                self.rewards[self.pos] = reward
                self.dones[self.pos] = done
                self.sources[self.pos] = torch.tensor(sources, dtype=torch.uint8, device=self.storage_device)

                self.pos += 1
                if self.pos == self.per_env_buffer_size:
                    self.full = True
                    self.pos = 0

    def sigmoid_function_prob(self, current_step: int, total_steps: int, start_p: float, final_p: float):
        # Handle edge case where start_p == final_p
        if abs(final_p - start_p) < 1e-6:
            return final_p

        # Inverse sigmoid at x=0 gives initial bias
        logit_start = math.log(start_p / (1.0 - start_p))
        logit_end = math.log(final_p / (1.0 - final_p))

        # Now solve for beta so that sigmoid(total_steps * beta + logit_start) = final_p
        beta = (logit_end - logit_start) / total_steps

        # Compute current prob at current_step
        logit = beta * current_step + logit_start
        return 1.0 / (1.0 + math.exp(-logit))

    def sample(self, batch_size: int, current_step: Optional[int] = None, total_steps: Optional[int] = None, start_p: Optional[float] = None, final_p: Optional[float] = None) -> ReplayBufferSample:
        if args.sample_mode_original == True:
            if self.full:
                batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size, ))
            else:
                batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
            env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
            return ReplayBufferSample(
                obs=self.obs[batch_inds, env_inds].to(self.sample_device),
                next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
                actions=self.actions[batch_inds, env_inds].to(self.sample_device),
                rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
                dones=self.dones[batch_inds, env_inds].to(self.sample_device),
                sources  = self.sources[batch_inds, env_inds].to(self.sample_device)
            )
        else: 
            assert None not in (current_step, total_steps, start_p, final_p), \
                "Must pass (current_step, total_steps, start_p, final_p) in 'mix' mode"
            p_env = self.sigmoid_function_prob(current_step, total_steps, start_p, final_p) #type: ignore
            k_env = int(torch.distributions.Binomial(batch_size, p_env).sample().long())

            # if demo buffer is empty, fallback to env
            if (self.demo_pos == 0 and not self.demo_full):
                return self._sample_from_buffer(batch_size, buffer='env')

            # draw k_env env samples and rest demo samples
            env_batch  = self._sample_from_buffer(k_env,              buffer='env')
            demo_batch = self._sample_from_buffer(batch_size - k_env, buffer='demo')

            # concatenate
            obs      = torch.cat([env_batch.obs,      demo_batch.obs],      dim=0)
            next_obs = torch.cat([env_batch.next_obs, demo_batch.next_obs], dim=0)
            actions  = torch.cat([env_batch.actions,  demo_batch.actions],  dim=0)
            rewards  = torch.cat([env_batch.rewards,  demo_batch.rewards],  dim=0)
            dones    = torch.cat([env_batch.dones,    demo_batch.dones],    dim=0)
            sources  = torch.cat([env_batch.sources,  demo_batch.sources],  dim=0)

            return ReplayBufferSample(obs, next_obs, actions, rewards, dones, sources)

    def _sample_from_buffer(self, batch_size: int, buffer: str) -> ReplayBufferSample:
        """
        Internal helper: draws uniformly from either 'env' or 'demo' circular buffer.
        """
        if buffer == 'env':
            buf_pos   = self.pos if not self.full else self.per_env_buffer_size
            obs_buf   = self.obs
            next_buf  = self.next_obs
            act_buf   = self.actions
            rew_buf   = self.rewards
            done_buf  = self.dones
            src_tag    = 0
        elif buffer == 'demo':
            buf_pos   = self.demo_pos if not self.demo_full else self.demo_per_env_buffer_size
            obs_buf   = self.demo_obs
            next_buf  = self.demo_next_obs
            act_buf   = self.demo_actions
            rew_buf   = self.demo_rewards
            done_buf  = self.demo_dones
            src_tag    = self.demo_sources  # already stored per‑element
        else:
            raise ValueError(f"Unknown buffer '{buffer}'")

        if buf_pos == 0:
            raise RuntimeError(f"No samples in {buffer} buffer yet")

        # pick random time‑indices and env‑indices
        idxs   = torch.randint(0, buf_pos, (batch_size,), device=self.storage_device)
        e_idxs = torch.randint(0, self.num_envs, (batch_size,),
                               device=self.storage_device)

        batch_obs      = obs_buf[idxs, e_idxs].to(self.sample_device)
        batch_next_obs = next_buf[idxs, e_idxs].to(self.sample_device)
        batch_acts     = act_buf[idxs, e_idxs].to(self.sample_device)
        batch_rews     = rew_buf[idxs, e_idxs].to(self.sample_device)
        batch_dones    = done_buf[idxs, e_idxs].to(self.sample_device)

        # source tags
        if buffer == 'env':
            batch_sources = torch.zeros(batch_size,
                                        dtype=torch.uint8,
                                        device=self.sample_device)
        else:
            batch_sources = src_tag[idxs, e_idxs].to(self.sample_device) #type: ignore 

        return ReplayBufferSample(
            obs     = batch_obs,
            next_obs= batch_next_obs,
            actions = batch_acts,
            rewards = batch_rews,
            dones   = batch_dones,
            sources = batch_sources,
        )
        
# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)

class WarmupActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(env.single_action_space.shape)),
            nn.Tanh()
        )
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_wu_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_wu_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x):
        action = self.net(x)
        return action * self.action_wu_scale + self.action_wu_bias

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(env.single_action_space.shape)),
            nn.Tanh()
        )
        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x):
        action = self.backbone(x)
        return action * self.action_scale + self.action_bias
    
class ActorWithLambda(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() +1, 256), # have to plus one cuz by the obs + lambda
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(env.single_action_space.shape)),
            nn.Tanh()
        )

        # precompute rescaling buffers
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, obs_plus_lambda: torch.Tensor) -> torch.Tensor:
        raw_action = self.backbone(obs_plus_lambda)
        return raw_action * self.action_scale + self.action_bias
    
class SoftQNetworkWithLambda(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear((np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)) + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs_plus_lam, a):
        x = torch.cat([obs_plus_lam, a], 1)
        return self.net(x)
    
class Logger:
    def __init__(self, log_wandb=False, tensorboard: Optional[SummaryWriter] = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step) # type: ignore
    def close(self):
        self.writer.close() # type: ignore

class CriticEsemble(nn.Module):
    def __init__(self, qfs: list[nn.Module], qfs_target: list[nn.Module], min_target_qfs: int, device):
        super().__init__()
        self.qfs = nn.ModuleList(qfs)
        self.qfs_target = nn.ModuleList(qfs_target)
        self.device = device
        self.min_qfs = min_target_qfs

    @torch.no_grad()
    def random_find_min_qf(self, obs, actions, qfs: bool = True):
        critics = self.qfs if qfs else self.qfs_target
        i, j = random.sample(range(len(critics)), 2)
        qi = critics[i](obs, actions).view(-1)
        qj = critics[j](obs, actions).view(-1)
        return torch.min(qi, qj)

    @torch.no_grad()    
    def random_close_qf(self, obs_plus_lam, actions):
        k = random.randint(self.min_qfs, len(self.qfs_target))
        idxs = random.sample(range(len(self.qfs_target)), k)
        min_q_val = torch.stack([self.qfs_target[i](obs_plus_lam, actions).view(-1) for i in idxs], dim=0)
        min_q_val = min_q_val.min(dim=0).values
        return min_q_val

class ManiSkillDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        device,
        normalize_states=False,
    ) -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]

        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.observations = []
        self.actions = []
        self.dones = []
        self.rewards = []
        self.total_frames = 0
        self.device = device
        load_count = len(self.episodes)
        print(f"Loading {load_count} episodes")

        for eps_id in tqdm.tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)

            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            self.observations.append(trajectory["obs"][:-1])
            self.actions.append(trajectory["actions"])
            self.rewards.append(trajectory["rewards"].reshape(-1, 1))
            self.dones.append(trajectory["success"].reshape(-1, 1))

        self.observations = np.vstack(self.observations)
        self.actions = np.vstack(self.actions)
        self.dones = np.vstack(self.dones)
        self.rewards = np.vstack(self.rewards)

        if normalize_states:
            mean, std = np.mean(self.observations), np.std(self.observations)
            self.observations = (self.observations - mean) / std

    def __len__(self):
        return len(self.observations) -1

    def __getitem__(self, idx):
        o  = torch.from_numpy(self.observations[idx]).float().to(self.device)
        no = torch.from_numpy(self.observations[idx+1]).float().to(self.device)
        a  = torch.from_numpy(self.actions[idx]).float().to(self.device)
        r  = torch.tensor(self.rewards[idx], dtype=torch.float32, device=self.device)
        d  = torch.from_numpy(self.dones[idx]).float().to(self.device)
        return o, a, r, d, no
    
class CheqAlgorithmTool:
    def __init__(self, qfs: list[nn.Module], ulow: float, uhigh: float, lam_low: float, lam_high: float):
        self.qfs = nn.ModuleList(qfs)
        self.ulow = ulow
        self.uhigh = uhigh
        self.lam_low = lam_low
        self.lam_high = lam_high

    def inject_weight_into_state(self, obs, lam):
        return torch.cat([obs, lam], dim=1) #[B, D+1]
    
    def compute_u(self, obs_plus_lambda, action):
        # print(f"obs{obs_plus_lam.shape}")
        # print(f"action{action.shape}")
        q_diff = torch.stack([qf(obs_plus_lambda, action) for qf in self.qfs], dim=1)
        return q_diff.std(dim=1, unbiased=False)
    
    def get_lambda(self, u):
        lam_vals = torch.empty_like(u)
        lower_mask = (u <= self.ulow)
        upper_mask = (u >= self.uhigh)
        mid_mask   = ~(lower_mask | upper_mask)

        # print(f"[get_lambda] lower={lower_mask.sum().item()}  upper={upper_mask.sum().item()}  mid={mid_mask.sum().item()}")


        lam_vals[lower_mask] = self.lam_high
        lam_vals[upper_mask] = self.lam_low

        mid_u = u[mid_mask]
        frac = (mid_u - self.uhigh)/(self.ulow - self.uhigh)  # in [0,1]
        lam_vals[mid_mask] = self.lam_low + frac*(self.lam_high - self.lam_low)

        return lam_vals  # shape [N,1]    

def algo_selector(algo_num: int) -> tuple[bool, bool, bool, bool]:
    # wu_demo, WSRL, IBRL, CHEQ 
    options = {
        1: (False, False, False, False),   #TD3
        2: (True,  False, False, False),   #TD3+wu_demo
        3: (False, True,  False, False),   #TD3+WSRL
        4: (True,  False, True, False),    #TD3+wu_demo+IBRL
        5: (False, True,  True, False),    #TD3+WSRL+IBRL
        6: (False, False, False, True),    #TD3+CHEQ
        7: (True, False, False, True),     #TD3+wu_demo+CHEQ
        8: (False, True, False, True),     #TD3+WSRL+CHEQ

    }
    return options.get(algo_num, (False, False, False, False))

class RecordEpisodeWandb(RecordEpisode):
    def __init__(
        self,
        env,
        output_dir: str,
        wandb_video_freq: Optional[int] = 0,
        **kwargs,
    ) -> None:
        super().__init__(env, output_dir, **kwargs)
        self.wandb_video_freq = wandb_video_freq


    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
    ):
        super().flush_video(name, suffix, verbose, ignore_empty_transition, save)
        if save:
            if name is None:
                video_name = "{}".format(self._video_id)
                if suffix:
                    video_name += "_" + suffix
            else:
                video_name = name
            if self.wandb_video_freq != 0 and self._video_id % self.wandb_video_freq == 0: #type: ignore
                print(f"Logging video {video_name} to wandb")
                video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
                wandb.log({"video": wandb.Video(f"{self.output_dir}/{video_name}", fps=self.video_fps)}) # add step

def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    # add the name label here

    wu_demo, wsrl, ibrl, cheq= algo_selector(args.algo_num)
    seed = random.randint(1, 10000)

    if args.exp_name is None:
        run_name = f"{args.env_id}__{args.algo_num}__{args.sample_mode_original}__{seed}__{args.utd}__{args.robot_uids}__{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu", robot_uids=args.robot_uids)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs) # type: ignore
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs) # type: ignore
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint or '')}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            # envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30) # type: ignore 
            envs = RecordEpisodeWandb(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        # eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30) # type: ignore
        eval_envs = RecordEpisodeWandb(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30,
                                       wandb_video_freq=(args.wandb_video_freq if args.track else 0)) # type: ignore
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group= args.wandb_group if args.wandb_group is not None else str(args.algo_num),
                tags=["wsrl", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    if cheq:
        actor_policy = ActorWithLambda(envs).to(device)
        actor_target = ActorWithLambda(envs).to(device)
        qfs = nn.ModuleList([SoftQNetworkWithLambda(envs).to(device) for _ in range(args.qfs_num)])
        qfs_target = nn.ModuleList([SoftQNetworkWithLambda(envs).to(device) for _ in range(args.qfs_num)])        
    else:
        actor_policy = Actor(envs).to(device)
        actor_target = Actor(envs).to(device)
        qfs = nn.ModuleList([SoftQNetwork(envs).to(device) for _ in range(args.qfs_num)])
        qfs_target = nn.ModuleList([SoftQNetwork(envs).to(device) for _ in range(args.qfs_num)])

    for i in range(len(qfs_target)):
        qfs_target[i].load_state_dict(qfs[i].state_dict())
    actor_target.load_state_dict(actor_policy.state_dict())
    critic_ensemble = CriticEsemble(list(qfs) ,list(qfs_target), min_target_qfs=args.min_q_target, device=device)
    cheq_algo_tool = CheqAlgorithmTool(list(qfs), args.ulow, args.uhigh, args.lam_low, args.lam_high)

    q_optimizer = optim.Adam([p for qf in qfs for p in qf.parameters()], lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor_policy.parameters()), lr=args.policy_lr) 

    #warmup_policy
    offline_policy = WarmupActor(envs).to(device)
    policy_path = f"examples/baselines/ibrl/policy/best_eval_success_once_{args.env_id}.pt"
    offline_policy.load_state_dict(torch.load(policy_path, map_location=device)["actor"], strict=False)  

    envs.single_observation_space.dtype = np.float32 # type: ignore
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        demo_buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device, 
        cheq_activate=cheq
    )

    #warmup phase
    if wsrl:
        offline_policy.eval()        
        wu_obs, _ = envs.reset(seed=seed)
        for _ in tqdm.tqdm(range(args.warmup_steps // args.num_envs), desc="Warmup phase", total=args.warmup_steps // args.num_envs):
            with torch.no_grad():
                wu_actions = offline_policy(wu_obs)
            next_wu_obs, wu_rew, wu_terminations, wu_truncations, _ = envs.step(wu_actions)
            if args.bootstrap_at_done == 'never':
                need_final_obs = torch.ones_like(wu_terminations, dtype=torch.bool) # type: ignore
                wu_stop_bootstrap = wu_truncations | wu_terminations # type: ignore # always stop bootstrap when episode ends
            else:
                if args.bootstrap_at_done == 'always':
                    need_final_obs = wu_truncations | wu_terminations # type: ignore # always need final obs when episode ends
                    wu_stop_bootstrap = torch.zeros_like(wu_terminations, dtype=torch.bool) # type: ignore # never stop bootstrap
                else: # bootstrap at truncated
                    need_final_obs = wu_truncations & (~wu_terminations) # type: ignore # only need final obs when truncated and not terminated
                    wu_stop_bootstrap = wu_terminations # only stop bootstrap when terminated, don't stop when truncated

            if cheq:
                wu_lam_feed = torch.zeros((args.num_envs, 1), device=device)
                wu_obs_add = cheq_algo_tool.inject_weight_into_state(wu_obs, wu_lam_feed)
                next_wu_obs_add = cheq_algo_tool.inject_weight_into_state(next_wu_obs, wu_lam_feed)
                rb.add(wu_obs_add, next_wu_obs_add, wu_actions, wu_rew, wu_stop_bootstrap, sources=2) # type: ignore
            else:
                rb.add(wu_obs, next_wu_obs, wu_actions, wu_rew, wu_stop_bootstrap, sources=2) # type: ignore
            wu_obs = next_wu_obs
        print("✅ Warm‑up policy phase complete.")

    elif wu_demo:
        demo_path = f"examples/baselines/ibrl/demos/{args.env_id}.state.pd_joint_vel.physx_cpu.h5"
        # are there any function that can load demo data?\
        ds = ManiSkillDataset(
        dataset_file=demo_path,
        device=device,
        normalize_states=args.normalize_states,
        )

        demo_loader = DataLoader(ds, batch_size=1, shuffle=True)
        for obs, action, reward, done, next_obs in tqdm.tqdm(demo_loader, desc="feeding RB from demos"):

            if cheq:
                obs = obs.repeat(args.num_envs, 1)
                next_obs = next_obs.repeat(args.num_envs, 1)
                wu_lam_feed = torch.zeros((args.num_envs, 1), device=device)
                obs = cheq_algo_tool.inject_weight_into_state(obs, wu_lam_feed)
                next_obs = cheq_algo_tool.inject_weight_into_state(next_obs, wu_lam_feed) 

            rb.add(obs, next_obs, action, reward, done, sources=1) # type: ignore
        print("✅ Warm‑up demo phase complete.")

    else:
        print("❌ Skipping warm‑up phase.")
        

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=seed)
    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    if cheq:
        lam_vals = torch.full((args.num_envs, 1), args.lam_low, device=device)

    while global_step < args.total_timesteps:

        # sources tag debugger
        # flat = rb.sources.view(-1)
        # tags = torch.unique(flat)
        # print(f"[step {global_step}] buffer contains tags {tags.tolist()}")

        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # evaluate
            actor_policy.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    if cheq:
                        eval_lam_vals = torch.full((args.num_eval_envs, 1),lam_vals[0, 0].item(), device=device) # type: ignore
                        obs_plus_lam_eval = cheq_algo_tool.inject_weight_into_state(eval_obs, eval_lam_vals)
                        eval_a_rl = actor_policy(obs_plus_lam_eval)
                        eval_a_il = offline_policy(eval_obs)

                        final_action = eval_lam_vals * eval_a_rl + (1 - eval_lam_vals) * eval_a_il

                        eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(final_action)
                    else:
                        eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor_policy(eval_obs))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
            pbar.set_description(
                f"success_once: {eval_metrics_mean['success_once']:.2f}, "
                f"return: {eval_metrics_mean['return']:.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
            actor_policy.train()

            if args.save_model:
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                checkpoint = {'actor': actor_policy.state_dict()}
                for i, qf_target in enumerate(qfs_target):
                    checkpoint[f"qf{i+1}"] = qf_target.state_dict()
                torch.save(checkpoint, model_path)
                print(f"model saved to {model_path}")

        action_low = torch.tensor(envs.single_action_space.low, device=device)
        action_high = torch.tensor(envs.single_action_space.high, device=device)  

        # Collect samples from environemnts
        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            # ALGO LOGIC: put action logic here
            if not learning_has_started:
                with torch.no_grad():
                    if cheq:
                        a_il = offline_policy(obs)
                        a_rl = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
                        obs_plus_lam = cheq_algo_tool.inject_weight_into_state(obs, lam_vals) # type: ignore
                        actions = lam_vals * a_rl + (1-lam_vals) * a_il # type: ignore
                    else:
                        actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            # ibrl action selection part
            elif ibrl:
                with torch.no_grad():
                    a_il = offline_policy(obs)
                    a_rl = (actor_policy(obs) + torch.randn_like(actor_policy(obs)) * args.exploration_noise).clamp(action_low, action_high)
                    
                    q_il = critic_ensemble.random_find_min_qf(obs, a_il) 
                    q_rl = critic_ensemble.random_find_min_qf(obs, a_rl) 

                    mask = (q_il > q_rl).unsqueeze(-1) 
                    actions = torch.where(mask, a_il, a_rl)

            elif cheq:
                with torch.no_grad():
                    obs_plus_lam = cheq_algo_tool.inject_weight_into_state(obs, lam_vals) # type: ignore
                    a_il = offline_policy(obs)
                    a_rl = actor_policy(obs_plus_lam)
                    actions = lam_vals * a_rl + (1-lam_vals) * a_il #type: ignore

                    logger.add_scalar("collect phase/lam_val(t)", lam_vals.mean().item(), global_step) # type: ignore
                    noise = torch.randn_like(actions) * args.exploration_noise
                    actions = (actions + noise).clamp(action_low, action_high)

                    if args.action_mixing_trigger:
                        u_val = cheq_algo_tool.compute_u(obs_plus_lam, actions)
                    else: 
                        u_val = cheq_algo_tool.compute_u(obs_plus_lam, a_rl)
                    lam_vals = cheq_algo_tool.get_lambda(u_val)
                    logger.add_scalar("collect phase/u_val(t+1)", u_val.mean().item(), global_step) # type: ignore
                    logger.add_scalar("collect phase/lam_val(t+1)", lam_vals.mean().item(), global_step) # type: ignore
         
            else:
                with torch.no_grad():
                    actions = actor_policy(obs)
                    # Add exploration noise
                    noise = torch.randn_like(actions) * args.exploration_noise
                    actions = (actions + noise).clamp(action_low, action_high)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # Log NaN/Inf statistics
            # n_obs_nan  = torch.isnan(next_obs).sum().item()
            # n_obs_inf  = torch.isinf(next_obs).sum().item()
            # n_rew_nan  = torch.isnan(rewards).sum().item()
            # n_rew_inf  = torch.isinf(rewards).sum().item()
            # print(f"[STEP] next_obs NaN={n_obs_nan}, Inf={n_obs_inf}; rewards NaN={n_rew_nan}, Inf={n_rew_inf}")
            # print(f"[STEP] next_obs min/max = {next_obs.min().item():.3f}/{next_obs.max().item():.3f}")
            # print(f"[STEP] rewards  min/max = {rewards.min().item():.3f}/{rewards.max().item():.3f}")

            real_next_obs = next_obs.clone() # type: ignore
            if args.bootstrap_at_done == 'never':
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool) # type: ignore
                stop_bootstrap = truncations | terminations # type: ignore # always stop bootstrap when episode ends
            else:
                if args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations # type: ignore # always need final obs when episode ends
                    stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool) # type: ignore # never stop bootstrap
                else: # bootstrap at truncated
                    need_final_obs = truncations & (~terminations) # type: ignore # only need final obs when truncated and not terminated
                    stop_bootstrap = terminations # only stop bootstrap when terminated, don't stop when truncated
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs] # type: ignore
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step) # type: ignore
            if cheq:
                next_obs_plus_lam = cheq_algo_tool.inject_weight_into_state(real_next_obs, lam_vals)# type: ignore
                rb.add(obs_plus_lam, next_obs_plus_lam, actions, rewards, stop_bootstrap, sources=0)# type: ignore
            else:
                rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap, sources=0) # type: ignore

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # ALGO LOGIC: training.
        if global_step < args.warmup_steps:
            continue

        update_time = time.perf_counter()
        learning_has_started = True

        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            # check it's offline+online or only online run
            # data = rb.sample_test(args.batch_size, args.sample_mode ,global_step, args.total_timesteps, args.start_p, args.final_p)   
            data = rb.sample(args.batch_size, global_step, args.total_timesteps, args.start_p, args.final_p)
            # log the soruce distribution
            unique_sources, counts = torch.unique(data.sources, return_counts=True)
            fracs = counts.float() / data.sources.shape[0]
            for src, frac in zip(unique_sources.tolist(), fracs.tolist()):
                logger.add_scalar(f"data/source_{src}_frac", frac, global_step) # type: ignore

            if ibrl:
                with torch.no_grad():
                    next_a_rl = actor_target(data.next_obs)
                    noise = (torch.randn_like(data.actions) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                    next_a_rl = (next_a_rl + noise).clamp(action_low, action_high)
                    next_a_il = offline_policy(data.next_obs)

                    next_min_q_target_rl = critic_ensemble.random_find_min_qf(data.next_obs, next_a_rl, qfs=False)
                    next_min_q_target_il = critic_ensemble.random_find_min_qf(data.next_obs, next_a_il, qfs=False)
                    next_max_q_target_val = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * torch.max(next_min_q_target_rl, next_min_q_target_il)

                qf_losses = [F.mse_loss(qf(data.obs, data.actions).view(-1), next_max_q_target_val) for qf in qfs]
                qf_loss_total = torch.stack(qf_losses, dim=0).sum()

                q_optimizer.zero_grad()
                qf_loss_total.backward()
                q_optimizer.step() 

                if global_update % args.policy_frequency == 0:
                    pi = actor_policy(data.obs)
                    qs = [qf(data.obs, pi) for qf in qfs]
                    qs_cat = torch.cat(qs, dim=1)
                    min_q_pi, _ = qs_cat.min(dim=1, keepdim=True)
                    
                    actor_loss = -min_q_pi.mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    #target network update
                    for qf, qf_target in zip(qfs, qfs_target):
                        for qf_para, qf_target_para in zip(qf.parameters(), qf_target.parameters()):
                            qf_target_para.data.copy_(args.tau * qf_para.data + (1 - args.tau) * qf_target_para.data)
                    for p, p_target in zip(actor_policy.parameters(), actor_target.parameters()):
                        p_target.data.copy_(args.tau * p.data + (1 - args.tau) * p_target.data)   
            
            elif cheq:
                with torch.no_grad():
                    next_action = actor_target(data.next_obs)
                    noise = (torch.randn_like(data.actions) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                    next_action = (next_action + noise).clamp(action_low, action_high)   
                    min_q_val = critic_ensemble.random_close_qf(data.next_obs, next_action)  
                    min_q_target_val = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_q_val

                mask = torch.bernoulli(torch.full((len(qfs),), args.p_masking, device=device))
                qf_losses = [m * F.mse_loss(qf(data.obs, data.actions).view(-1), min_q_target_val) for m, qf in zip(mask, qfs)]
                qf_loss_total = torch.stack(qf_losses, dim=0).sum()

                q_optimizer.zero_grad()
                qf_loss_total.backward()
                q_optimizer.step()  

                if global_update % args.policy_frequency == 0:
                    pi = actor_policy(data.obs)
                    pi_q_list_val = [qf(data.obs, pi) for qf in qfs]
                    min_pi_q_val, _ = torch.cat(pi_q_list_val, dim=1).min(dim=1, keepdim=True)

                    actor_loss = -min_pi_q_val.mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    for qf, qf_target in zip(qfs, qfs_target):
                        for qf_para, qf_target_para in zip(qf.parameters(), qf_target.parameters()):
                            qf_target_para.data.copy_(args.tau * qf_para.data + (1 - args.tau) * qf_target_para.data)
                    for p, p_target in zip(actor_policy.parameters(), actor_target.parameters()):
                        p_target.data.copy_(args.tau * p.data + (1 - args.tau) * p_target.data)   
                    
            # update the value networks
            else: 
                with torch.no_grad():
                    next_actions= actor_target(data.next_obs)
                    noise = (torch.randn_like(data.actions) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                    next_state_actions = (next_actions + noise).clamp(action_low, action_high)

                    #double q learning
                    min_qf_next_target = critic_ensemble.random_find_min_qf(data.next_obs, next_state_actions, qfs=False)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf_losses = [F.mse_loss(qf(data.obs, data.actions).view(-1), next_q_value) for qf in qfs]
                qf_loss_total = torch.stack(qf_losses, dim=0).sum()
                q_optimizer.zero_grad()
                qf_loss_total.backward()
                q_optimizer.step() 

                # update the policy network
                if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                    pi = actor_policy(data.obs)  
                    qs = [qf(data.obs, pi) for qf in qfs]
                    qs_cat = torch.cat(qs, dim=1)
                    min_q_pi, _ = qs_cat.min(dim=1, keepdim=True)
                    
                    actor_loss = -min_q_pi.mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                # update the target networks
                    for qf, qf_target in zip(qfs, qfs_target):
                        for qf_para, qf_target_para in zip(qf.parameters(), qf_target.parameters()):
                            qf_target_para.data.copy_(args.tau * qf_para.data + (1 - args.tau) * qf_target_para.data)
                    for param, target_param in zip(actor_policy.parameters(), actor_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            for i, (qf, loss) in enumerate(zip(qfs, qf_losses), start=1): # type: ignore
                q_vals = qf(data.obs, data.actions).view(-1) # type: ignore
                logger.add_scalar(f"losses/qf{i}_values", q_vals.mean().item(), global_step) # type: ignore
                logger.add_scalar(f"losses/qf{i}_loss",  loss.item(), global_step) # type: ignore
            logger.add_scalar("losses/qf_loss", (qf_loss_total / len(qfs)).item(), global_step) # type: ignore
            if global_update % args.policy_frequency == 0:
                logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step) # type: ignore
            logger.add_scalar("time/update_time", update_time, global_step) # type: ignore
            logger.add_scalar("time/rollout_time", rollout_time, global_step) # type: ignore
            logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step) # type: ignore
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step) # type: ignore
            logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step) # type: ignore

    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        final_checkpoint = {'actor': actor_policy.state_dict()}
        for i, qf_target in enumerate(qfs_target):
            final_checkpoint[f"qf{i+1}"] = qf_target.state_dict()
        torch.save(final_checkpoint, model_path)
        print(f"model saved to {model_path}")
        writer.close() # type: ignore
    envs.close()
