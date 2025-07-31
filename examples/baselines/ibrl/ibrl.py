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
import tyro

import mani_skill.envs


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
    algo_wsrl: bool = False
    """whether to use the warmup policy for action selection during warmup phase"""

    #wu_demo
    wu_demo: bool = False
    """whether to use the warmup demo for action selection during warmup phase"""
    normalize_states: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""

    #ibrl
    algo_ibrl: bool = False
    """if false, use TD3, if true, use IBRL"""

    #jsrl
    algo_jsrl: bool = False
    """jsrl algo trigger"""

    #training sample para, all the prob is the env sample, for sigmoid function
    final_p: float = 0.99
    start_p: float = 0.2

    #sample tag
    sample_mode_original: bool = False

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
    def __init__(self, env, num_envs: int, buffer_size: int, demo_buffer_size: int, storage_device: torch.device, sample_device: torch.device):
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
        self.obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_action_space.shape).to(storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.sources = torch.zeros((self.per_env_buffer_size, self.num_envs), dtype=torch.uint8).to(storage_device)

        self.demo_obs = torch.zeros((self.demo_per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.demo_next_obs = torch.zeros((self.demo_per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
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

class CriticEsemble:
    def __init__(self, q1: nn.Module, q2: nn.Module):
        self.q1 = q1
        self.q2 = q2

    @torch.no_grad()
    def clipped_q_min(self, obs, actions):
        """Return min_i Q_i(obs, actions), shape [batch,1]."""
        q1 = self.q1(obs, actions).view(-1)
        q2 = self.q2(obs, actions).view(-1)
        return torch.min(q1, q2)


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

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

FEATURE_LABELS = {
    "algo_ibrl": "IBRL",
    "algo_wsrl": "WSRL",
    "algo_jsrl": "JSRL",
    "wu_demo":   "wu_demo",
    # → when you add a new bool flag, just drop it in here
}

def NameBuilder(**flags):
    base = "TD3"
    # pick out only the flags that are true
    selected = [FEATURE_LABELS[name]
                for name, is_on in flags.items()
                if is_on and name in FEATURE_LABELS]

    if not selected:
        return base

    # join all the active pieces with " + "
    return base + " + " + " + ".join(selected)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    # add the name label here
    algo_name = NameBuilder(
        algo_ibrl=args.algo_ibrl,
        algo_wsrl=args.algo_wsrl,
        algo_jsrl = args.algo_jsrl,
        wu_demo=args.wu_demo,
    )

    seed = random.randint(1, 10000)

    if args.exp_name is None:
        run_name = f"{args.env_id}__{algo_name}__{args.sample_mode_original}__{seed}__{args.utd}__{args.robot_uids}__{time.strftime('%Y-%m-%d_%H-%M-%S')}"
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
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30) # type: ignore
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30) # type: ignore
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
                group= args.wandb_group if args.wandb_group is not None else algo_name,
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

    max_action = float(envs.single_action_space.high[0])

    # add actor target
    actor_policy = Actor(envs).to(device)
    actor_target = Actor(envs).to(device)

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)

    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor_policy.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    actor_target.load_state_dict(actor_policy.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
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
        sample_device=device
    )

    #warmup phase
    if args.algo_wsrl:
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
            rb.add(wu_obs, next_wu_obs, wu_actions, wu_rew, wu_stop_bootstrap, sources=2) # type: ignore
            wu_obs = next_wu_obs
        print("✅ Warm‑up policy phase complete.")

    elif args.wu_demo:
        demo_path = f"demos/{args.env_id}.state.pd_joint_vel.physx_cpu.h5"
        # are there any function that can load demo data?\
        ds = ManiSkillDataset(
        dataset_file=demo_path,
        device=device,
        normalize_states=args.normalize_states,
        )

        demo_loader = DataLoader(ds, batch_size=1, shuffle=True)
        for obs, action, reward, done, next_obs in tqdm.tqdm(demo_loader, desc="feeding RB from demos"):
            rb.add(obs, next_obs, action, reward, done, sources=1) # type: ignore
        print("✅ Warm‑up demo phase complete.")

    else:
        print("❌ Skipping warm‑up phase.")
        args.warmup_steps = 0
        

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=seed)
    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

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
                torch.save({
                    'actor': actor_policy.state_dict(),
                    'qf1': qf1_target.state_dict(),
                    'qf2': qf2_target.state_dict(),
                    # 'log_alpha': log_alpha,
                }, model_path)
                print(f"model saved to {model_path}")

        action_low = torch.tensor(envs.single_action_space.low, device=device)
        action_high = torch.tensor(envs.single_action_space.high, device=device)  

        # Collect samples from environemnts
        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            # ALGO LOGIC: put action logic here
            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            # ibrl action selection part
            elif args.algo_ibrl:
                with torch.no_grad():
                    a_il = offline_policy(obs)
                    a_rl = (actor_policy(obs) + torch.randn_like(actor_policy(obs)) * args.exploration_noise).clamp(action_low, action_high)
                    
                    critic_ensemble = CriticEsemble(qf1, qf2)
                    q_il = critic_ensemble.clipped_q_min(obs, a_il) 
                    q_rl = critic_ensemble.clipped_q_min(obs, a_rl) 

                    mask = (q_il > q_rl).unsqueeze(-1) 
                    actions = torch.where(mask, a_il, a_rl)
            elif args.algo_jsrl:
                with torch.no_grad():
                    a_il = offline_policy(obs)
                    a_rl = (actor_policy(obs) + torch.randn_like(actor_policy(obs)) * args.exploration_noise).clamp(action_low, action_high)

                    js_ratio = global_step/args.total_timesteps
                    mask = (torch.rand(len(obs), device=device) < js_ratio)
                    actions = torch.where(mask.unsqueeze(-1), a_rl, a_il)   
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

            if args.algo_ibrl:
                with torch.no_grad():
                    next_a_rl = actor_target(data.next_obs)
                    noise = (torch.randn_like(data.actions) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                    next_a_rl = (next_a_rl + noise).clamp(action_low, action_high)
                    next_a_il = offline_policy(data.next_obs)

                    critic_ensemble = CriticEsemble(qf1_target, qf2_target)
                    next_min_q_rl = critic_ensemble.clipped_q_min(data.next_obs, next_a_rl)
                    next_min_q_il = critic_ensemble.clipped_q_min(data.next_obs, next_a_il)

                    next_max_q_val = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * torch.max(next_min_q_rl, next_min_q_il)
                    train_mask = (next_min_q_il > next_min_q_rl).unsqueeze(-1) # 0 for RL, 1 for IL

                qf1_a_values = qf1(data.obs, data.actions).view(-1)
                qf2_a_values = qf2(data.obs, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_max_q_val)
                qf2_loss = F.mse_loss(qf2_a_values, next_max_q_val)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step() 

                for p, p_target in zip(qf1.parameters(), qf1_target.parameters()):
                    p_target.data.copy_(args.tau * p.data + (1 - args.tau) * p_target.data)
                for p, p_target in zip(qf2.parameters(), qf2_target.parameters()):
                    p_target.data.copy_(args.tau * p.data + (1 - args.tau) * p_target.data)  

                if global_update % args.policy_frequency == 0:
                    pi = actor_policy(data.obs)
                    q1_pi = qf1(data.obs, pi)
                    q2_pi = qf2(data.obs, pi)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    
                    actor_loss = -min_q_pi.mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                # soft‐update actor
                if global_update % args.target_network_frequency == 0:
                    for p, p_target in zip(actor_policy.parameters(), actor_target.parameters()):
                        p_target.data.copy_(args.tau * p.data + (1 - args.tau) * p_target.data)   
                    
            # update the value networks
            else: 
                with torch.no_grad():
                    next_actions= actor_target(data.next_obs)
                    noise = (torch.randn_like(data.actions) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                    next_state_actions = (next_actions + noise).clamp(action_low, action_high)

                    #double q learning
                    qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                    qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                    # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

                qf1_a_values = qf1(data.obs, data.actions).view(-1)
                qf2_a_values = qf2(data.obs, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # update the policy network
                if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                    # find minimun actor loss
                    qf1_pi = qf1(data.obs, actor_policy(data.obs))
                    qf2_pi = qf2(data.obs, actor_policy(data.obs))
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    # actor loss
                    actor_loss = -min_qf_pi.mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                # update the target networks
                if global_update % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    # target_actor iteration
                    for param, target_param in zip(actor_policy.parameters(), actor_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step) # type: ignore
            logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step) # type: ignore
            logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step) # type: ignore
            logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step) # type: ignore
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step) # type: ignore
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
        torch.save({
            'actor': actor_policy.state_dict(),
            'qf1': qf1_target.state_dict(),
            'qf2': qf2_target.state_dict(),
            # 'log_alpha': log_alpha,
        }, model_path)
        print(f"model saved to {model_path}")
        writer.close() # type: ignore
    envs.close()
