#!/usr/bin/env python

import os
import random
import time
import json
import h5py
import tyro
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataclasses import dataclass
from typing import Optional
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


@dataclass
class Args:
    # Basic experiment / logging
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    wandb_group: str = "SAC"
    capture_video: bool = True
    save_trajectory: bool = False
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None
    log_freq: int = 1000
    eval_freq: int = 1000  
    save_model_dir: Optional[str] = "runs"

    # CQL hyperparams
    cql_variant: str = "R"
    offline_training_steps: int = 100000  
    buffer_size: int = 1000000
    buffer_device: str = "cuda"
    batch_size: int = 256
    gamma: float = 0.95
    tau: float = 0.01
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 1
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    cql_alpha: float = 15.0
    cql_num_random: int = 10
    cql_temp: float = 1.0

    env_id: str = "PickCube-v1"
    env_vectorization: str = "gpu"
    num_eval_envs: int = 16
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = "pd_joint_delta_pos"
    num_eval_steps: int = 50
    eval_partial_reset: bool = False

    demo_path: Optional[str] = None

    save_buffer_path: str = "/content/replay_buffer.h5"

    # left from older code
    grad_steps_per_iteration: int = 0
    steps_per_env: int = 0


@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int,
                 storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.per_env_buffer_size = buffer_size // num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device

        self.pos = 0
        self.full = False

        obs_shape = env.single_observation_space.shape
        act_shape = env.single_action_space.shape

        self.obs = torch.zeros((self.per_env_buffer_size, num_envs) + obs_shape,
                               device=self.storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, num_envs) + obs_shape,
                                    device=self.storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, num_envs) + act_shape,
                                   device=self.storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, num_envs),
                                   device=self.storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, num_envs),
                                 device=self.storage_device)

    def add(self, obs_batch, next_obs_batch, action_batch, reward_batch, done_batch):
        if reward_batch.ndim == 2:
            reward_batch = reward_batch.squeeze(-1)
        if done_batch.ndim == 2:
            done_batch = done_batch.squeeze(-1)

        self.obs[self.pos] = obs_batch
        self.next_obs[self.pos] = next_obs_batch
        self.actions[self.pos] = action_batch
        self.rewards[self.pos] = reward_batch
        self.dones[self.pos] = done_batch

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSample:
        max_idx = self.per_env_buffer_size if self.full else self.pos
        time_inds = torch.randint(0, max_idx, (batch_size,))
        env_inds = torch.randint(0, self.num_envs, (batch_size,))

        return ReplayBufferSample(
            obs=self.obs[time_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[time_inds, env_inds].to(self.sample_device),
            actions=self.actions[time_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[time_inds, env_inds].to(self.sample_device),
            dones=self.dones[time_inds, env_inds].to(self.sample_device),
        )


def save_replay_buffer(rb: ReplayBuffer, filepath: str):
    num_samples = rb.per_env_buffer_size if rb.full else rb.pos
    obs_np = rb.obs[:num_samples].cpu().numpy().reshape(-1, *rb.obs.shape[2:])
    next_obs_np = rb.next_obs[:num_samples].cpu().numpy().reshape(-1, *rb.obs.shape[2:])
    actions_np = rb.actions[:num_samples].cpu().numpy().reshape(-1, *rb.actions.shape[2:])
    rewards_np = rb.rewards[:num_samples].cpu().numpy().reshape(-1)
    dones_np = rb.dones[:num_samples].cpu().numpy().reshape(-1)

    print(f"Saving replay buffer with {obs_np.shape[0]} transitions to {filepath}")
    with h5py.File(filepath, "w") as f:
        f.create_dataset("obs", data=obs_np)
        f.create_dataset("next_obs", data=next_obs_np)
        f.create_dataset("actions", data=actions_np)
        f.create_dataset("rewards", data=rewards_np)
        f.create_dataset("dones", data=dones_np)


def _load_h5_data_recursive(h5_obj):
    out = {}
    for k in h5_obj.keys():
        if isinstance(h5_obj[k], h5py.Dataset):
            out[k] = h5_obj[k][:]
        else:
            out[k] = _load_h5_data_recursive(h5_obj[k])
    return out


def populate_replay_buffer_from_demo(rb: ReplayBuffer, demo_path: str, device: torch.device):
    print(f"Loading demonstration from {demo_path}")
    json_file = demo_path.replace(".h5", ".json")
    with open(json_file, "r") as f:
        info = json.load(f)
    episodes_info = info["episodes"]

    with h5py.File(demo_path, "r") as demo_file:
        from tqdm import tqdm
        for ep_dict in tqdm(episodes_info, desc="Loading episodes"):
            ep_id = ep_dict["episode_id"]
            traj_group = demo_file[f"traj_{ep_id}"]
            traj_data = _load_h5_data_recursive(traj_group)

            obs_array = traj_data["obs"]
            act_array = traj_data["actions"]
            if "rewards" in traj_data:
                rew_array = traj_data["rewards"]
            elif "success" in traj_data:
                rew_array = traj_data["success"].astype(np.float32)
            else:
                rew_array = np.zeros(len(act_array), dtype=np.float32)

            if "success" in traj_data:
                done_array = traj_data["success"]
            else:
                done_array = np.zeros(len(act_array), dtype=np.float32)

            T = len(act_array)
            chunk_size = rb.num_envs
            for t in range(0, T, chunk_size):
                if t + chunk_size > T:
                    break
                obs_block = obs_array[t : t+chunk_size]
                next_obs_block = obs_array[t+1 : t+1+chunk_size]
                act_block = act_array[t : t+chunk_size]
                rew_block = rew_array[t : t+chunk_size]
                done_block = done_array[t : t+chunk_size]

                obs_torch = torch.tensor(obs_block, dtype=torch.float32, device=device)
                next_obs_torch = torch.tensor(next_obs_block, dtype=torch.float32, device=device)
                act_torch = torch.tensor(act_block, dtype=torch.float32, device=device)
                rew_torch = torch.tensor(rew_block, dtype=torch.float32, device=device)
                done_torch = torch.tensor(done_block, dtype=torch.float32, device=device)

                rb.add(obs_torch, next_obs_torch, act_torch, rew_torch, done_torch)

    print("Finished loading demonstration data.")


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)

        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x):
        hidden = self.backbone(x)
        mean = self.fc_mean(hidden)
        log_std = self.fc_logstd(hidden)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        y = torch.tanh(z)
        action = y * self.action_scale + self.action_bias

        log_prob = normal.log_prob(z)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action

    def get_eval_action(self, x):
        with torch.no_grad():
            mean, _ = self(x)
            y = torch.tanh(mean)
            action = y * self.action_scale + self.action_bias
        return action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: Optional[SummaryWriter] = None):
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
def evaluate_offline(args, actor, device, logger, global_step):
    actor.eval()
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend=args.env_vectorization)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    eval_envs = gym.make(args.env_id,
                         num_envs=args.num_eval_envs,
                         reconfiguration_freq=args.eval_reconfiguration_freq,
                         **env_kwargs)
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    ignore_terminations = not args.eval_partial_reset
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=ignore_terminations, record_metrics=True)

    obs, _ = eval_envs.reset()
    ep_returns = []
    ep_success = []
    for _ in range(args.num_eval_steps):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            actions = actor.get_eval_action(obs_t)
            actions = actions.cpu().numpy()
        next_obs, rewards, terms, truncs, infos = eval_envs.step(actions)
        obs = next_obs
        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                if k == "return":
                    ep_returns.append(v[done_mask].mean().item())
                elif k == "success_once":
                    ep_success.append(v[done_mask].float().mean().item())
    avg_return = np.mean(ep_returns) if len(ep_returns) > 0 else 0.0
    avg_success = np.mean(ep_success) if len(ep_success) > 0 else 0.0
    print(f"[Eval@{global_step}] return: {avg_return:.2f}, success: {avg_success:.2f}")
    if logger is not None:
        logger.add_scalar("eval/avg_return", avg_return, global_step)
        logger.add_scalar("eval/success_once", avg_success, global_step)
    actor.train()
    eval_envs.close()


def offline_training_phase(
    args,
    actor, qf1, qf2, qf1_target, qf2_target,
    rb,
    actor_optimizer,
    q1_optimizer,
    q2_optimizer,
    a_optimizer,
    log_alpha,
    target_entropy,
    device,
    action_dim,
    action_low,
    action_high,
    logger
):
    from tqdm.auto import tqdm
    total_updates = args.offline_training_steps
    pbar = tqdm(range(total_updates), desc="Offline CQL Training")
    volume = torch.prod(action_high - action_low).item()
    log_uniform_density = np.log(1.0 / volume)
    global_update = 0
    start_time = time.perf_counter()
    for _ in pbar:
        global_update += 1
        data = rb.sample(args.batch_size)
        with torch.no_grad():
            next_a, next_log_pi, _ = actor.get_action(data.next_obs)
            q1_next_tgt = qf1_target(data.next_obs, next_a)
            q2_next_tgt = qf2_target(data.next_obs, next_a)
            min_q_next_tgt = torch.min(q1_next_tgt, q2_next_tgt)
            min_q_next_tgt = min_q_next_tgt - log_alpha.exp() * next_log_pi
            next_q = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_q_next_tgt.view(-1)
        q1_pred = qf1(data.obs, data.actions).view(-1)
        q2_pred = qf2(data.obs, data.actions).view(-1)
        q1_loss = F.mse_loss(q1_pred, next_q)
        q2_loss = F.mse_loss(q2_pred, next_q)
        q1_data_mean = q1_pred.mean()
        q2_data_mean = q2_pred.mean()
        if args.cql_variant.upper() == "H":
            bs = data.obs.shape[0]
            temp = args.cql_temp
            rand_actions = torch.rand(bs, args.cql_num_random, action_dim, device=device)
            rand_actions = action_low + (action_high - action_low) * rand_actions
            obs_expanded = data.obs.unsqueeze(1).expand(bs, args.cql_num_random, -1)
            obs_expanded = obs_expanded.reshape(bs * args.cql_num_random, -1)
            rand_actions_flat = rand_actions.reshape(bs * args.cql_num_random, action_dim)
            q1_rand = qf1(obs_expanded, rand_actions_flat).view(bs, args.cql_num_random)
            q2_rand = qf2(obs_expanded, rand_actions_flat).view(bs, args.cql_num_random)
            q1_rand_lse = torch.logsumexp(q1_rand / temp, dim=1) * temp - log_uniform_density
            q2_rand_lse = torch.logsumexp(q2_rand / temp, dim=1) * temp - log_uniform_density
            cql_penalty_1 = (q1_rand_lse - q1_data_mean).mean()
            cql_penalty_2 = (q2_rand_lse - q2_data_mean).mean()
        else:
            bs = data.obs.shape[0]
            temp = args.cql_temp
            pol_acts = []
            for _ in range(args.cql_num_random):
                a_samp, _, _ = actor.get_action(data.obs)
                pol_acts.append(a_samp.unsqueeze(1))
            pol_acts = torch.cat(pol_acts, dim=1)
            obs_expanded = data.obs.unsqueeze(1).expand(bs, args.cql_num_random, -1)
            obs_expanded = obs_expanded.reshape(bs * args.cql_num_random, -1)
            pol_acts_flat = pol_acts.reshape(bs * args.cql_num_random, action_dim)
            q1_pol = qf1(obs_expanded, pol_acts_flat).view(bs, args.cql_num_random)
            q2_pol = qf2(obs_expanded, pol_acts_flat).view(bs, args.cql_num_random)
            # compute weighted q exp q
            # Shift q values for numerical stability
            # Get max Q values
            q1_max = q1_pol.max(dim=1, keepdim=True)[0]
            q2_max = q2_pol.max(dim=1, keepdim=True)[0]

            max_threshold = 700.0  
            q1_max_clipped = torch.clamp(q1_max, max=max_threshold)
            q2_max_clipped = torch.clamp(q2_max, max=max_threshold)

            # Use clipped values for numerical stability
            q1_pol_lse = ((q1_pol - q1_max_clipped) * torch.exp(q1_pol - q1_max_clipped)).mean(dim=1)
            q2_pol_lse = ((q2_pol - q2_max_clipped) * torch.exp(q2_pol - q2_max_clipped)).mean(dim=1)

            cql_penalty_1 = (q1_pol_lse - q1_data_mean).mean()
            cql_penalty_2 = (q2_pol_lse - q2_data_mean).mean()
        total_q1_loss = q1_loss + args.cql_alpha * cql_penalty_1
        total_q2_loss = q2_loss + args.cql_alpha * cql_penalty_2
        total_loss = total_q1_loss + total_q2_loss
        q1_optimizer.zero_grad()
        q2_optimizer.zero_grad()
        total_loss.backward()
        q1_optimizer.step()
        q2_optimizer.step()
        if global_update % args.policy_frequency == 0:
            pi, log_pi, _ = actor.get_action(data.obs)
            q1_pi = qf1(data.obs, pi)
            q2_pi = qf2(data.obs, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (log_alpha.exp() * log_pi - min_q_pi).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            if args.autotune:
                with torch.no_grad():
                    _, log_pi_samp, _ = actor.get_action(data.obs)
                alpha_loss = - (log_alpha.exp() * (log_pi_samp + target_entropy).detach()).mean()
                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
            else:
                alpha_loss = torch.tensor(0.0, device=device)
        if global_update % args.target_network_frequency == 0:
            for param, targ_param in zip(qf1.parameters(), qf1_target.parameters()):
                targ_param.data.copy_(args.tau * param.data + (1 - args.tau) * targ_param.data)
            for param, targ_param in zip(qf2.parameters(), qf2_target.parameters()):
                targ_param.data.copy_(args.tau * param.data + (1 - args.tau) * targ_param.data)
        if args.eval_freq > 0 and (global_update % args.eval_freq == 0):
            evaluate_offline(args, actor, device, logger, global_update)
        if (global_update % args.log_freq == 0) and (logger is not None):
            logger.add_scalar("losses/qf1_loss", q1_loss.item(), global_update)
            logger.add_scalar("losses/qf2_loss", q2_loss.item(), global_update)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_update)
            logger.add_scalar("losses/total_q1_loss", total_q1_loss.item(), global_update)
            logger.add_scalar("losses/total_q2_loss", total_q2_loss.item(), global_update)
            logger.add_scalar("metrics/q1_data_mean", q1_data_mean.item(), global_update)
            logger.add_scalar("metrics/q2_data_mean", q2_data_mean.item(), global_update)
            logger.add_scalar("metrics/next_q_mean", next_q.mean().item(), global_update)
            logger.add_scalar("metrics/min_q_next_tgt", min_q_next_tgt.mean().item(), global_update)
            logger.add_scalar("metrics/actor_log_pi", log_pi.mean().item(), global_update)
            if args.cql_variant.upper() == "H":
                logger.add_scalar("metrics/q1_rand_lse", q1_rand_lse.mean().item(), global_update)
                logger.add_scalar("metrics/q2_rand_lse", q2_rand_lse.mean().item(), global_update)
            else:
                logger.add_scalar("metrics/q1_pol_lse", q1_pol_lse.mean().item(), global_update)
                logger.add_scalar("metrics/q2_pol_lse", q2_pol_lse.mean().item(), global_update)
            logger.add_scalar("metrics/cql_penalty_1", cql_penalty_1.item(), global_update)
            logger.add_scalar("metrics/cql_penalty_2", cql_penalty_2.item(), global_update)
            logger.add_scalar("metrics/alpha", log_alpha.exp().item(), global_update)
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_update)
        pbar.set_postfix({
            "Q1Loss": f"{q1_loss.item():.4f}",
            "Q2Loss": f"{q2_loss.item():.4f}",
        })
    elapsed = time.perf_counter() - start_time
    print(f"Offline training complete in {elapsed:.1f}s total.")


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = "cql_offline"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend=args.env_vectorization)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    dummy_env = gym.make(args.env_id, num_envs=1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    if isinstance(dummy_env.action_space, gym.spaces.Dict):
        dummy_env = FlattenActionSpaceWrapper(dummy_env)
    rb = ReplayBuffer(dummy_env, num_envs=16, buffer_size=args.buffer_size,
                      storage_device=torch.device(args.buffer_device), sample_device=device)
    if not args.demo_path:
        raise ValueError("No --demo_path provided. This script requires offline data.")
    populate_replay_buffer_from_demo(rb, args.demo_path, device)
    save_replay_buffer(rb, args.save_buffer_path)
    actor = Actor(dummy_env).to(device)
    qf1 = SoftQNetwork(dummy_env).to(device)
    qf2 = SoftQNetwork(dummy_env).to(device)
    qf1_target = SoftQNetwork(dummy_env).to(device)
    qf2_target = SoftQNetwork(dummy_env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q1_optimizer = optim.Adam(list(qf1.parameters()), lr=args.q_lr)
    q2_optimizer = optim.Adam(list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    if args.autotune:
        target_entropy = -torch.prod(torch.tensor(dummy_env.action_space.shape, device=device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        log_alpha = torch.tensor(np.log(args.alpha), device=device, requires_grad=False)
        target_entropy = None
        a_optimizer = None
    action_low = (actor.action_bias - actor.action_scale).detach()
    action_high = (actor.action_bias + actor.action_scale).detach()
    action_dim = action_low.shape[0]

    # --- Setup wandb logging if tracking is enabled ---
    if not args.evaluate and args.track:
        import wandb
        # Build environment config dictionaries explicitly
        env_cfg = env_kwargs.copy()
        env_cfg.update({"num_envs": 16, "env_id": args.env_id})
        if args.control_mode is not None:
            env_cfg["control_mode"] = args.control_mode
        eval_env_cfg = env_kwargs.copy()
        eval_env_cfg.update({"num_envs": args.num_eval_envs, "env_id": args.env_id})
        if args.control_mode is not None:
            eval_env_cfg["control_mode"] = args.control_mode

        config = vars(args).copy()
        config["env_cfg"] = env_cfg
        config["eval_env_cfg"] = eval_env_cfg

        wandb.init(project=args.wandb_project_name,
                   entity=args.wandb_entity,
                   config=config,
                   name=run_name,
                   group=args.wandb_group,
                   save_code=True)
    writer = SummaryWriter(f"{args.save_model_dir}/{run_name}")
    writer.add_text("hyperparameters", "\n".join([f"{k}: {v}" for k, v in vars(args).items()]))
    logger = Logger(log_wandb=args.track, tensorboard=writer)

    offline_training_phase(
        args,
        actor, qf1, qf2,
        qf1_target, qf2_target,
        rb,
        actor_optimizer,
        q1_optimizer,
        q2_optimizer,
        a_optimizer,
        log_alpha,
        target_entropy,
        device,
        action_dim, action_low, action_high,
        logger
    )
    if args.save_model:
        save_dir = f"{args.save_model_dir}/{run_name}"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "final_ckpt.pt")
        torch.save({
            'actor': actor.state_dict(),
            'qf1': qf1_target.state_dict(),
            'qf2': qf2_target.state_dict(),
            'log_alpha': log_alpha,
        }, model_path)
        print(f"Final model saved to {model_path}")
    logger.close()
    writer.close()
