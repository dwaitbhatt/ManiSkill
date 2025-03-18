import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from replay_buffer import ReplayBufferSample
from utils import Logger
from replay_buffer import ReplayBuffer
from utils import Args


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


# TD3 Actor (deterministic policy)
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
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
        action = self.net(x)
        return action * self.action_scale + self.action_bias

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class ActorCriticAgent:
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        self.envs = envs
        self.device = device
        self.args = args

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    def update_critic(self, data: ReplayBufferSample):
        pass
    
    def update_actor(self, data: ReplayBufferSample):
        pass

    def update_target_networks(self):
        pass

    def update(self, data: ReplayBufferSample):
        pass
    

class TD3Agent(ActorCriticAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)

        self.actor = Actor(envs).to(device)
        self.actor_target = Actor(envs).to(device)
        
        self.qf1 = SoftQNetwork(envs).to(device)
        self.qf2 = SoftQNetwork(envs).to(device)
        self.qf1_target = SoftQNetwork(envs).to(device)
        self.qf2_target = SoftQNetwork(envs).to(device)
        
        if args.checkpoint is not None:
            ckpt = torch.load(args.checkpoint)
            self.actor.load_state_dict(ckpt['actor'])
            self.qf1.load_state_dict(ckpt['qf1'])
            self.qf2.load_state_dict(ckpt['qf2'])

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)

        self.logging_tracker = {}

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            actions = self.actor(obs)
            # Add exploration noise
            noise = torch.randn_like(actions) * self.args.exploration_noise
            actions = (actions + noise).clamp(self.envs.single_action_space.low[0], self.envs.single_action_space.high[0])
        return actions
    
    def update_critic(self, data: ReplayBufferSample, global_step: int):
        # update the value networks
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(data.actions) * self.args.target_policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            )
            
            next_state_actions = (self.actor_target(data.next_obs) + noise).clamp(
                self.envs.single_action_space.low[0], self.envs.single_action_space.high[0]
            )
            
            qf1_next_target = self.qf1_target(data.next_obs, next_state_actions)
            qf2_next_target = self.qf2_target(data.next_obs, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(data.obs, data.actions).view(-1)
        qf2_a_values = self.qf2(data.obs, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/qf1_values"] = qf1_a_values.mean().item()
            self.logging_tracker["losses/qf2_values"] = qf2_a_values.mean().item()
            self.logging_tracker["losses/qf1_loss"] = qf1_loss.item()
            self.logging_tracker["losses/qf2_loss"] = qf2_loss.item()
            self.logging_tracker["losses/qf_loss"] = qf_loss.item() / 2.0

    def update_actor(self, data: ReplayBufferSample, global_step: int):
        actor_loss = -self.qf1(data.obs, self.actor(data.obs)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()    

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/actor_loss"] = actor_loss.item()

    def update_target_networks(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def update(self, rb: ReplayBuffer, global_update: int, global_step: int):
        for local_update in range(self.args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(self.args.batch_size)

            self.update_critic(data, global_step)

            # Delayed policy updates
            if global_update % self.args.policy_frequency == 0:
                self.update_actor(data, global_step)
                self.update_target_networks()
        return global_update

    def log_losses(self, logger: Logger, global_step: int):
        for k, v in self.logging_tracker.items():
            logger.add_scalar(k, v, global_step)

    def save_model(self, run_name: str):
        model_path = f"{self.args.save_model_dir}/{run_name}/final_ckpt.pt"
        torch.save({
            'actor': self.actor.state_dict(),
            'qf1': self.qf1_target.state_dict(),
            'qf2': self.qf2_target.state_dict(),
        }, model_path)
        print(f"model saved to {model_path}")
