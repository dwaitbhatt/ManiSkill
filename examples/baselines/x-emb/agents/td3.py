import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from .actor_critic import ActorCriticAgent, NormalizedActor
from replay_buffer import ReplayBuffer, ReplayBufferSample
from utils import Args, Logger
from layers import mlp


# Simple deterministic actor
class DeterministicActor(NormalizedActor):
    def __init__(self, env, args: Args):
        super().__init__(env, args)
        self.net = mlp(
            np.prod(env.single_observation_space.shape), 
            [args.mlp_dim] * args.num_layers,
            np.prod(env.single_action_space.shape),
        )

    def forward(self, x):
        action = self.net(x)
        action = torch.tanh(action)
        return action * self.action_scale + self.action_bias

    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)


class TD3Agent(ActorCriticAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args, actor_class=DeterministicActor)
        self.actor_target = DeterministicActor(envs, args).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            actions = self.actor(obs)
            # Add exploration noise
            noise = torch.randn_like(actions) * self.args.exploration_noise
            actions = (actions + noise).clamp(self.envs.single_action_space.low[0], self.envs.single_action_space.high[0])
        return actions
    
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            actions = self.actor.get_eval_action(obs)
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
        super().update_target_networks()
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def save_model(self, model_path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'qf1': self.qf1_target.state_dict(),
            'qf2': self.qf2_target.state_dict(),
        }, model_path)
        print(f"model saved to {model_path}")
