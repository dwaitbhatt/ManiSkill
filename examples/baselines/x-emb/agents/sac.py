import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from .actor_critic import ActorCriticAgent, NormalizedActor
from replay_buffer import ReplayBufferSample
from utils import Args

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class GaussianActor(NormalizedActor):
    def __init__(self, env):
        super().__init__(env)
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

    def get_gaussian_params(self, x):
        """
        Return the Gaussian policy distribution parameters.

        Returns:
            A tuple containing:
            
            - (unnormalized) mean: The mean of the action distribution before normalization
            - log_std: The log standard deviation of the action distribution, bounded between -5 and 2 for
            numerical stability
        """
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def forward(self, x):
        mean, log_std = self.get_gaussian_params(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action


class SACAgent(ActorCriticAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args, actor_class=GaussianActor)
        
        self.actor: GaussianActor

        # Automatic entropy tuning
        if args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)[0]
    
    def update_critic(self, data: ReplayBufferSample, global_step: int):
        # update the value networks
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor(data.next_obs)
            qf1_next_target = self.qf1_target(data.next_obs, next_state_actions)
            qf2_next_target = self.qf2_target(data.next_obs, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)
            # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

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
        pi, log_pi, _ = self.actor(data.obs)
        qf1_pi = self.qf1(data.obs, pi)
        qf2_pi = self.qf2(data.obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.args.autotune:
            with torch.no_grad():
                _, log_pi, _ = self.actor(data.obs)
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/actor_loss"] = actor_loss.item()
            self.logging_tracker["losses/alpha"] = self.alpha
            if self.args.autotune:
                self.logging_tracker["losses/alpha_loss"] = alpha_loss.item()
    
    def save_model(self, model_path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'qf1': self.qf1_target.state_dict(),
            'qf2': self.qf2_target.state_dict(),
            'log_alpha': self.log_alpha,
        }, model_path)
        print(f"model saved to {model_path}")
    