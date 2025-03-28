####################################################################################
# SAC for Cross-Embodiment Transfer
####################################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from .actor_critic import ActorCriticAgent
from replay_buffer import ReplayBuffer, ReplayBufferSample
from utils import Args
from layers import mlp, weight_init, SimNorm

torch.autograd.set_detect_anomaly(True)
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class LatentGaussianActor(nn.Module):
    """
    The actor network for the latent space. Note that this is not a normalized actor, since it
    operates in the latent space. Latent actions are SimNormed.
    """
    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__()
        self.args = args
        self.backbone = mlp(
            args.latent_obs_dim,
            [args.mlp_dim] * (args.num_layers - 1), 
            args.mlp_dim
        )
        self.fc_mean = nn.Linear(args.mlp_dim, args.latent_action_dim)
        self.fc_logstd = nn.Linear(args.mlp_dim, args.latent_action_dim)

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

        latent_action = SimNorm(self.args)(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(1, keepdim=True)

        latent_mean = SimNorm(self.args)(mean)
        return latent_action, log_prob, latent_mean

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        latent_mean = SimNorm(self.args)(mean)
        return latent_mean


class SoftLatentObsQNetwork(nn.Module):
    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__()
        self.net = mlp(
            args.latent_obs_dim + np.prod(envs.single_action_space.shape), 
            [args.mlp_dim] * args.num_layers, 
            1
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


class SACTransferAgent(ActorCriticAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        self.robot_obs_dim = 0
        for v in envs.unwrapped._get_obs_agent().values():
            self.robot_obs_dim += len(v[0])

        self.env_obs_dim = envs.single_observation_space.shape[0] - self.robot_obs_dim
        self.latent_obs_dim = args.latent_robot_obs_dim + args.latent_env_obs_dim
        args.latent_obs_dim = self.latent_obs_dim

        self.latent_dynamics = mlp(
            self.latent_obs_dim + args.latent_action_dim,
            [args.mlp_dim] * args.num_layers,
            self.latent_obs_dim,
            final_act=SimNorm(args)
        ).to(device)
        self.rew_predictor = mlp(
            self.latent_obs_dim + args.latent_action_dim,
            [args.mlp_dim] * args.num_layers,
            1
        ).to(device)
        
        self.robot_obs_encoder = mlp(
            self.robot_obs_dim,
            [args.enc_dim] * (args.num_layers - 1),
            args.latent_robot_obs_dim,
            final_act=SimNorm(args)
        ).to(device)
        self.env_obs_encoder = mlp(
            self.env_obs_dim,
            [args.enc_dim] * (args.num_layers - 1),
            args.latent_env_obs_dim,
            final_act=SimNorm(args)
        ).to(device)
        self.act_encoder = mlp(
            np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape),
            [args.enc_dim] * (args.num_layers - 1),
            args.latent_action_dim,
            final_act=SimNorm(args)
        ).to(device)
        self.act_decoder = mlp(
            self.latent_obs_dim + args.latent_action_dim,
            [args.mlp_dim] * args.num_layers,
            np.prod(envs.single_action_space.shape),
        ).to(device)

        self.latent_optimizers = torch.optim.Adam([
            {'params': self.latent_dynamics.parameters()},
            {'params': self.rew_predictor.parameters()},
            {'params': self.robot_obs_encoder.parameters(), 'lr': args.enc_lr_scale * args.lr},
            {'params': self.env_obs_encoder.parameters(), 'lr': args.enc_lr_scale * args.lr},
            {'params': self.act_encoder.parameters(), 'lr': args.enc_lr_scale * args.lr},
            {'params': self.act_decoder.parameters(), 'lr': args.enc_lr_scale * args.lr},
        ], lr=args.lr)

        super().__init__(envs, device, args, actor_class=LatentGaussianActor, qf_class=SoftLatentObsQNetwork)
        if args.autotune:
            self.target_entropy = -torch.Tensor([args.latent_action_dim]).to(device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.lr)
        else:
            self.alpha = args.alpha

        self.latent_actor: LatentGaussianActor = self.actor
        self.latent_actor_optimizer = self.actor_optimizer

    def initialize_networks(self):
        super().initialize_networks()
        self.latent_dynamics.apply(weight_init)
        self.rew_predictor.apply(weight_init)
        self.robot_obs_encoder.apply(weight_init)
        self.env_obs_encoder.apply(weight_init)
        self.act_encoder.apply(weight_init)
        self.act_decoder.apply(weight_init)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        robot_obs = obs[:, :self.robot_obs_dim]
        env_obs = obs[:, self.robot_obs_dim:]
        latent_robot_obs = self.robot_obs_encoder(robot_obs)
        latent_env_obs = self.env_obs_encoder(env_obs)
        return torch.cat([latent_robot_obs, latent_env_obs], dim=-1)
    
    def encode_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.act_encoder(torch.cat([obs, action], dim=-1))
    
    def decode_action(self, latent_obs: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        return self.act_decoder(torch.cat([latent_obs, latent_action], dim=-1))
    
    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        latent_obs = self.encode_obs(obs)
        latent_action, _, _ = self.latent_actor(latent_obs)
        return self.decode_action(latent_obs, latent_action)
    
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        latent_obs = self.encode_obs(obs)
        latent_action = self.latent_actor.get_eval_action(latent_obs)
        return self.decode_action(latent_obs, latent_action)

    def update_actor(self, data: ReplayBufferSample, global_step: int):
        latent_obs = self.encode_obs(data.obs)
        latent_pi, latent_log_pi, _ = self.latent_actor(latent_obs)
        pi = self.decode_action(latent_obs, latent_pi)
        qf1_pi = self.qf1(latent_obs, pi)
        qf2_pi = self.qf2(latent_obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        latent_actor_loss = ((self.alpha * latent_log_pi) - min_qf_pi).mean()

        self.latent_actor_optimizer.zero_grad()
        latent_actor_loss.backward()
        self.latent_actor_optimizer.step()

        if self.args.autotune:
            with torch.no_grad():
                _, latent_log_pi, _ = self.latent_actor(latent_obs)
            alpha_loss = (-self.log_alpha.exp() * (latent_log_pi + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/actor_loss"] = latent_actor_loss.item()
            self.logging_tracker["losses/alpha"] = self.alpha
            if self.args.autotune:
                self.logging_tracker["losses/alpha_loss"] = alpha_loss.item()
        
    def get_critic_loss(self, data: ReplayBufferSample, global_step: int):
        # Latent projections
        latent_obs = self.encode_obs(data.obs)
        latent_next_obs = self.encode_obs(data.next_obs)

        # update the value networks
        with torch.no_grad():
            latent_next_actions, latent_next_log_pi, _ = self.latent_actor(latent_next_obs)
            next_actions = self.decode_action(latent_next_obs, latent_next_actions)
            qf1_next_target = self.qf1_target(latent_next_obs, next_actions)
            qf2_next_target = self.qf2_target(latent_next_obs, next_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * latent_next_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)
            # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

        qf1_a_values = self.qf1(latent_obs, data.actions).view(-1)
        qf2_a_values = self.qf2(latent_obs, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # self.obs_encoder_optimizer.zero_grad()
        # self.act_decoder_optimizer.zero_grad()
        # self.q_optimizer.zero_grad()
        # qf_loss.backward()
        # self.obs_encoder_optimizer.step()
        # self.act_decoder_optimizer.step()
        # self.q_optimizer.step()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/qf1_values"] = qf1_a_values.mean().item()
            self.logging_tracker["losses/qf2_values"] = qf2_a_values.mean().item()
            self.logging_tracker["losses/qf1_loss"] = qf1_loss.item()
            self.logging_tracker["losses/qf2_loss"] = qf2_loss.item()
            self.logging_tracker["losses/qf_loss"] = qf_loss.item() / 2.0
        
        return qf_loss

    def get_latent_dynamics_loss(self, data: ReplayBufferSample, global_step: int):
        latent_obs = self.encode_obs(data.obs)
        latent_next_obs = self.encode_obs(data.next_obs)
        latent_action = self.encode_action(data.obs, data.actions)
        
        pred_latent_next_obs = self.latent_dynamics(torch.cat([latent_obs, latent_action], dim=-1))
        latent_dynamics_loss = F.mse_loss(pred_latent_next_obs, latent_next_obs)

        # self.obs_encoder_optimizer.zero_grad()
        # self.act_encoder_optimizer.zero_grad()
        # self.latent_dynamics_optimizer.zero_grad()
        # latent_dynamics_loss.backward()
        # self.obs_encoder_optimizer.step()
        # self.act_encoder_optimizer.step()
        # self.latent_dynamics_optimizer.step()
        return latent_dynamics_loss

    def get_rew_predictor_loss(self, data: ReplayBufferSample, global_step: int):
        latent_obs = self.encode_obs(data.obs)
        latent_action = self.encode_action(data.obs, data.actions)
        
        pred_rew = self.rew_predictor(torch.cat([latent_obs, latent_action], dim=-1))
        rew_loss = F.mse_loss(pred_rew, data.rewards[:, None])

        # self.obs_encoder_optimizer.zero_grad()
        # self.act_encoder_optimizer.zero_grad()
        # self.rew_predictor_optimizer.zero_grad()
        # rew_loss.backward()
        # self.obs_encoder_optimizer.step()
        # self.act_encoder_optimizer.step()
        # self.rew_predictor_optimizer.step()
        return rew_loss

    def update(self, rb: ReplayBuffer, global_update: int, global_step: int):
        """
        This function performs a single update of the Latent SAC model. Details of which components are 
        updated by which gradients are as follows:

        - Q networks: gradients from critic_loss
        - Latent dynamics: gradients from latent_dynamics_loss
        - Reward predictor: gradients from rew_predictor_loss
        - Actor: gradients from latent_actor_loss
        - Alpha (if autotuning): gradients from alpha_loss
        - Observation Encoder: gradients from critic_loss, latent_actor_loss, latent_dynamics_loss, rew_predictor_loss
        - Action Encoder: gradients from latent_dynamics_loss, rew_predictor_loss
        - Action Decoder: gradients from latent_actor_loss
        """
        for local_update in range(self.args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(self.args.batch_size)

            self.latent_optimizers.zero_grad()

            self.update_actor(data, global_step)

            critic_loss = self.get_critic_loss(data, global_step)
            latent_dynamics_loss = self.get_latent_dynamics_loss(data, global_step)
            rew_predictor_loss = self.get_rew_predictor_loss(data, global_step)
            total_latent_loss = critic_loss + latent_dynamics_loss + rew_predictor_loss

            self.q_optimizer.zero_grad()
            total_latent_loss.backward()
            self.q_optimizer.step()

            self.latent_optimizers.step()

            self.update_target_networks()

        return global_update
    
    def save_model(self, model_path: str):
        torch.save({
            'latent_actor': self.latent_actor.state_dict(),
            'qf1': self.qf1_target.state_dict(),
            'qf2': self.qf2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'robot_obs_encoder': self.robot_obs_encoder.state_dict(),
            'env_obs_encoder': self.env_obs_encoder.state_dict(),
            'act_encoder': self.act_encoder.state_dict(),
            'act_decoder': self.act_decoder.state_dict(),
            'latent_dynamics': self.latent_dynamics.state_dict(),
            'rew_predictor': self.rew_predictor.state_dict(),
        }, model_path)
        print(f"model saved to {model_path}")

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path)
        self.latent_actor.load_state_dict(checkpoint['latent_actor'])
        self.qf1_target.load_state_dict(checkpoint['qf1'])
        self.qf2_target.load_state_dict(checkpoint['qf2'])
        self.log_alpha = checkpoint['log_alpha']
        self.robot_obs_encoder.load_state_dict(checkpoint['robot_obs_encoder'])
        self.env_obs_encoder.load_state_dict(checkpoint['env_obs_encoder'])
        self.act_encoder.load_state_dict(checkpoint['act_encoder'])
        self.latent_dynamics.load_state_dict(checkpoint['latent_dynamics'])
        self.rew_predictor.load_state_dict(checkpoint['rew_predictor'])
        print(f"model loaded from {model_path}")