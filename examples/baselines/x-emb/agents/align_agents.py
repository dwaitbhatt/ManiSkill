from agents import ActorCriticAgent, SACTransferAgent
from layers import copy_partial_mlp_weights, freeze_mlp_layers, disc_mlp
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from replay_buffer import ReplayBufferSample
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Args

class AgentAligner:
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args, 
                 source_agent: SACTransferAgent,
                 target_agent: SACTransferAgent):
        self.envs = envs
        self.device = device
        self.args = args
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.logging_tracker = {}

        assert self.source_agent.latent_actor.latent_obs_dim == self.target_agent.latent_actor.latent_obs_dim
        assert self.source_agent.latent_actor.latent_action_dim == self.target_agent.latent_actor.latent_action_dim

        # Copy and freeze shared (latent) modules
        self.copy_and_freeze_components(self.target_agent.shared_modules)

        # Identical env obs encoder
        self.target_agent.env_obs_encoder.load_state_dict(self.source_agent.env_obs_encoder.state_dict())
        freeze_mlp_layers(self.target_agent.env_obs_encoder, 0, -1)

        if args.adapted_target_nets:
            # Adapters for robot obs encoder and action encoder/decoder
            copy_partial_mlp_weights(self.source_agent.robot_obs_encoder, self.target_agent.robot_obs_encoder, args.adapter_layers, -1)
            copy_partial_mlp_weights(self.source_agent.act_encoder, self.target_agent.act_encoder, args.adapter_layers, -1)
            copy_partial_mlp_weights(self.source_agent.act_decoder, self.target_agent.act_decoder, 0, -1 - args.adapter_layers)

            if args.only_train_adapters:
                freeze_mlp_layers(self.target_agent.robot_obs_encoder, args.adapter_layers, -1)
                freeze_mlp_layers(self.target_agent.act_encoder, args.adapter_layers, -1)
                freeze_mlp_layers(self.target_agent.act_decoder, 0, -1 - args.adapter_layers)
        
        self.latent_disc = disc_mlp(self.source_agent.latent_actor.latent_obs_dim + self.source_agent.latent_actor.latent_action_dim, 
                                        [args.mlp_dim] * args.num_layers, 
                                        1)
        self.latent_disc_optimizer = torch.optim.Adam(self.latent_disc.parameters(), lr=args.lr)
        self.obs_encoder_optimizer = torch.optim.Adam(self.target_agent.robot_obs_encoder.parameters(), args.lr)
        self.act_encoder_optimizer = torch.optim.Adam(self.target_agent.act_encoder.parameters(), args.lr)
        self.act_decoder_optimizer = torch.optim.Adam(self.target_agent.act_decoder.parameters(), lr=args.lr)

        self.source_agent.freeze_parameters()


    def copy_and_freeze_components(self, target_agent_modules: list[nn.Module]):
        for t_module in target_agent_modules:
            # Handle tensor parameters separately
            if isinstance(t_module, torch.Tensor):
                t_module.data = self.source_agent.__getattribute__(t_module.__name__).data
            else:
                t_module.load_state_dict(self.source_agent.__getattribute__(t_module.__name__).state_dict())
            t_module.requires_grad = False
        

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1)).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(real_samples.shape[0], 1, requires_grad=False, device=self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def get_latent_generator_loss(self, target_data: ReplayBufferSample, global_step: int):
        target_latent_obs = self.target_agent.encode_obs(target_data.obs)
        target_latent_act = self.target_agent.encode_action(target_data.obs, target_data.actions)
        target_input = torch.cat([target_latent_obs, target_latent_act], dim=-1)
        latent_gen_loss = -self.latent_disc(target_input).mean()
        
        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/latent_gen_loss"] = latent_gen_loss.item()
        
        return latent_gen_loss


    def update_discriminator(self, source_data: ReplayBufferSample, target_data: ReplayBufferSample, global_step: int):
        with torch.no_grad():
            source_latent_obs = self.source_agent.encode_obs(source_data.obs)
            source_latent_act = self.source_agent.encode_action(source_data.obs, source_data.actions)
            source_input = torch.cat([source_latent_obs, source_latent_act], dim=-1)

            target_latent_obs = self.target_agent.encode_obs(target_data.obs)
            target_latent_act = self.target_agent.encode_action(target_data.obs, target_data.actions)
            target_input = torch.cat([target_latent_obs, target_latent_act], dim=-1)

        latent_disc_loss = self.latent_disc(target_input).mean() - self.latent_disc(source_input).mean()
        latent_gp = self.compute_gradient_penalty(self.latent_disc, source_input, target_input)
        latent_disc_loss += latent_gp * self.args.lambda_latent_gp

        self.latent_disc_optimizer.zero_grad()
        latent_disc_loss.backward()
        self.latent_disc_optimizer.step()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/latent_disc_loss"] = latent_disc_loss.item()
            self.logging_tracker["losses/latent_gp"] = latent_gp.item()


    def update(self, source_data: ReplayBufferSample, target_data: ReplayBufferSample, global_step: int):
        for _ in range(5):
            self.update_discriminator(source_data, target_data, global_step)
        
        self.obs_encoder_optimizer.zero_grad()
        self.act_encoder_optimizer.zero_grad()
        self.act_decoder_optimizer.zero_grad()

        latent_gen_loss = self.get_latent_generator_loss(target_data, global_step)
        latent_dynamics_loss = self.target_agent.get_latent_dynamics_loss(target_data, global_step)
        total_encoders_loss = latent_gen_loss + self.args.lambda_latent_dynamics_loss * latent_dynamics_loss
        total_encoders_loss.backward()

        self.obs_encoder_optimizer.step()

        act_recon_loss = self.target_agent.get_action_recon_loss(target_data, global_step)
        act_recon_loss.backward()

        self.act_encoder_optimizer.step()
        self.act_decoder_optimizer.step()
