from agents import ActorCriticAgent, SACTransferAgent
from layers import copy_partial_mlp_weights, freeze_mlp_layers, disc_mlp
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from replay_buffer import ReplayBuffer, ReplayBufferSample
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Args, Logger

class AgentAligner:
    def __init__(self, device: torch.device, args: Args, 
                 source_agent: SACTransferAgent,
                 target_agent: SACTransferAgent):
        self.device = device
        self.args = args
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.logging_tracker = {}

        assert self.source_agent.latent_actor.latent_obs_dim == self.target_agent.latent_actor.latent_obs_dim
        assert self.source_agent.latent_actor.latent_action_dim == self.target_agent.latent_actor.latent_action_dim

        # Copy and freeze shared (latent) modules
        self.copy_and_freeze_components(self.source_agent.shared_modules, self.target_agent.shared_modules)

        # Identical env obs encoder
        self.target_agent.env_obs_encoder.load_state_dict(self.source_agent.env_obs_encoder.state_dict())
        freeze_mlp_layers(self.target_agent.env_obs_encoder, 0, -1)

        if args.adapted_target_nets:
            # Adapters for robot obs encoder and action encoder/decoder
            copy_partial_mlp_weights(self.source_agent.robot_obs_encoder, self.target_agent.robot_obs_encoder, args.adapter_layers, -1)
            copy_partial_mlp_weights(self.source_agent.act_encoder, self.target_agent.act_encoder, args.adapter_layers, -1)
            copy_partial_mlp_weights(self.source_agent.act_decoder.net, self.target_agent.act_decoder.net, 0, -1 - args.adapter_layers)

            if args.only_train_adapters:
                freeze_mlp_layers(self.target_agent.robot_obs_encoder, args.adapter_layers, -1)
                freeze_mlp_layers(self.target_agent.act_encoder, args.adapter_layers, -1)
                freeze_mlp_layers(self.target_agent.act_decoder.net, 0, -1 - args.adapter_layers)
        
        self.latent_disc = disc_mlp(self.args.latent_robot_obs_dim + self.args.latent_action_dim, 
                                        [args.mlp_dim] * args.num_layers, 
                                        1).to(self.device)
        self.latent_disc_optimizer = torch.optim.Adam(self.latent_disc.parameters(), lr=args.lr)
        
        self.source_disc = disc_mlp(self.source_agent.robot_action_dim, 
                                    [args.mlp_dim] * args.num_layers, 
                                    1).to(self.device)
        self.source_disc_optimizer = torch.optim.Adam(self.source_disc.parameters(), lr=args.lr)

        self.target_disc = disc_mlp(self.target_agent.robot_action_dim, 
                                    [args.mlp_dim] * args.num_layers, 
                                    1).to(self.device)
        self.target_disc_optimizer = torch.optim.Adam(self.target_disc.parameters(), lr=args.lr)

        self.obs_encoder_optimizer = torch.optim.Adam(self.target_agent.robot_obs_encoder.parameters(), args.lr)
        self.act_encoder_optimizer = torch.optim.Adam(self.target_agent.act_encoder.parameters(), args.lr)
        self.act_decoder_optimizer = torch.optim.Adam(self.target_agent.act_decoder.parameters(), lr=args.lr)

        self.source_agent.freeze_parameters()


    def copy_and_freeze_components(self, source_agent_modules: list[nn.Module], target_agent_modules: list[nn.Module]):
        for s_module, t_module in zip(source_agent_modules, target_agent_modules):
            # Handle tensor parameters separately
            if isinstance(t_module, torch.Tensor):
                t_module.data = s_module.data
            else:
                state_dict_to_copy = s_module.state_dict()
                if "exclude_from_copy" in t_module.__dict__:
                    for key in t_module.exclude_from_copy:
                        state_dict_to_copy.pop(key)
                t_module.load_state_dict(state_dict_to_copy, strict=False)
            t_module.requires_grad = False
        

    def compute_gradient_penalty(self, D, real_samples, fake_samples) -> torch.Tensor:
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
    

    def pad_obs(self, data: ReplayBufferSample) -> ReplayBufferSample:
        fake_env_obs = torch.zeros((data.obs.shape[0], self.target_agent.env_obs_dim), device=data.obs.device)
        data.obs = torch.cat([data.obs, fake_env_obs], dim=-1)
        data.next_obs = torch.cat([data.next_obs, fake_env_obs], dim=-1)
        return data


    def get_second_robot_generator_loss(self, source_data: ReplayBufferSample, target_data: ReplayBufferSample, 
                                        second_agent_name: str, global_step: int) -> torch.Tensor:
        if second_agent_name == "source":
            first_agent = self.target_agent
            first_robot_data = target_data
            second_agent = self.source_agent
            second_disc = self.source_disc
        elif second_agent_name == "target":
            first_agent = self.source_agent
            first_robot_data = source_data
            second_agent = self.target_agent
            second_disc = self.target_disc
        else:
            raise ValueError(f"Invalid second agent name: {second_agent_name}, must be 'source' or 'target'")
        
        with torch.set_grad_enabled(second_agent_name == "source"):
            # If condition is true, then first agent is target, and we compute their gradients
            # Hence when second agent is source, we train target encoders
            first_latent_act = first_agent.encode_action(first_robot_data.actions)
        
        with torch.set_grad_enabled(second_agent_name == "target"):
            # If condition is true, then second agent is target, and we compute their gradients
            # Hence when second agent is target, we train target decoder
            first_to_second_act = second_agent.decode_action(first_latent_act)
        
        second_gen_loss = -second_disc(first_to_second_act).mean()
        
        if (global_step - self.args.alignment_batch_size) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker[f"losses/{second_agent_name}_gen_loss"] = second_gen_loss.item()
        
        return second_gen_loss
    

    def update_second_robot_discriminator(self, source_data: ReplayBufferSample, target_data: ReplayBufferSample, 
                                          second_agent_name: str, global_step: int):
        if second_agent_name == "source":
            first_agent = self.target_agent
            first_robot_data = target_data
            second_agent = self.source_agent
            second_robot_data = source_data
            second_disc = self.source_disc
            second_disc_optimizer = self.source_disc_optimizer
        elif second_agent_name == "target":
            first_agent = self.source_agent
            first_robot_data = source_data
            second_agent = self.target_agent
            second_robot_data = target_data
            second_disc = self.target_disc
            second_disc_optimizer = self.target_disc_optimizer
        else:
            raise ValueError(f"Invalid second agent name: {second_agent_name}, must be 'source' or 'target'")
        
        with torch.set_grad_enabled(second_agent_name == "source"):
            first_latent_act = first_agent.encode_action(first_robot_data.actions)
            first_to_second_act = second_agent.decode_action(first_latent_act)
            
        second_robot_action = second_robot_data.actions   
        second_disc_loss = second_disc(first_to_second_act).mean() - second_disc(second_robot_action).mean()
        second_gp = self.compute_gradient_penalty(second_disc, second_robot_action, first_to_second_act)
        second_disc_loss += second_gp * self.args.lambda_gp
        
        second_disc_optimizer.zero_grad()
        second_disc_loss.backward()
        second_disc_optimizer.step()
        
        if (global_step - self.args.alignment_batch_size) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker[f"losses/{second_agent_name}_disc_loss"] = second_disc_loss.item()
            self.logging_tracker[f"losses/{second_agent_name}_gp"] = second_gp.item()


    def get_latent_generator_loss(self, target_data: ReplayBufferSample, global_step: int) -> torch.Tensor:
        target_latent_robot_obs = self.target_agent.encode_obs(target_data.obs, skip_env_obs=True)
        target_latent_act = self.target_agent.encode_action(target_data.actions)
        target_input = torch.cat([target_latent_robot_obs, target_latent_act], dim=-1)
        
        latent_gen_loss = -self.latent_disc(target_input).mean()
        
        if (global_step - self.args.alignment_batch_size) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/latent_gen_loss"] = latent_gen_loss.item()
        
        return latent_gen_loss


    def update_latent_discriminator(self, source_data: ReplayBufferSample, target_data: ReplayBufferSample, global_step: int):
        with torch.no_grad():
            source_latent_robot_obs = self.source_agent.encode_obs(source_data.obs, skip_env_obs=True)
            source_latent_act = self.source_agent.encode_action(source_data.actions)
            source_latent_input = torch.cat([source_latent_robot_obs, source_latent_act], dim=-1)

            target_latent_robot_obs = self.target_agent.encode_obs(target_data.obs, skip_env_obs=True)
            target_latent_act = self.target_agent.encode_action(target_data.actions)
            target_latent_input = torch.cat([target_latent_robot_obs, target_latent_act], dim=-1)

        latent_disc_loss = self.latent_disc(target_latent_input).mean() - self.latent_disc(source_latent_input).mean()
        latent_gp = self.compute_gradient_penalty(self.latent_disc, source_latent_input, target_latent_input)
        latent_disc_loss += latent_gp * self.args.lambda_gp

        self.latent_disc_optimizer.zero_grad()
        latent_disc_loss.backward()
        self.latent_disc_optimizer.step()

        if (global_step - self.args.alignment_batch_size) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/latent_disc_loss"] = latent_disc_loss.item()
            self.logging_tracker["losses/latent_gp"] = latent_gp.item()


    def get_action_recon_loss(self, target_data: ReplayBufferSample, global_step: int):
        act_recon_loss = self.target_agent.get_action_recon_loss(target_data, global_step)

        if (global_step - self.args.alignment_batch_size) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/action_recon_loss"] = act_recon_loss.item()

        return act_recon_loss


    def get_action_cycle_consistency_loss(self, source_data: ReplayBufferSample, target_data: ReplayBufferSample, global_step: int):
        target_latent_obs = self.target_agent.encode_obs(target_data.obs)
        target_latent_act = self.target_agent.encode_action(target_data.obs, target_data.actions)

        with torch.no_grad():
            source_constr_act = self.source_agent.decode_action(target_latent_obs, target_latent_act)
            source_latent_obs = self.source_agent.encode_obs(source_data.obs)
            source_latent_constr_act = self.source_agent.encode_action(source_data.obs, source_constr_act)
        
        target_constr_act = self.target_agent.decode_action(source_latent_obs, source_latent_constr_act)

        act_cycle_loss = F.mse_loss(target_constr_act, target_data.actions)

        if (global_step - self.args.alignment_batch_size) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/action_cycle_consistency_loss"] = act_cycle_loss.item()

        return act_cycle_loss


    def update(self, source_data: ReplayBufferSample, target_data: ReplayBufferSample, global_step: int):
        source_data = self.pad_obs(source_data)
        target_data = self.pad_obs(target_data)
        
        for _ in range(self.args.discriminator_update_freq):
            if self.args.use_latent_adversary:
                self.update_latent_discriminator(source_data, target_data, global_step)
            if self.args.use_source_adversary:
                self.update_second_robot_discriminator(source_data, target_data, "source", global_step)
            if self.args.use_target_adversary:
                self.update_second_robot_discriminator(source_data, target_data, "target", global_step)
        self.obs_encoder_optimizer.zero_grad()
        self.act_encoder_optimizer.zero_grad()
        self.act_decoder_optimizer.zero_grad()

        encoder_generator_loss = 0
        if self.args.use_latent_adversary:
            encoder_generator_loss += self.get_latent_generator_loss(target_data, global_step)
        if self.args.use_source_adversary:
            encoder_generator_loss += self.get_second_robot_generator_loss(source_data, target_data, "source", global_step)
        latent_dynamics_loss = self.target_agent.get_latent_dynamics_loss(target_data, global_step)
        total_encoders_loss = encoder_generator_loss + self.args.lambda_latent_dynamics_loss * latent_dynamics_loss
        total_encoders_loss.backward()

        self.obs_encoder_optimizer.step()

        act_recon_loss = self.get_action_recon_loss(target_data, global_step)
        act_cycle_loss = self.get_action_cycle_consistency_loss(source_data, target_data, global_step)
        
        decoder_generator_loss = 0
        if self.args.use_target_adversary:
            decoder_generator_loss += self.get_second_robot_generator_loss(source_data, target_data, "target", global_step)
        total_action_loss = act_recon_loss + act_cycle_loss + decoder_generator_loss
        total_action_loss.backward()

        self.act_encoder_optimizer.step()
        self.act_decoder_optimizer.step()


    def log_losses(self, logger: Logger, global_step: int):
        for k, v in self.logging_tracker.items():
            logger.add_scalar(k, v, global_step)


class JointTrainer(AgentAligner):
    def __init__(self, device: torch.device, args: Args, 
                 source_agent: SACTransferAgent,
                 target_agent: SACTransferAgent):
        self.device = device
        self.args = args
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.logging_tracker = {}

        assert self.source_agent.latent_actor.latent_obs_dim == self.target_agent.latent_actor.latent_obs_dim
        assert self.source_agent.latent_actor.latent_action_dim == self.target_agent.latent_actor.latent_action_dim

        del self.target_agent.env_obs_encoder
        self.target_agent.env_obs_encoder = self.source_agent.env_obs_encoder
        del self.target_agent.latent_actor
        self.target_agent.latent_actor = self.source_agent.latent_actor
        del self.target_agent.log_alpha
        self.target_agent.log_alpha = self.source_agent.log_alpha
        del self.target_agent.latent_forward_dynamics
        self.target_agent.latent_forward_dynamics = self.source_agent.latent_forward_dynamics
        del self.target_agent.latent_inverse_dynamics
        self.target_agent.latent_inverse_dynamics = self.source_agent.latent_inverse_dynamics
        del self.target_agent.rew_predictor
        self.target_agent.rew_predictor = self.source_agent.rew_predictor

        self.target_agent.shared_modules = [
            self.target_agent.env_obs_encoder,
            self.target_agent.latent_actor, self.target_agent.log_alpha, 
            self.target_agent.latent_forward_dynamics, self.target_agent.latent_inverse_dynamics, 
            self.target_agent.rew_predictor
        ]

        # Ensure all shared modules not just same arch and params, but the same objects 
        for s_module, t_module in zip(self.source_agent.shared_modules, self.target_agent.shared_modules):
            assert s_module is t_module, f"Shared modules are not the same object: {s_module} != {t_module}"

        del self.target_agent.obs_encoder_optimizer
        self.target_agent.obs_encoder_optimizer = torch.optim.Adam([
                {'params': self.target_agent.robot_obs_encoder.parameters()},
                {'params': self.target_agent.env_obs_encoder.parameters()},
            ], lr=args.lr * args.enc_lr_scale)
        
        del self.target_agent.latent_actor_optimizer
        self.target_agent.latent_actor_optimizer = torch.optim.Adam(self.target_agent.latent_actor.parameters(), lr=args.lr)
        
        del self.target_agent.a_optimizer
        self.target_agent.alpha = self.target_agent.log_alpha.exp().item()
        self.target_agent.a_optimizer = torch.optim.Adam([self.target_agent.log_alpha], lr=args.lr)
        
        del self.target_agent.latent_dynamics_optimizer
        self.target_agent.latent_dynamics_optimizer = torch.optim.Adam([
                {'params': self.target_agent.latent_forward_dynamics.parameters()},
                {'params': self.target_agent.latent_inverse_dynamics.parameters()},
            ], lr=args.lr)
            
        del self.target_agent.rew_predictor_optimizer
        self.target_agent.rew_predictor_optimizer = torch.optim.Adam(self.target_agent.rew_predictor.parameters(), lr=args.lr)

        if args.adapted_target_nets:
            # Adapters for robot obs encoder and action encoder/decoder
            copy_partial_mlp_weights(self.source_agent.robot_obs_encoder, self.target_agent.robot_obs_encoder, args.adapter_layers, -1)
            copy_partial_mlp_weights(self.source_agent.act_encoder, self.target_agent.act_encoder, args.adapter_layers, -1)
            copy_partial_mlp_weights(self.source_agent.act_decoder.net, self.target_agent.act_decoder.net, 0, -1 - args.adapter_layers)

            if args.only_train_adapters:
                freeze_mlp_layers(self.target_agent.robot_obs_encoder, args.adapter_layers, -1)
                freeze_mlp_layers(self.target_agent.act_encoder, args.adapter_layers, -1)
                freeze_mlp_layers(self.target_agent.act_decoder.net, 0, -1 - args.adapter_layers)
        
    def update(self, source_buffer: ReplayBuffer, target_buffer: ReplayBuffer, global_update: int, global_step: int):
        global_update_target = global_update
        global_update = self.source_agent.update(source_buffer, global_update, global_step)
        global_update = self.target_agent.update(target_buffer, global_update_target, global_step)
        return global_update

    def log_losses(self, logger: Logger, global_step: int):
        self.source_agent.log_losses(logger, global_step)
        self.target_agent.log_losses(logger, global_step)