from typing import Type

import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from utils import Args, Logger
from replay_buffer import ReplayBuffer, ReplayBufferSample
from layers import mlp, weight_init

import numpy as np
import torch.nn as nn
import torch.optim as optim

class SoftQNetwork(nn.Module):
    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__()
        self.net = mlp(
            np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape), 
            [args.mlp_dim] * args.num_layers, 
            1
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


class NormalizedActor(nn.Module):
    """
    Base class for all actors. 
    The network outputs should be between -1 and 1. They should be normalized to the action
    space limits in the forward pass using the `action_scale` and `action_bias` buffers.
    """

    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__()
        self.args = args

        h, l = envs.single_action_space.high, envs.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # These buffers are persistent and will be saved in state_dict()

    def forward(self, x):
        """
        This should return an (unnormalized) action sampled from the policy distribution.
        """
        raise NotImplementedError
    
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        This should return the best (unnormalized) action as per the current policy.
        """
        raise NotImplementedError

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class ActorCriticAgent:
    """
    Base class for all actor-critic agents. 
    For training, inherited agents must implement the `update_actor` and `update_critic` methods.
    For evaluation, inherited agents must implement the `sample_action` method.
    """

    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args, 
                 actor_class: Type[NormalizedActor],
                 qf_class: Type[nn.Module] = SoftQNetwork):
        self.envs = envs
        self.device = device
        self.args = args

        self.actor = actor_class(envs, args).to(device)
        self.actor_target = actor_class(envs, args).to(device)
        
        self.qf1 = qf_class(envs, args).to(device)
        self.qf2 = qf_class(envs, args).to(device)
        self.qf1_target = qf_class(envs, args).to(device)
        self.qf2_target = qf_class(envs, args).to(device)
        
        if args.checkpoint is not None:
            ckpt = torch.load(args.checkpoint)
            self.actor.load_state_dict(ckpt['actor'])
            self.qf1.load_state_dict(ckpt['qf1'])
            self.qf2.load_state_dict(ckpt['qf2'])
        else:
            self.initialize_networks()

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.lr)

        self.logging_tracker = {}

    def initialize_networks(self):
        self.actor.apply(weight_init)
        self.qf1.apply(weight_init)
        self.qf2.apply(weight_init)

    def log_losses(self, logger: Logger, global_step: int):
        for k, v in self.logging_tracker.items():
            logger.add_scalar(k, v, global_step)

    def update_target_networks(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def update(self, rb: ReplayBuffer, global_update: int, global_step: int):
        """
        Update the actor and critic networks as per the algorithm.
        """
        for local_update in range(self.args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(self.args.batch_size)

            self.update_critic(data, global_step)

            # Delayed policy updates
            if global_update % self.args.policy_frequency == 0:
                self.update_actor(data, global_step)
            if global_update % self.args.target_network_frequency == 0:
                self.update_target_networks()
        return global_update

    def update_critic(self, data: ReplayBufferSample):
        raise NotImplementedError
    
    def update_actor(self, data: ReplayBufferSample):
        raise NotImplementedError

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample an action from the actor. This action will be used to collect experience during training.
        """
        raise NotImplementedError
    
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the best action from the actor. This action will be used to evaluate the policy.
        """
        raise NotImplementedError
    
    def save_model(self, model_path: str):
        raise NotImplementedError
    
    def load_model(self, model_path: str):
        raise NotImplementedError
