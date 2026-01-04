# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP, Memory


class MoeActorCriticRecurrent(nn.Module):
    """Recurrent actor-critic with a shared-gating MoE head for actor and critic."""

    is_recurrent = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        num_experts: int = 6,
        actor_expert_hidden_dims=[256, 128, 128],
        critic_expert_hidden_dims=[256, 128, 128],
        gating_hidden_dims=[128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        encoder_hidden_dims=None,
        **kwargs,
    ):
        # accept legacy hidden dim keys to remain compatible with standard configs
        if "actor_hidden_dims" in kwargs:
            actor_expert_hidden_dims = kwargs.pop("actor_hidden_dims")
        if "critic_hidden_dims" in kwargs:
            critic_expert_hidden_dims = kwargs.pop("critic_hidden_dims")
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "MoeActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys())
            )
        super().__init__()

        if gating_hidden_dims is None or len(gating_hidden_dims) == 0:
            raise ValueError("gating_hidden_dims must contain at least one dimension.")

        # resolve observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = self._resolve_obs_dim(obs, obs_groups["policy"])
        num_critic_obs = self._resolve_obs_dim(obs, obs_groups["critic"])

        # encoders (optional)
        self.actor_encoder = self._build_encoder(num_actor_obs, encoder_hidden_dims, activation)
        self.critic_encoder = self._build_encoder(num_critic_obs, encoder_hidden_dims, activation)
        actor_rnn_input_dim = self._encoder_output_dim(num_actor_obs, encoder_hidden_dims)
        critic_rnn_input_dim = self._encoder_output_dim(num_critic_obs, encoder_hidden_dims)

        # memory blocks
        self.memory_a = Memory(actor_rnn_input_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(critic_rnn_input_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        # shared gating network
        self.gating = MLP(rnn_hidden_dim, num_experts, gating_hidden_dims, activation)

        # experts
        self.actor_experts = nn.ModuleList(
            [MLP(rnn_hidden_dim, num_actions, actor_expert_hidden_dims, activation) for _ in range(num_experts)]
        )
        self.critic_experts = nn.ModuleList(
            [MLP(rnn_hidden_dim, 1, critic_expert_hidden_dims, activation) for _ in range(num_experts)]
        )

        # observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()

        print(f"Actor encoder: {self.actor_encoder}")
        print(f"Actor memory: {self.memory_a}")
        print(f"Shared gating: {self.gating}")
        print(f"Actor experts: {self.actor_experts}")
        print(f"Critic encoder: {self.critic_encoder}")
        print(f"Critic memory: {self.memory_c}")
        print(f"Critic experts: {self.critic_experts}")

        # action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    """
    Properties
    """

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    """
    Core API
    """

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, obs, masks=None, hidden_states=None):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        actor_obs = self.actor_encoder(actor_obs)
        out_mem = self.memory_a(actor_obs, masks, hidden_states).squeeze(0)
        self._update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        actor_obs = self.actor_encoder(actor_obs)
        out_mem = self.memory_a(actor_obs).squeeze(0)
        action_mean = self._moe_forward(out_mem, self.actor_experts)
        return action_mean

    def evaluate(self, obs, masks=None, hidden_states=None):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        critic_obs = self.critic_encoder(critic_obs)
        out_mem = self.memory_c(critic_obs, masks, hidden_states).squeeze(0)
        values = self._moe_forward(out_mem, self.critic_experts)
        return values

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    """
    Helpers
    """

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def _resolve_obs_dim(self, obs, groups):
        dim = 0
        for obs_group in groups:
            assert len(obs[obs_group].shape) == 2, "The MoeActorCriticRecurrent module only supports 1D observations."
            dim += obs[obs_group].shape[-1]
        return dim

    def _build_encoder(self, input_dim, hidden_dims, activation):
        if hidden_dims is None or len(hidden_dims) == 0:
            return nn.Identity()
        hidden_layers = hidden_dims[:-1] if len(hidden_dims) > 1 else [hidden_dims[0]]
        return MLP(input_dim, hidden_dims[-1], hidden_layers, activation)

    def _encoder_output_dim(self, input_dim, hidden_dims):
        if hidden_dims is None or len(hidden_dims) == 0:
            return input_dim
        return hidden_dims[-1]

    def _moe_forward(self, features: torch.Tensor, experts: nn.ModuleList) -> torch.Tensor:
        weights = torch.softmax(self.gating(features), dim=-1)
        expert_outputs = torch.stack([expert(features) for expert in experts], dim=1)
        mixed_output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        return mixed_output

    def _update_distribution(self, features):
        mean = self._moe_forward(features, self.actor_experts)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)
