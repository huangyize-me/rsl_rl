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
    """Recurrent actor-critic with shared LSTM backbone and MoE head for actor and critic.
    
    Architecture follows Algorithm 1 from the paper:
    Obs -> Encoder -> Shared LSTM (h_t) -> Gating (Weights) & Experts (Outputs) -> Weighted Action/Value
    """

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
        clip_min_std: float = 0.05,
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

        # encoder for policy obs (used for shared LSTM input)
        self.encoder = self._build_encoder(num_actor_obs, encoder_hidden_dims, activation)
        encoder_output_dim = self._encoder_output_dim(num_actor_obs, encoder_hidden_dims)
        
        # critic encoder for privileged observations (for value estimation)
        self.critic_encoder = self._build_encoder(num_critic_obs, encoder_hidden_dims, activation)
        critic_encoder_output_dim = self._encoder_output_dim(num_critic_obs, encoder_hidden_dims)

        # Shared LSTM backbone (Algorithm 1: h_t <- LSTM([l_t, c_t]))
        # Input: encoded policy obs + encoded critic obs (privileged info)
        shared_rnn_input_dim = encoder_output_dim + critic_encoder_output_dim
        self.memory = Memory(shared_rnn_input_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        # shared gating network: g_hat <- softmax(g(h_t))
        self.gating = MLP(rnn_hidden_dim, num_experts, gating_hidden_dims, activation)

        # experts: a_t <- sum(g_hat_i * f_i(h_t))
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

        # store dimensions for inference mode
        self._num_actor_obs = num_actor_obs
        self._encoder_output_dim = encoder_output_dim
        self._critic_encoder_output_dim = critic_encoder_output_dim

        print(f"Encoder (policy): {self.encoder}")
        print(f"Critic encoder (privileged): {self.critic_encoder}")
        print(f"Shared memory (LSTM): {self.memory}")
        print(f"Shared gating: {self.gating}")
        print(f"Actor experts: {self.actor_experts}")
        print(f"Critic experts: {self.critic_experts}")

        # action noise
        self.noise_std_type = noise_std_type
        self.clip_min_std = clip_min_std  # Table IX: clip min std = 0.05
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
        self.memory.reset(dones)

    def act(self, obs, masks=None, hidden_states=None):
        """Forward pass for action sampling during rollout.
        
        Uses shared LSTM with concatenated policy obs and privileged obs as input.
        """
        # Encode policy observations
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        encoded_actor = self.encoder(actor_obs)
        
        # Encode critic (privileged) observations
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        encoded_critic = self.critic_encoder(critic_obs)
        
        # Concatenate for shared LSTM input: l_t = [z_t, e_t, p_t]
        lstm_input = torch.cat([encoded_actor, encoded_critic], dim=-1)
        
        # Shared LSTM: h_t <- LSTM([l_t, c_t])
        h_t = self.memory(lstm_input, masks, hidden_states)
        
        # Only squeeze if sequence length is 1 (rollout phase)
        if h_t.shape[0] == 1:
            h_t = h_t.squeeze(0)
        
        # Cache h_t for value estimation
        self._cached_h_t = h_t
        
        self._update_distribution(h_t)
        return self.distribution.sample()

    def act_inference(self, obs):
        """Forward pass for action inference.
        
        In Oracle Stage 1: use full privileged observations from simulator.
        In Stage 2 (future): use estimator to predict privileged info.
        """
        # Encode policy observations
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        encoded_actor = self.encoder(actor_obs)
        
        # Oracle Stage 1: get privileged observations from simulator (same as training)
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        encoded_critic = self.critic_encoder(critic_obs)
        
        # Concatenate for shared LSTM input
        lstm_input = torch.cat([encoded_actor, encoded_critic], dim=-1)
        
        # Shared LSTM
        h_t = self.memory(lstm_input).squeeze(0)
        
        # MoE forward for action
        action_mean = self._moe_forward(h_t, self.actor_experts)
        return action_mean

    def evaluate(self, obs, masks=None, hidden_states=None):
        """Evaluate value using cached h_t from act() or compute fresh.
        
        During rollout, reuse cached h_t for efficiency.
        During learning, compute fresh with provided hidden states.
        """
        if hasattr(self, '_cached_h_t') and masks is None and hidden_states is None:
            # Reuse cached h_t from act() during rollout
            h_t = self._cached_h_t
        else:
            # Compute fresh during learning phase
            actor_obs = self.get_actor_obs(obs)
            actor_obs = self.actor_obs_normalizer(actor_obs)
            encoded_actor = self.encoder(actor_obs)
            
            critic_obs = self.get_critic_obs(obs)
            critic_obs = self.critic_obs_normalizer(critic_obs)
            encoded_critic = self.critic_encoder(critic_obs)
            
            lstm_input = torch.cat([encoded_actor, encoded_critic], dim=-1)
            h_t = self.memory(lstm_input, masks, hidden_states)
            
            if h_t.shape[0] == 1:
                h_t = h_t.squeeze(0)
        
        values = self._moe_forward(h_t, self.critic_experts)
        return values

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        # Return shared memory hidden states (duplicated for compatibility)
        return self.memory.hidden_states, self.memory.hidden_states

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
        # Use dim=-2 to ensure compatibility with both 2D (num_envs, hidden) and 3D (seq_len, batch, hidden) inputs
        weights = torch.softmax(self.gating(features), dim=-1)
        expert_outputs = torch.stack([expert(features) for expert in experts], dim=-2)
        mixed_output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=-2)
        return mixed_output

    def _update_distribution(self, features):
        mean = self._moe_forward(features, self.actor_experts)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Clip minimum standard deviation (Table IX: clip_min_std = 0.05)
        std = torch.clamp(std, min=self.clip_min_std)
        self.distribution = Normal(mean, std)
