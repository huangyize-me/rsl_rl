# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class MoeActorCriticRecurrent(nn.Module):
    """Actor-critic with shared MLP backbone and MoE head for actor and critic.
    
    Architecture (moe_v2):
    [ 输入 Obs ] -> [ 归一化层 ] -> [ 共享骨干 (MLP 512-256) ] -> [ 潜在特征 h ]
                                                                     |
                                           +-------------------------+-------------------------+
                                           |                         |                         |
                                   [ 门控网络 Gating ]       [ Actor 专家组 (n) ]      [ Critic 专家组 (n) ]
                                   (计算权重 W1..Wn)         (生成动作 A1..An)         (生成价值 V1..Vn)
                                           |                         |                         |
                                           +------------+------------+-------------------------+
                                                        |
                                            [ 最终输出 = Σ (Wi * Xi) ]
    """

    # 去掉 LSTM 后，设置为 False 以节省 PPO 计算时的序列处理开销
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        num_experts: int = 4,
        actor_expert_hidden_dims=[256, 128],
        critic_expert_hidden_dims=[256, 128],
        gating_hidden_dims=[128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        clip_min_std: float = 0.05,
        encoder_hidden_dims=None,
        # 保留这些参数以兼容旧配置，但不再使用
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
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
            kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "MoeActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys())
            )
        super().__init__()

        if gating_hidden_dims is None or len(gating_hidden_dims) == 0:
            raise ValueError("gating_hidden_dims must contain at least one dimension.")
        
        if encoder_hidden_dims is None or len(encoder_hidden_dims) == 0:
            raise ValueError("encoder_hidden_dims must contain at least one dimension for Shared Backbone.")

        # 存储专家数量，避免硬编码
        self.num_experts = num_experts

        # resolve observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = self._resolve_obs_dim(obs, obs_groups["policy"])
        num_critic_obs = self._resolve_obs_dim(obs, obs_groups["critic"])

        # observation normalization (关键！防止 MLP 骨干网络在训练初期饱和)
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()

        # Shared Backbone: 统一的共享编码器
        # 如果 Actor 和 Critic 观测一致，只需一个 shared_backbone
        # 这里我们合并 actor 和 critic obs 作为输入
        shared_input_dim = num_actor_obs
        self.shared_backbone = self._build_shared_backbone(shared_input_dim, encoder_hidden_dims, activation)
        backbone_output_dim = encoder_hidden_dims[-1]  # 潜在特征 h 的维度

        # 门控网络: g_hat <- softmax(g(h))
        self.gating = MLP(backbone_output_dim, num_experts, gating_hidden_dims, activation)

        # 专家组: 使用 self.num_experts 变量，严禁硬编码
        self.actor_experts = nn.ModuleList(
            [MLP(backbone_output_dim, num_actions, actor_expert_hidden_dims, activation) for _ in range(self.num_experts)]
        )
        self.critic_experts = nn.ModuleList(
            [MLP(backbone_output_dim, 1, critic_expert_hidden_dims, activation) for _ in range(self.num_experts)]
        )

        # store dimensions for inference mode
        self._num_actor_obs = num_actor_obs
        self._num_critic_obs = num_critic_obs
        self._backbone_output_dim = backbone_output_dim

        print(f"[MoE v2] Shared Backbone: {self.shared_backbone}")
        print(f"[MoE v2] Gating Network: {self.gating}")
        print(f"[MoE v2] Actor Experts ({self.num_experts}): {self.actor_experts}")
        print(f"[MoE v2] Critic Experts ({self.num_experts}): {self.critic_experts}")
        print(f"[MoE v2] Actor Obs Normalization: {actor_obs_normalization}")
        print(f"[MoE v2] Critic Obs Normalization: {critic_obs_normalization}")

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
        # 没有 LSTM，不需要重置隐藏状态
        pass

    def act(self, obs, masks=None, hidden_states=None):
        """Forward pass for action sampling during rollout.
        
        Architecture: Obs -> Normalize -> Shared Backbone -> h -> Gating + Experts -> Action
        """
        # 获取并归一化观测
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        
        # 拼接输入到共享骨干网络
        #backbone_input = torch.cat([actor_obs, critic_obs], dim=-1)
        backbone_input = actor_obs
        
        # 共享骨干网络提取潜在特征 h
        h = self.shared_backbone(backbone_input)
        
        # 缓存 h 用于 evaluate
        self._cached_h = h
        
        # MoE forward for action
        self._update_distribution(h)
        return self.distribution.sample()

    def act_inference(self, obs):
        """Forward pass for action inference (deterministic).
        
        In Oracle Stage 1: use full privileged observations from simulator.
        In Stage 2 (future): use estimator to predict privileged info.
        """
        # 获取并归一化观测
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        
        # 拼接输入到共享骨干网络
       # backbone_input = torch.cat([actor_obs, critic_obs], dim=-1)
        backbone_input = actor_obs
        
        # 共享骨干网络提取潜在特征 h
        h = self.shared_backbone(backbone_input)
        
        # MoE forward for action (deterministic)
        action_mean = self._moe_forward(h, self.actor_experts)
        return action_mean

    def evaluate(self, obs, masks=None, hidden_states=None):
        """Evaluate value using cached h from act() or compute fresh.
        
        During rollout, reuse cached h for efficiency.
        During learning, compute fresh.
        """
        if hasattr(self, '_cached_h') and masks is None and hidden_states is None:
            # Reuse cached h from act() during rollout
            h = self._cached_h
        else:
            # Compute fresh during learning phase
            actor_obs = self.get_actor_obs(obs)
            actor_obs = self.actor_obs_normalizer(actor_obs)
            
            critic_obs = self.get_critic_obs(obs)
            critic_obs = self.critic_obs_normalizer(critic_obs)
            
            #backbone_input = torch.cat([actor_obs, critic_obs], dim=-1)
            backbone_input = actor_obs
            h = self.shared_backbone(backbone_input)
        
        values = self._moe_forward(h, self.critic_experts)
        return values

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        # 没有 LSTM，返回 None（保持接口兼容）
        return None, None

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

    def _build_shared_backbone(self, input_dim, hidden_dims, activation):
        """构建共享骨干网络 (MLP 512-256).
        
        Args:
            input_dim: 输入维度（归一化后的观测）
            hidden_dims: 隐藏层维度列表，如 [512, 256]
            activation: 激活函数
        
        Returns:
            nn.Sequential: 共享骨干网络，输出维度为 hidden_dims[-1]
        """
        if len(hidden_dims) == 1:
            # 只有一层
            return MLP(input_dim, hidden_dims[0], [], activation)
        else:
            # 多层 MLP: 输出维度为 hidden_dims[-1]
            hidden_layers = hidden_dims[:-1]
            return MLP(input_dim, hidden_dims[-1], hidden_layers, activation)

    def _moe_forward(self, features: torch.Tensor, experts: nn.ModuleList) -> torch.Tensor:
        """MoE forward pass.
        
        1. 经过门控网络并计算 Softmax
        2. 并行计算所有专家的输出
        3. 加权求和: Σ (Wi * Ai)
        """
        # 1. 门控网络计算权重
        gate_output = self.gating(features)
        weights = torch.softmax(gate_output, dim=-1)  # (batch, n)
        
        # 2. 并行计算所有专家的输出
        expert_outputs = torch.stack([expert(features) for expert in experts], dim=-2)  # (batch, n, output_dim)
        
        # 3. 加权求和: Σ (Wi * Xi)
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
