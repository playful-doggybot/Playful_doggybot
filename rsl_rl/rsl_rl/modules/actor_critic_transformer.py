import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories

class ActorCriticTransformer(ActorCritic):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        transformer_hidden_size=256,
                        transformer_num_layers=1,
                        transformer_nhead = 4,
                        init_noise_std=1.0,
                        num_state_chunck = 4,
                        **kwargs):
        super().__init__(num_actor_obs=transformer_hidden_size*num_state_chunck,
                         num_critic_obs=transformer_hidden_size*num_state_chunck,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std,
                         **kwargs,
                        )

        activation = get_activation(activation)

        self.memory_a = TransformerBlock(num_actor_obs, transformer_hidden_size, transformer_nhead, transformer_num_layers, num_state_chunck)
        self.memory_c = TransformerBlock(num_critic_obs, transformer_hidden_size, transformer_nhead, transformer_num_layers, num_state_chunck)

        print(f"Actor Transformer: {self.memory_a} with num layers {transformer_num_layers}")
        print(f"Critic Transformer: {self.memory_c} with num layers {transformer_num_layers}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, **kwargs):
        input_a = self.memory_a(observations)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, **kwargs):
        input_c = self.memory_c(critic_observations)
        return super().evaluate(input_c.squeeze(0))
    

class TransformerBlock(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, nhead=4, nlayers=3, num_state_chunck = 5):
        super().__init__()

        self.num_state_chunck = num_state_chunck

        # 状态编码器 和 动作解码器
        self.state_encoder = torch.nn.Linear(int(input_dim / num_state_chunck), hidden_dim)

        # Transformer 编码器
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2, dropout=0., batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=nlayers)


    def forward(self, states):
        """
        计算动作概率分布。

        Args:
            states: 状态序列，形状为 (batch_size, seq_len, input_dim)

        Returns:
            动作概率分布，形状为 (batch_size, action_dim)
        """
        states = torch.reshape(states, (states.shape[0], self.num_state_chunck, -1))
        # 编码状态
        encoded_states = self.state_encoder(states)

        # 使用 Transformer 编码状态序列
        encoded_states = self.transformer_encoder(encoded_states)

        encoded_states = torch.reshape(encoded_states, (encoded_states.shape[0], -1))

        return encoded_states
    
    def reset(self, dones=None):
        pass