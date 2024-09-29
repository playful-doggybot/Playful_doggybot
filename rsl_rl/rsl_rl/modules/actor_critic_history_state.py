# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size + input_size, hidden_size)

        self.hidden_states = None
    
    # def forward(self, input, masks=None, hidden_states=None):
    #     batch_mode = masks is not None
    #     if batch_mode:
    #         # batch mode (policy update): need saved hidden states
    #         if hidden_states is None:
    #             raise ValueError("Hidden states not passed to memory module during policy update")
    #         out, _ = self.rnn(input, hidden_states)
    #         out = unpad_trajectories(out, masks)
    #     else:
    #         # inference mode (collection): use hidden states of last step
    #         out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
    #     return out
        
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
            input_unpad = unpad_trajectories(input, masks)
            # 将RNN的输出和原始input进行拼接
            out = torch.cat([input_unpad, out], dim=-1)
            # 将拼接后的向量通过线性层
            out = self.linear(out)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
            out = torch.cat([input.unsqueeze(0), out], dim=-1)
            out = self.linear(out.squeeze(0))
            # out = out.squeeze(0)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        if self.hidden_states is None:
            return
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0


class ActorCriticHistory(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        super().__init__(num_actor_obs=rnn_hidden_size ,
                         num_critic_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std,
                         **kwargs,
                        )

        activation = get_activation(activation)

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a} with num layers {rnn_num_layers}")
        print(f"Critic RNN: {self.memory_c} with num layers {rnn_num_layers}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a)

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c)
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states



# import numpy as np

# import torch
# import torch.nn as nn
# from torch.distributions import Normal
# from torch.nn.modules import rnn
# from .actor_critic import ActorCritic, get_activation
# from rsl_rl.utils import unpad_trajectories

# class ActorCriticHistory(ActorCritic):
#     is_recurrent = True
#     def __init__(self,  num_actor_obs,
#                         num_critic_obs,
#                         num_actions,
#                         actor_hidden_dims=[256, 256, 256],
#                         critic_hidden_dims=[256, 256, 256],
#                         activation='elu',
#                         rnn_type='lstm',
#                         rnn_hidden_size=256,
#                         rnn_num_layers=1,
#                         init_noise_std=1.0,
#                         **kwargs):
#         super().__init__(num_actor_obs=rnn_hidden_size + num_actor_obs,
#                          num_critic_obs=rnn_hidden_size + num_critic_obs,
#                          num_actions=num_actions,
#                          actor_hidden_dims=actor_hidden_dims,
#                          critic_hidden_dims=critic_hidden_dims,
#                          activation=activation,
#                          init_noise_std=init_noise_std,
#                          **kwargs,
#                         )

#         activation = get_activation(activation)
#         self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
#         self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

#         print(f"Actor RNN: {self.memory_a} with num layers {rnn_num_layers}")
#         print(f"Critic RNN: {self.memory_c} with num layers {rnn_num_layers}")
        
#         self.rnn_hidden_size = rnn_hidden_size

#     def reset(self, dones=None):
#         self.memory_a.reset(dones)
#         self.memory_c.reset(dones)

#     def act(self, observations, masks=None, hidden_states=None):
#         input_a = self.memory_a(observations, masks, hidden_states)
#         if hidden_states is None:
#             input_hidden = torch.zeros(observations.shape[0], self.rnn_hidden_size, device=observations.device)
#         else:
#             input_hidden = hidden_states.clone()
#         all_input = torch.cat([observations, input_hidden], dim=-1)
#         return super().act(all_input)

#     def act_inference(self, observations):
#         input_a = self.memory_a(observations)
#         return super().act_inference(input_a.squeeze(0))

#     def evaluate(self, critic_observations, masks=None, hidden_states=None):
#         input_c = self.memory_c(critic_observations, masks, hidden_states)
#         if hidden_states is None:
#             input_hidden = torch.zeros(critic_observations.shape[0], self.rnn_hidden_size, device=critic_observations.device)
#         else:
#             input_hidden = hidden_states.clone()
#         all_input = torch.cat([critic_observations, input_hidden], dim=-1)
#         return super().evaluate(all_input)
    
#     def get_hidden_states(self):
#         return self.memory_a.hidden_states, self.memory_c.hidden_states


# class Memory(torch.nn.Module):
#     def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
#         super().__init__()
#         # RNN
#         rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
#         self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
#         self.hidden_states = None
    
#     def forward(self, input, masks=None, hidden_states=None):
#         batch_mode = masks is not None
#         if batch_mode:
#             # batch mode (policy update): need saved hidden states
#             if hidden_states is None:
#                 raise ValueError("Hidden states not passed to memory module during policy update")
#             out, _ = self.rnn(input, hidden_states)
#             out = unpad_trajectories(out, masks)
#         else:
#             # inference mode (collection): use hidden states of last step
#             out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
#         return out

#     def reset(self, dones=None):
#         # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
#         if self.hidden_states is None:
#             return
#         for hidden_state in self.hidden_states:
#             hidden_state[..., dones, :] = 0.0


# import numpy as np

# import torch
# import torch.nn as nn
# from torch.distributions import Normal
# from torch.nn.modules import rnn
# from rsl_rl.utils import unpad_trajectories

# class Memory(torch.nn.Module):
#     def __init__(self, input_size, num_hist = 16, hidden_dims = [128, 64]):
#         super().__init__()
#         # RNN
#         # rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        
#         # self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
#         # self.rnn = nn.Linear(input_size, num_hist)
#         critic_layers = []
#         critic_layers.append(nn.Linear(input_size+num_hist, hidden_dims[0]))
#         for l in range(len(hidden_dims)):
#             if l == len(hidden_dims) - 1:
#                 critic_layers.append(nn.Linear(hidden_dims[l], num_hist))
#             else:
#                 critic_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
#         # critic_layers.append(nn.Linear(hidden_dims[-1], num_hist))

#         self.rnn = nn.Sequential(*critic_layers)
#         # self.linear_layer = nn.Linear(hidden_size, hist_dims)
#         print("Init rnn:", self.rnn)
#         self.hidden_states = None
#         self.hidden_states_last = None
#         self.num_hist = num_hist
    
#     def forward(self, input):
#         if self.hidden_states is None:
#             dim = input.shape[0]
#             self.hidden_states = torch.zeros(dim, self.num_hist, device=input.device)

#         self.hidden_states_last = self.hidden_states.clone()
#         rnn_input = torch.cat([input, self.hidden_states_last], dim=-1)
#         self.hidden_states = self.rnn(rnn_input)
#         return self.hidden_states_last.clone()

#     def reset(self, dones=None):
#         # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
#         if self.hidden_states is None:
#             return
#         self.hidden_states = None

#     def get_hidden_states(self):
#         return self.hidden_states_last

# # class ActorBackbone(nn.Module):
# #     def __init__(self,
# #                  num_actor_obs,
# #                  actor_hidden_dims,
# #                  activation,
# #                 mu_activation, # If set, the last layer will be added with a special activation layer.
# #                 num_actions,
# #                 num_hist,
# #                 **kwargs) -> None:
# #         # Policy
# #         if kwargs:
# #             print("ActorBackbone.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
# #         super(ActorBackbone, self).__init__()

# #         activation = get_activation(activation)
        
# #         mlp_input_dim_a = num_actor_obs + num_hist
        
# #         actor_layers = []
# #         actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
# #         actor_layers.append(activation)
# #         for l in range(len(actor_hidden_dims)):
# #             if l == len(actor_hidden_dims) - 1:
# #                 actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions + num_hist))
# #                 if mu_activation:
# #                     actor_layers.append(get_activation(mu_activation))
# #             else:
# #                 actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
# #                 actor_layers.append(activation)
# #         self.actor = nn.Sequential(*actor_layers)

# #         self.history_encoder = Memory(num_actor_obs, 'gru', num_hist)

# #     def forward(self, obs):
# #         history = self.history_encoder(obs)
# #         backbone_input = torch.cat([obs, history.squeeze()], dim=-1)
# #         backbone_output = self.actor(backbone_input)
# #         return backbone_output



# class ActorCriticHistory(nn.Module):
#     is_recurrent = False
#     is_history_encoder = True
#     def __init__(self,  num_actor_obs,
#                         num_critic_obs,
#                         num_actions,
#                         actor_hidden_dims=[256, 256, 256],
#                         critic_hidden_dims=[256, 256, 256],
#                         activation='elu',
#                         init_noise_std=1.0,
#                         mu_activation= None, # If set, the last layer will be added with a special activation layer.
#                         **kwargs):
#         if kwargs:
#             print("ActorCriticHistory.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
#         super(ActorCriticHistory, self).__init__()

#         activation = get_activation(activation)

#         num_hist = 16

#         # self.actor = ActorBackbone(
#         #         num_actor_obs,
#         #         actor_hidden_dims,
#         #         activation,
#         #         mu_activation,
#         #         num_actions,
#         #         num_hist
#         #     )
#         mlp_input_dim_a = num_actor_obs + num_hist
        
#         actor_layers = []
#         actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
#         actor_layers.append(activation)
#         for l in range(len(actor_hidden_dims)):
#             if l == len(actor_hidden_dims) - 1:
#                 actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions + num_hist))
#                 if mu_activation:
#                     actor_layers.append(get_activation(mu_activation))
#             else:
#                 actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
#                 actor_layers.append(activation)
#         actor_layers.append(nn.Linear(num_actions + num_hist, num_actions))
#         self.actor = nn.Sequential(*actor_layers)

#         gru_hidden_size = 256
#         self.history_encoder = Memory(num_actor_obs, num_hist, hidden_dims = [128])
#         self.linear_layer = nn.Linear(gru_hidden_size, num_hist)
        
#         mlp_input_dim_c = num_critic_obs
#         # Value function
#         critic_layers = []
#         critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
#         critic_layers.append(activation)
#         for l in range(len(critic_hidden_dims)):
#             if l == len(critic_hidden_dims) - 1:
#                 critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
#             else:
#                 critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
#                 critic_layers.append(activation)
#         self.critic = nn.Sequential(*critic_layers)

#         print(f"Actor MLP: {self.actor}")
#         print(f"Critic MLP: {self.critic}")

#         # Action noise
#         self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
#         self.distribution = None
#         # disable args validation for speedup
#         Normal.set_default_validate_args = False
        
#         # seems that we get better performance without init
#         # self.init_memory_weights(self.memory_a, 0.001, 0.)
#         # self.init_memory_weights(self.memory_c, 0.001, 0.)


#     @staticmethod
#     # not used at the moment
#     def init_weights(sequential, scales):
#         [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
#          enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


#     def reset(self, dones=None):
#         pass

#     def forward(self):
#         raise NotImplementedError
    
#     @property
#     def action_mean(self):
#         return self.distribution.mean

#     @property
#     def action_std(self):
#         return self.distribution.stddev
    
#     @property
#     def entropy(self):
#         return self.distribution.entropy().sum(dim=-1)

#     def update_distribution(self, observations):
#         history = self.history_encoder(observations)
#         # history_input = self.linear_layer(history)
#         backbone_input = torch.cat([observations, history], dim=-1)
#         mean = self.actor(backbone_input)
#         self.distribution = Normal(mean, mean*0. + self.std)

#     def act(self, observations, **kwargs):
#         self.update_distribution(observations)
#         return self.distribution.sample()
    
#     def get_actions_log_prob(self, actions):
#         return self.distribution.log_prob(actions).sum(dim=-1)

#     def act_inference(self, observations):
#         history = self.history_encoder(observations)
#         # history_input = self.linear_layer(history)
#         backbone_input = torch.cat([observations, history], dim=-1)
#         actions_mean = self.actor(backbone_input)
#         return actions_mean

#     def evaluate(self, critic_observations, **kwargs):
#         value = self.critic(critic_observations)
#         return value

#     @torch.no_grad()
#     def clip_std(self, min= None, max= None):
#         self.std.copy_(self.std.clip(min= min, max= max))

# def get_activation(act_name):
#     if act_name == "elu":
#         return nn.ELU()
#     elif act_name == "selu":
#         return nn.SELU()
#     elif act_name == "relu":
#         return nn.ReLU()
#     elif act_name == "crelu":
#         return nn.ReLU()
#     elif act_name == "lrelu":
#         return nn.LeakyReLU()
#     elif act_name == "tanh":
#         return nn.Tanh()
#     elif act_name == "sigmoid":
#         return nn.Sigmoid()
#     else:
#         print("invalid activation function!")
#         return None
