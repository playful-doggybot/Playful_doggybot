
from legged_gym import LEGGED_GYM_ROOT_DIR
from collections import OrderedDict
import os
import json
import time
import numpy as np
np.float = np.float32
import isaacgym
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import update_class_from_dict
from legged_gym.utils.observation import get_obs_slice
from legged_gym.debugger import break_into_debugger


def main(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    for _ in range(50):
        env.step(torch.zeros(1, 12, device= env.device))

    state_dict = torch.load(args.f, map_location="cpu")
    obs = env.get_observations().detach().cpu()
    critic_obs = obs #= env.get_privileged_observations()

    """ The following code is the hacking, which only supports the GRU cases """
    prop_slice = slice(0, 48)
    _prop_slice = slice(48, None)
    config_slice = slice(48+6, 48+6+17)
    _config_slice_0 = slice(48, 48+6)
    _config_slice_1 = slice(48+6+17, None)

    print("actor gru input weight:", state_dict["model_state_dict"]["memory_a.rnn.weight_ih_l0"].shape)
    print("obs_shape:", obs.shape)
    # update bias from clipped weights
    state_dict["model_state_dict"]["memory_a.rnn.bias_ih_l0"] += torch.matmul(
        state_dict["model_state_dict"]["memory_a.rnn.weight_ih_l0"][:, _prop_slice],
        obs[:, _prop_slice].transpose(0, 1),
    )[:, 0]
    # update weight from clipped weights
    state_dict["model_state_dict"]["memory_a.rnn.weight_ih_l0"] = state_dict["model_state_dict"]["memory_a.rnn.weight_ih_l0"][:, prop_slice]

    print("critic gru input weight:", state_dict["model_state_dict"]["memory_c.rnn.weight_ih_l0"].shape)
    print("critic_obs_shape:", critic_obs.shape)
    # update bias from clipped weights
    c_weights = state_dict["model_state_dict"]["memory_c.rnn.weight_ih_l0"]
    state_dict["model_state_dict"]["memory_c.rnn.bias_ih_l0"] += torch.matmul(
        torch.cat([c_weights[:, _config_slice_0], c_weights[:, _config_slice_1]], dim= -1,),
        torch.cat([critic_obs[:, _config_slice_0], obs[:, _config_slice_1]],dim= -1,).transpose(0, 1),
    )[:, 0]
    # update weight from clipped weights
    state_dict["model_state_dict"]["memory_c.rnn.weight_ih_l0"] = torch.cat([
        c_weights[:, prop_slice],
        c_weights[:, config_slice],
    ], dim= -1)

    state_dict.pop("optimizer_state_dict")

    # save the new state dict
    torch.save(state_dict, args.t)

if __name__ == "__main__":
    custom_args = [
        dict(name= "-f", help= "from which logdir", type=str,),
        dict(name= "-t", help= "to which logdir", type=str,),
    ]
    args = get_args(custom_args)
    main(args)