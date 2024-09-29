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


from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Dict
from collections import OrderedDict, defaultdict
from copy import copy

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import get_terrain_cls
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.helpers import euler_from_quaternion
from .legged_robot_config import LeggedRobotCfg


class LeggedRobotMixCmd(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ 
        Initializes a LeggedRobot object.
        Call create_sim() (which creates, simulation, terrain and environments).

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            sim_device (string): 'cuda' or 'cpu'
            headless (bool): Run without rendering if True
        """
        cfg.terrain.measure_heights = True # force height measurement that have full obs from parent class implementation.
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = getattr(self.cfg.viewer, "debug_viz", False)
        self.init_done = False

        # Parse the config file and initialize the parent class
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
            # self.attach_camera()

        # Initilize pytorch buffers used during training
        self._init_buffers()
        self._prepare_reward_function()

        
            
        self.init_done = True

    ######### property functions #########
    # observations property #
    @property
    def all_obs_components(self):
        components = set(self.cfg.env.obs_components)
        if getattr(self.cfg.env, "privileged_obs_components", None):
            components.update(self.cfg.env.privileged_obs_components)
        return components
    
    @property
    def all_obs_components(self) -> set:
        components = set(self.cfg.env.obs_components)
        if getattr(self.cfg.env, "privileged_obs_components", None):
            components.update(self.cfg.env.privileged_obs_components)
        return components
    
    @property
    def obs_segments(self) -> defaultdict:
        return self.get_obs_segment_from_components(self.cfg.env.obs_components)
    
    @property
    def num_obs(self) -> int:
        """ get this value from self.cfg.env """
        assert "proprioception" in self.cfg.env.obs_components, "missing critical observation component 'proprioception'"
        return self.get_num_obs_from_components(self.cfg.env.obs_components) * self.cfg.env.num_state_chunck
    
    # robot property #
    @property
    def base_lin_vel(self) -> Tensor:
        return quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])

    @property
    def base_ang_vel(self) -> Tensor:
        return quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

    @property
    def projected_gravity(self) -> Tensor:
        return quat_rotate_inverse(self.base_quat, self.gravity_vec)
    
    @property
    def yaw(self) -> Tensor:
        _, _, yaw = euler_from_quaternion(self.base_quat)
        return yaw
    
    @property
    def pitch(self) -> Tensor:
        _, pitch, _ = euler_from_quaternion(self.base_quat)
        return pitch
    
    # target property #
    # @property
    # def goal_pos(self) -> Tensor:
    #     """
    #     If use fake goal position, goal position is different with object position.
    #     """
    #     return self.goal_pos

    @property
    def goal_position_rel(self) -> Tensor:
        return self.goal_pos - self.grasp_point_pos

    @property
    def goal_rel_robot(self) -> Tensor:
        return quat_rotate_inverse(self.grasp_point_quat, self.goal_position_rel)

    @property
    def goal_rel_norm(self) -> Tensor:
        return torch.norm(self.goal_position_rel, dim=-1, keepdim=True)
    
    @property
    def goal_rel_xy_norm(self) -> Tensor:
        return torch.norm(self.goal_position_rel[:,:2], dim=-1, keepdim=True)

    @property
    def normalized_goal_position_rel(self) -> Tensor:
        return self.goal_position_rel / (self.goal_rel_norm + 1e-5)

    @property
    def goal_yaw(self) -> Tensor:
        object_base_rel = self.object_pos - self.root_states[:, :3]
        return torch.atan2(object_base_rel[:, 1], object_base_rel[:, 0])
        # return torch.atan2(self.normalized_goal_position_rel[:, 1], self.normalized_goal_position_rel[:, 0])

    @property
    def gripper_open_dist_tensor(self):
        gripper_open_dist = torch.norm(self.l_finger_pos - self.r_finger_pos, dim=-1) 
        return gripper_open_dist

    @property 
    def gravity_tensor(self):
        mass = self.cfg.asset.density * (4 / 3 * 3.14 * self.cfg.init_state.object_radius**3) + 1.75e-5
        # mass = 1.
        gravity = mass * (-torch.tensor(self.cfg.sim.gravity))
        return gravity
    
    ######### property utility functions ###########
    def update_object_confidence(self) -> Tensor:
        confidence = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        if not self.cfg.commands.remote_control:
            resolution = self.resolution_tensor
            fov_width, fov_height = self.fov_width, self.fov_height
            center_x, center_y = self.center_x, self.center_y

            object_coords = quat_rotate_inverse(self.realsense_quat, self.goal_pos- self.realsense_pos) 
            
            object_x = object_coords[:,1] / object_coords[:,0]
            object_y = object_coords[:,2] / object_coords[:,0]

            # Calculate the object's coordinates on the camera image
            object_x = (object_x / fov_width) * resolution[0] + center_x
            object_y = (object_y / fov_height) * resolution[1] + center_y

            is_in_view = torch.logical_and(torch.logical_and(0 <= object_x, object_x <= resolution[0]), 
                                        torch.logical_and(0 <= object_y, object_y <= resolution[1]))
            is_in_view = torch.logical_and(is_in_view, (object_coords[:,0] >= 0))
            
            confidence = torch.tensor(is_in_view).float()

            # random wall noise
            env_ids = torch.randint(0, self.num_envs, (int(self.num_envs / 5),), device=self.device)
            confidence[env_ids] = 0.

        return confidence
    
    def get_obs_segment_from_components(self, components:list) -> defaultdict:
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict() # dict subclass that remembers the order entries were added
        if "proprioception" in components:
            segments["proprioception"] = (3*3+2*self.num_dof+self.num_actions,)
        if "target" in components:
            segments["target"] = (6,)
        # if "height_measurements" in components:
            # segments["height_measurements"] = (187,)
        if "forward_depth" in components:
            segments["forward_depth"] = (1, *self.cfg.sensor.forward_camera.resolution)
        # if "base_pose" in components:
        #     segments["base_pose"] = (6,) # xyz + rpy
        if "robot_config" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            # robot shape friction
            # CoM (Center of Mass) x, y, z
            # base mass (payload)
            # motor strength for each joint
            segments["robot_config"] = (1 + 3 + 1 + self.cfg.env.num_actions,)

        if "commands" in components:
            segments["commands"] = (self.cfg.commands.num_commands, )
        return segments
    
    def get_num_obs_from_components(self, components):
        obs_segments = self.get_obs_segment_from_components(components)
        num_obs = sum([np.prod(v) for v in obs_segments.values()])
        return num_obs


    ######### logic functions ###########
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # actions[:, -1] = actions[:, -2]
        self.prev_actions = actions.clone()
        self.pre_physics_step(actions)

        # step physics and render each frame
        self.render()
        for dec_i in range(self.cfg.control.decimation):
            # self.actions[:, -2] = -0.03
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) # TODO: set_dof_actuation_force_tensor_indexed and set_dof_position_target_tensor_indexed
            self.apply_force_to_ball()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.post_decimation_step(dec_i)
        
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def pre_physics_step(self, actions):
        self.forward_depth_refreshed = False # incase _get_forward_depth_obs is called multiple times
        self.proprioception_refreshed = False
        self.volume_sample_points_refreshed = False 

        if isinstance(self.cfg.normalization.clip_actions, (tuple, list)):
            self.cfg.normalization.clip_actions = torch.tensor(
                self.cfg.normalization.clip_actions,
                device= self.device,
            )
        if isinstance(getattr(self.cfg.normalization, "clip_actions_low", None), (tuple, list)):
            self.cfg.normalization.clip_actions_low = torch.tensor(
                self.cfg.normalization.clip_actions_low,
                device= self.device
            )
            
        if isinstance(getattr(self.cfg.normalization, "clip_actions_high", None), (tuple, list)):
            self.cfg.normalization.clip_actions_high = torch.tensor(
                self.cfg.normalization.clip_actions_high,
                device= self.device
            )

        # some customized action clip methods to bound the action output
        if getattr(self.cfg.normalization, "clip_actions_method", None) == "hard":
            actions_low = getattr(
                self.cfg.normalization, "clip_actions_low",
                self.dof_pos_limits[:, 0] - self.default_dof_pos,
            )
            actions_high = getattr(
                self.cfg.normalization, "clip_actions_high",
                self.dof_pos_limits[:, 1] - self.default_dof_pos,
            )
            self.actions = torch.clip(actions, actions_low, actions_high)

        else:
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

    def post_decimation_step(self, dec_i):
        # self.last_dof_vel[:] = self.dof_vel[:]

        self.substep_torques[:, dec_i, :] = self.torques
        self.substep_dof_vel[:, dec_i, :] = self.dof_vel
        self.substep_exceed_dof_pos_limits[:, dec_i, :] = (self.dof_pos < self.dof_pos_limits[:, 0]) | (self.dof_pos > self.dof_pos_limits[:, 1])

        self.max_torques = torch.maximum(
            torch.max(torch.abs(self.torques), dim= -1)[0],
            self.max_torques,
        )
        ### The set torque limit is usally smaller than the robot dataset
        self.torque_exceed_count_substep[(torch.abs(self.torques) > self.torque_limits).any(dim= -1)] += 1
        
        ### count how many times in the episode the robot is out of dof pos limit (summing all dofs)
        self.out_of_dof_pos_limit_count_substep += self._reward_dof_pos_limits().int()
        ### or using a1_const.h value to check whether the robot is out of dof pos limit        

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        self.gripper_close_time += (self.goal_rel_norm <= self.cfg.rewards.grasp_norm_threshold).squeeze().float() * self.dt

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_torques[:] = self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_goals()
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 0.04, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        
        if hasattr(self.cfg, "termination"): 
            r, p, y = get_euler_xyz(self.base_quat)
            r[r > np.pi] -= np.pi * 2 # to range (-pi, pi)
            p[p > np.pi] -= np.pi * 2 # to range (-pi, pi)
            z = self.root_states[:, 2] - self.env_origins[:, 2]
            
            if "roll" in self.cfg.termination.termination_terms:
                r_term_buff = torch.abs(r) > self.cfg.termination.roll_kwargs["threshold"]
                self.reset_buf |= r_term_buff
            if "pitch" in self.cfg.termination.termination_terms:
                p_term_buff = torch.abs(p) > self.cfg.termination.pitch_kwargs["threshold"]
                self.reset_buf |= p_term_buff
            if "z_low" in self.cfg.termination.termination_terms:
                z_low_term_buff = z < self.cfg.termination.z_low_kwargs["threshold"]
                self.reset_buf |= z_low_term_buff
            if "z_high" in self.cfg.termination.termination_terms:
                z_high_term_buff = z > self.cfg.termination.z_high_kwargs["threshold"]
                self.reset_buf |= z_high_term_buff
            if "ball_fly_away" in self.cfg.termination.termination_terms:
                ball_term_buff = self.goal_pos[:,2] > self.cfg.termination.ball_fly_away_kwargs["threshold"][1]
                self.reset_buf |= ball_term_buff
                ball_term_buff = self.goal_pos[:,2] < self.cfg.termination.ball_fly_away_kwargs["threshold"][0]
                self.reset_buf |= ball_term_buff
            # if "catch_ball" in self.cfg.termination.termination_terms:
            #     close_buff = (self.gripper_close_time >= self.cfg.termination.catch_ball_kwargs["threshold"])
            #     self.reset_buf |= close_buff
            
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # for curriculum learning, only add count after a whole traj
        self.success_times[env_ids] += (self.gripper_close_time[env_ids] > 1.).float().squeeze()
        self.num_returns_after_curriculum[env_ids] += 1
        curriculum_update_env_ids = (self.num_returns_after_curriculum % self.cfg.init_state.curriculum_length == 0).nonzero(as_tuple=False).flatten()
        
        if self.cfg.commands.curriculum and (self.common_step_counter % self.cfg.init_state.curriculum_length == 0):
        # if self.cfg.commands.curriculum and len(curriculum_update_env_ids) > 0:
            self.update_command_curriculum(env_ids)
            self.num_returns_after_curriculum[curriculum_update_env_ids] = 0.
            self.success_times[curriculum_update_env_ids] = 0.
        
        self._fill_extras(env_ids)

        # reset robot states and npc states
        self._reset_dofs(env_ids)
        
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._reset_buffers(env_ids)
        self._reset_npc_states(env_ids)
        
        # Need to set two actors together, cannot call set_actor_root_state_tensor_indexed twice within one time step.
        actor_idx = torch.cat([
            env_ids * self.all_root_states.shape[0] / self.num_envs,
            env_ids * self.all_root_states.shape[0] / self.num_envs + 1 
        ])
        actor_idx_int32 = actor_idx.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(actor_idx_int32), len(actor_idx_int32))
        
 
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        if not self.headless:
            self.rew_buf_msg = {}
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if not self.headless:
                self.rew_buf_msg[name] = rew
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _get_proprioception_obs(self, privileged= False):
        obs_buf = torch.cat((   self.base_lin_vel * self.obs_scales.lin_vel,
                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                self.projected_gravity,
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.prev_actions,
                                ),dim=-1)
        if (not privileged) and (not getattr(self.cfg.env, "use_lin_vel", True)):
            obs_buf[:, :3] = 0.

        if hasattr(self.cfg.sensor, "proprioception") and getattr(self.cfg.sensor.proprioception, "delay_action_obs", 0): 
            if hasattr(self, "current_proprioception_latency"):
                self.proprioception_buffer = torch.cat(
                    [
                        obs_buf.unsqueeze(dim=1),
                        self.proprioception_buffer[:, :-1, :],
                    ],
                    dim=1
                )
                frame_select = (self.current_proprioception_latency / self.dt).to(int)
                frame_select = torch.minimum(frame_select, torch.tensor(self.cfg.sensor.proprioception.buffer_length-1, device=self.device))
                refresh_mask = ((self.episode_length_buf % int(self.cfg.sensor.proprioception.refresh_duration / self.dt)) == 0)
                env_ids = refresh_mask.nonzero(as_tuple=False).flatten()
                obs_buf[env_ids] = self.proprioception_buffer[refresh_mask, frame_select[refresh_mask]].squeeze(1)

        # Do not delay previous actions
        obs_buf[:, -12:] = self.prev_actions[:, :]
        return obs_buf

    def _get_target_obs(self, privileged= False):
        self.object_confidence = self.update_object_confidence()
        if self.cfg.sensor.realsense.only_update_insight:
            # update self.goal_rel_robot_obs if can see object
            
            env_ids = self.object_confidence.nonzero(as_tuple=False).flatten()
            self.goal_rel_robot_obs[env_ids] = self.goal_rel_robot[env_ids] + (2 * torch.rand_like(self.goal_rel_robot[env_ids]) - 1)*self.cfg.noise.noise_scales.goal # add noise here

            # Depth noise
            mask = ((torch.norm(self.root_states[:, 7:10]) > self.cfg.noise.noise_scales.depth_vel_threshold) and (self.object_confidence > 0.)).squeeze(0)
            env_ids = mask.nonzero(as_tuple=False).flatten()
            self.goal_rel_robot_obs[env_ids, 0] += torch_rand_float(self.cfg.noise.noise_scales.depth[0], self.cfg.noise.noise_scales.depth[1], (len(env_ids),1), device=self.device).squeeze()
        else:
            self.goal_rel_robot_obs[:] = self.goal_rel_robot[:]
            
        obs_buf = torch.cat((
            self.goal_rel_robot_obs,
            # (self.goal_yaw - self.yaw).unsqueeze(1),
            torch.zeros_like(self.goal_pos[:,:1]),
            self.object_confidence.unsqueeze(1),
            self.init_object_pos[:,2].unsqueeze(1),
        ), dim = -1)
        if self.cfg.sensor.realsense.no_absolute_height_hint:
            obs_buf[:,-1] = 0.
        
        # add latency and refresh frequency
        if hasattr(self.cfg.sensor, "realsense") and getattr(self.cfg.sensor.realsense, "delay_action_obs", 0) and (self.cfg.sensor.realsense.refresh_duration > self.dt): 
            if hasattr(self, "current_target_latency"):
                self.target_obs_buffer = torch.cat( # add latest observation into buffer
                    [
                        obs_buf.unsqueeze(1),
                        self.target_obs_buffer[:,:-1,:],
                    ],
                    dim=1)
            refresh_mask = ((self.episode_length_buf % int(self.cfg.sensor.realsense.refresh_duration / self.dt)) == 0)
            frame_select = (self.current_target_latency / self.dt).to(int)
            frame_select = torch.minimum(frame_select, torch.tensor(self.cfg.sensor.realsense.buffer_length-1, device=self.device))
            env_ids = refresh_mask.nonzero(as_tuple=False).flatten()
            self.target_obs_last[env_ids] = self.target_obs_buffer[refresh_mask, frame_select[refresh_mask]]
            
            obs_buf = self.target_obs_last.clone()
        else:
            self.target_obs_last = obs_buf

        if not self.cfg.sensor.realsense.use_absolute_height:
            obs_buf[:, -1] = 0.
        return obs_buf

    def _get_commands_obs(self, privileged = False):
        return self.commands * self.commands_scale
    
    def _get_robot_config_obs(self, privileged= False):
        return self.robot_config_buffer        

    def compute_observations(self):
        """ Computes observations
        """
        # force refresh graphics if needed
        obs = []        
        for k, _ in self.obs_segments.items():
            # get the observation from specific component name
            # such as "_get_proprioception_obs", "_get_forward_depth_obs"
            obs.append(
                getattr(self, "_get_" + k + "_obs")(privileged=False) * \
                getattr(self.obs_scales, k, 1.)
            )
        
        
        obs = torch.cat(obs, dim=1) # list -> tensor
        if self.add_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec
        
        self.obs_history_buf = torch.cat(
            [
                obs.unsqueeze(dim=1),
                torch.reshape(self.obs_history_buf, (self.num_envs, self.cfg.env.num_state_chunck, -1))[:, :-1, :]
            ],
            dim=1
        )
        self.obs_buf = self.obs_history_buf.flatten(start_dim=1)
        
        self.privileged_obs_buf = None
        
        # add simple noise if needed
        

    
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_terrain()
        self._create_envs()

    def _create_sensors(self, env_handle=None, actor_handle= None):
        sensor_handle_dict = dict()

        if not self.headless:    
            sensor_handle_dict["forward_camera"] = self._create_onboard_camera(env_handle, actor_handle)
            sensor_handle_dict["viewer_camera"] = self._create_viewer_camera(env_handle, actor_handle)
            # self.first_view_camera_handle = camera_handle # TODO: simplify

        return sensor_handle_dict

    def _create_onboard_camera(self, env_handle, actor_handle):
        # Only use during play.
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg.sensor.realsense.resolution[0]
        camera_props.height = self.cfg.sensor.realsense.resolution[1]
        camera_props.enable_tensors = True
        camera_props.horizontal_fov = self.cfg.sensor.realsense.horizontal_fov

        first_view_camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)

        position = self.cfg.sensor.realsense.position
        rotation = self.cfg.sensor.realsense.rotation

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(position[0], position[1], position[2])
        local_transform.r = gymapi.Quat.from_euler_zyx(rotation[0], rotation[1], rotation[2])

        self.gym.attach_camera_to_body(
            first_view_camera_handle, 
            env_handle, 
            actor_handle, 
            local_transform, 
            gymapi.FOLLOW_TRANSFORM)

        return first_view_camera_handle
    
    def _create_viewer_camera(self,  env_handle, actor_handle):
        # Only use during play.
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg.sensor.viewer_camera.resolution[0]
        camera_props.height = self.cfg.sensor.viewer_camera.resolution[1]
        camera_props.enable_tensors = True
        camera_props.horizontal_fov = self.cfg.sensor.viewer_camera.horizontal_fov

        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)

        position = self.cfg.sensor.viewer_camera.position
        rotation = self.cfg.sensor.viewer_camera.rotation

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(position[0], position[1], position[2])
        local_transform.r = gymapi.Quat.from_euler_zyx(rotation[0], rotation[1], rotation[2])

        self.gym.attach_camera_to_body(
            camera_handle, 
            env_handle, 
            actor_handle, 
            local_transform, 
            gymapi.FOLLOW_POSITION)

        return camera_handle

    def set_camera(self, position, lookat):
        """ Set camera position and direction for viewer
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if env_id == 0:
            all_obs_components = self.all_obs_components
            if "robot_config" in all_obs_components:
                all_obs_components
                self.robot_config_buffer = torch.empty(
                    self.num_envs, 1 + 3 + 1 + self.num_actions,
                    dtype= torch.float32,
                    device= self.device,
                )
        
        if hasattr(self, "robot_config_buffer"):
            self.robot_config_buffer[env_id, 0] = props[0].friction
        
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                props["velocity"][i] = self.cfg.asset.joint_max_velocity
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            # allow config to override torque limits
            if hasattr(self.cfg.control, "torque_limits"):
                if not isinstance(self.cfg.control.torque_limits, (tuple, list)):
                    self.torque_limits = torch.ones(self.num_dof, dtype= torch.float, device= self.device, requires_grad= False)
                    self.torque_limits *= self.cfg.control.torque_limits
                else:
                    self.torque_limits = torch.tensor(self.cfg.control.torque_limits, dtype= torch.float, device= self.device, requires_grad= False)
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])

        if self.cfg.domain_rand.randomize_com:
            rng_com_x = self.cfg.domain_rand.com_range.x
            rng_com_y = self.cfg.domain_rand.com_range.y
            rng_com_z = self.cfg.domain_rand.com_range.z
            rand_com = np.random.uniform(
                [rng_com_x[0], rng_com_y[0], rng_com_z[0]],
                [rng_com_x[1], rng_com_y[1], rng_com_z[1]],
                size=(3,),
            )
            props[0].com += gymapi.Vec3(*rand_com)

        if hasattr(self, "robot_config_buffer"):
            self.robot_config_buffer[env_id, 1] = props[0].com.x
            self.robot_config_buffer[env_id, 2] = props[0].com.y
            self.robot_config_buffer[env_id, 3] = props[0].com.z
            self.robot_config_buffer[env_id, 4] = props[0].mass
            self.robot_config_buffer[env_id, 5:5+self.num_actions] = self.motor_strength[env_id] if hasattr(self, "motor_strength") else 1.
        
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        
        # log max power across current env step
        self.max_power_per_timestep = torch.maximum(
            self.max_power_per_timestep,
            torch.max(torch.sum(self.substep_torques * self.substep_dof_vel, dim= -1), dim= -1)[0],
        )

        # if self.cfg.terrain.measure_heights:
        #     self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
            
        if hasattr(self, "actions_history_buffer"):
            resampling_time = getattr(self.cfg.control, "action_delay_resampling_time", self.dt)
            resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_action_delay(resample_env_ids)

        if hasattr(self, "proprioception_buffer"):
            resampling_time = getattr(self.cfg.sensor.proprioception, "latency_resampling_time", self.dt)
            resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_proprioception_latency(resample_env_ids)
        
        if hasattr(self, "target_buffer"):
            resampling_time = getattr(self.cfg.sensor.realsense, "latency_resampling_time", self.dt)
            resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_target_latency(resample_env_ids)


        self.torque_exceed_count_envstep[(torch.abs(self.substep_torques) > self.torque_limits).any(dim= 1).any(dim= 1)] += 1
        
    def _resample_action_delay(self, env_ids):
        self.current_action_delay[env_ids] = torch_rand_float(
            self.cfg.control.action_delay_range[0],
            self.cfg.control.action_delay_range[1],
            (len(env_ids), 1),
            device= self.device,
        ).flatten()

    def _resample_proprioception_latency(self, env_ids):
        self.current_proprioception_latency[env_ids] = torch_rand_float(
            self.cfg.sensor.proprioception.latency_range[0],
            self.cfg.sensor.proprioception.latency_range[1],
            (len(env_ids), 1),
            device= self.device,
        ).flatten()

    def _resample_target_latency(self, env_ids):
        self.current_target_latency[env_ids] = torch_rand_float(
            self.cfg.sensor.realsense.latency_range[0],
            self.cfg.sensor.realsense.latency_range[1],
            (len(env_ids), 1),
            device= self.device,
        ).flatten()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0: # env_ids=[], return to avoid sample error
            return

        m = torch.distributions.categorical.Categorical(torch.tensor(self.cfg.commands.commands_probabilities))
        index = m.sample(sample_shape=torch.Size([len(env_ids)]))
        # walk_mask = env_ids[index.nonzero().squeeze(-1).bool()]
        mask = env_ids[index.bool()]

        self.commands[env_ids, :] = 0.
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges['grasp'][0], self.command_ranges['grasp'][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[mask, 0] = 0.
        self.commands[mask, 1] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(mask), 1), device=self.device).squeeze(1)
        self.commands[mask, 2] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(mask), 1), device=self.device).squeeze(1)
        self.commands[mask, 3] = torch_rand_float(self.command_ranges["yaw"][0], self.command_ranges["yaw"][1], (len(mask), 1), device=self.device).squeeze(1)
        self.commands[mask, 4] = torch_rand_float(self.command_ranges["pitch"][0], self.command_ranges["pitch"][1], (len(mask), 1), device=self.device).squeeze(1)


        # set small commands to zero
        self.commands[env_ids, 0] *= (torch.abs(self.commands[env_ids, 0]) > 0.1)
        self.commands[env_ids, 1] *= (torch.abs(self.commands[env_ids, 1]) > 0.1)
        self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > 0.1)
        self.commands[env_ids, 3] *= (torch.abs(self.commands[env_ids, 3]) > 0.1)
        self.commands[env_ids, 4] *= (torch.abs(self.commands[env_ids, 4]) > 0.1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if hasattr(self, "motor_strength"):
            actions = self.motor_strength * actions
        #pd controller
        if isinstance(self.cfg.control.action_scale, (tuple, list)):
            self.cfg.control.action_scale = torch.tensor(self.cfg.control.action_scale, device= self.sim_device)
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_npc_states(self, env_ids):
        torch.seed()
        z_height = self.cfg.init_state.object_pos_z_curriculum_increment * self.object_height_levels[env_ids]

        self.object_pos[env_ids, 0] = torch_rand_float(
                                                        self.cfg.init_state.object_pos_x[0], 
                                                        self.cfg.init_state.object_pos_x[1], 
                                                        (1,1), device=self.device)
        self.object_pos[env_ids, 1] = torch_rand_float(
                                                        self.cfg.init_state.object_pos_y[0], 
                                                        self.cfg.init_state.object_pos_y[1], 
                                                        (1,1), device=self.device)
        self.object_pos[env_ids, 2] = torch_rand_float(
                                                        self.cfg.init_state.object_pos_z[0], 
                                                        self.cfg.init_state.object_pos_z[1], 
                                                        (1,1), device=self.device)

        self.object_pos[env_ids, 2] += z_height # update object height level by curriculum
        self.object_pos[env_ids, :3] += self.env_origins[env_ids]
        
        self.init_object_pos[env_ids] = self.object_pos[env_ids].clone()
        
        self.object_states[env_ids, 7:13] = torch.zeros_like(self.object_states[env_ids, 7:13])

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if getattr(self.cfg.domain_rand, "init_dof_pos_ratio_range", None) is not None:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
                self.cfg.domain_rand.init_dof_pos_ratio_range[0],
                self.cfg.domain_rand.init_dof_pos_ratio_range[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            )
        else:
            self.dof_pos[env_ids] = self.default_dof_pos
        # self.dof_vel[env_ids] = 0. # history init method
        dof_vel_range = getattr(self.cfg.domain_rand, "init_dof_vel_range", [-3., 3.])
        self.dof_vel[env_ids] = torch.rand_like(self.dof_vel[env_ids]) * abs(dof_vel_range[1] - dof_vel_range[0]) + min(dof_vel_range)

        # Each env has multiple actors. So the actor index is not the same as env_id. But robot actor is always the first.
        dof_idx = env_ids * self.all_root_states.shape[0] / self.num_envs
        dof_idx_int32 = dof_idx.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.all_dof_states),
                                              gymtorch.unwrap_tensor(dof_idx_int32), len(dof_idx_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.custom_origins:  
            if hasattr(self.cfg.domain_rand, "init_base_pos_range"):
                self.root_states[env_ids, 0:1] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["x"], (len(env_ids), 1), device=self.device)
                self.root_states[env_ids, 1:2] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["y"], (len(env_ids), 1), device=self.device)
            else:
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center

        
        # base rotation (roll and pitch)
        if hasattr(self.cfg.domain_rand, "init_base_rot_range"):
            base_roll = torch_rand_float(
                *self.cfg.domain_rand.init_base_rot_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_pitch = torch_rand_float(
                *self.cfg.domain_rand.init_base_rot_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_quat = quat_from_euler_xyz(base_roll, base_pitch, torch.zeros_like(base_roll))
            self.root_states[env_ids, 3:7] = base_quat
        # base velocities
        if getattr(self.cfg.domain_rand, "init_base_vel_range", None) is None:
            base_vel_range = (-0.5, 0.5)
        else:
            base_vel_range = self.cfg.domain_rand.init_base_vel_range
        if isinstance(base_vel_range, (tuple, list)):
            self.root_states[env_ids, 7:13] = torch_rand_float(
                *base_vel_range,
                (len(env_ids), 6),
                device=self.device,
            ) # [7:10]: lin vel, [10:13]: ang vel
        elif isinstance(base_vel_range, dict):
            self.root_states[env_ids, 7:8] = torch_rand_float(
                *base_vel_range["x"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 8:9] = torch_rand_float(
                *base_vel_range["y"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 9:10] = torch_rand_float(
                *base_vel_range["z"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 10:11] = torch_rand_float(
                *base_vel_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 11:12] = torch_rand_float(
                *base_vel_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 12:13] = torch_rand_float(
                *base_vel_range["yaw"],
                (len(env_ids), 1),
                device=self.device,
            )
        else:
            raise NameError(f"Unknown base_vel_range type: {type(base_vel_range)}")
        
        # Each env has multiple actors. So the actor index is not the same as env_id. But robot actor is always the first.
        # actor_idx = env_ids * self.all_root_states.shape[0] / self.num_envs
        # actor_idx_int32 = actor_idx.to(dtype=torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.all_root_states),
        #                                              gymtorch.unwrap_tensor(actor_idx_int32), len(actor_idx_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        move_up, move_down = self._get_terrain_curriculum_move(env_ids)
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _get_terrain_curriculum_move(self, env_ids):
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        return move_up, move_down
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if not self.init_done:
            return

        if hasattr(self.cfg.init_state, "object_pos_z_curriculum_increment") and hasattr(self.cfg.init_state,"object_pos_z_max_curriculum_level"):
            # threshold = self.reward_scales["gripper_close"]
            # rewards = self.episode_sums["gripper_close"][env_ids] # / self.max_episode_length, if close, rewards= gripper_close
            # move_up = rewards > 0.98 * threshold
            # move_down = rewards < 0.95 * threshold
            move_up = self.gripper_close_time[env_ids] > 1.
            move_down = self.gripper_close_time[env_ids] <= 0.05
            # move_up = self.success_times[env_ids] > self.cfg.init_state.curriculum_length * self.cfg.init_state.curriculum_up
            # move_down = self.success_times[env_ids] < self.cfg.init_state.curriculum_length * self.cfg.init_state.curriculum_down
            before = self.object_height_levels[env_ids][:]
            self.object_height_levels[env_ids] += 1 * move_up - 1 * move_down
            max_level = self.cfg.init_state.object_pos_z_max_curriculum_level
            self.object_height_levels[env_ids] = torch.where(self.object_height_levels[env_ids]>= max_level,
                                                             torch.randint_like(self.object_height_levels[env_ids], max_level),
                                                             torch.clip(self.object_height_levels[env_ids], 0))
            print("Updated!", before, self.object_height_levels[env_ids], self.success_times[env_ids])


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(int(self.num_obs / self.cfg.env.num_state_chunck), device=self.device)
        self.add_noise = cfg.noise.add_noise
        
        segment_start_idx = 0
        # write noise for each corresponding component.
        for k, v in self.obs_segments.items():
            segment_length = np.prod(v)
            # write sensor scale to provided noise_vec, e.g. "_write_forward_depth_noise".
            getattr(self, "_write_" + k + "_noise")(noise_vec[segment_start_idx: segment_start_idx + segment_length])
            segment_start_idx += segment_length
        return noise_vec

    def _write_proprioception_noise(self, noise_vec):
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        # noise_vec[9:12] = 0. # commands
        noise_vec[9:9+self.num_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_dof:9+2*self.num_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_dof:9+2*self.num_dof+self.num_actions] = 0. # previous actions

    def _write_target_noise(self, noise_vec):
        # if current goal is object, set noise wrt whether camera can see object.
        noise_vec[:] = self.cfg.noise.noise_scales.goal 
        # noise_vec[:] = 0.

    def _write_commands_noise(self, noise_vec):
        noise_vec[:] = self.cfg.noise.noise_scales.commands

    def _write_robot_config_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "robot_config"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.robot_config * self.cfg.noise.noise_level * self.obs_scales.robot_config

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states.view(self.num_envs, -1, 13)[:, 0, :] # (num_envs, 13)
        self.all_dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state = self.all_dof_states.view(self.num_envs, -1, 2)[:, :self.num_dof, :] # (num_envs, 2)
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., :self.num_dof, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., :self.num_dof, 1]
        self.base_quat = self.root_states[:, 3:7]
        # self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        # self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        # create wapper tensors for target object
        self.object_states = self.all_root_states.view(self.num_envs, -1, 13)[:, 1, :]
        self.object_pos = self.object_states[:, :3]
        self.object_quat = self.object_states[:, 3:7]
        # self.goal_position_rel = self.object_states[:, :3] - self.root_states[:,:3]
        # self.goal_rel_robot = torch.rand_like(self.goal_position_rel, device=self.device)
        # self.goal_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.init_object_pos = torch.zeros_like(self.object_pos)
        self.goal_rel_robot_obs = torch.rand_like(self.object_pos)
        self.goal_rel_robot_obs[:, 2] = 0.001

        self.gripper_close_time = torch.zeros(self.num_envs, device=self.device)
        
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) 
        self.commands_scale = torch.ones(self.cfg.commands.num_commands, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        # self.measured_heights = 0
        self.substep_torques = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_dof_vel = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limits = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.bool, device=self.device, requires_grad=False)
        self.max_power_per_timestep = torch.zeros(self.num_envs, dtype= torch.float32, device= self.device)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            self.desired_dof_pos[i] = self.cfg.init_state.desired_dof_pos[name]
            # found = False
            # for dof_name in self.cfg.control.stiffness.keys():
            #     if dof_name in name:
            #         self.p_gains[i] = self.cfg.control.stiffness[dof_name]
            #         self.d_gains[i] = self.cfg.control.damping[dof_name]
            #         found = True
            # if not found:
            #     self.p_gains[i] = 0.
            #     self.d_gains[i] = 0.
            #     if self.cfg.control.control_type in ["P", "V"]:
            #         print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.desired_dof_pos = self.desired_dof_pos.unsqueeze(0)

        # motor strengths and allow domain rand to change them
        self.motor_strength = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_motor:
            mtr_rng = self.cfg.domain_rand.leg_motor_strength_range
            self.motor_strength = torch_rand_float(
                mtr_rng[0],
                mtr_rng[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
            self.p_gains = torch_rand_float(
                mtr_rng[0],
                mtr_rng[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            ) * self.cfg.control.stiffness['joint']
            self.d_gains = torch_rand_float(
                mtr_rng[0],
                mtr_rng[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            ) * self.cfg.control.damping['joint']

        # sensor tensors
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.all_rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        self.grasp_point_pos = self.all_rigid_body_states[:,self.finger_grasp_point_indice][:,:3]
        self.grasp_point_quat = self.all_rigid_body_states[:,self.finger_grasp_point_indice][:,3:7]
        self.last_grasp_point_pos = torch.zeros_like(self.grasp_point_pos)
        self.l_finger_pos = self.all_rigid_body_states[:,self.finger_indices[0]][:,:3]
        self.r_finger_pos = self.all_rigid_body_states[:,self.finger_indices[1]][:,:3]
        
        self.goal_pos = self.all_rigid_body_states[:, -1][:, :3]
        self.goal_quat = self.all_rigid_body_states[:, -1][:, 3:7]
        
        self.realsense_states = self.all_rigid_body_states[:, self.realsense_indice]
        self.realsense_pos = self.realsense_states[:, :3]
        self.realsense_quat = self.realsense_states[:,3:7]

        fov_rad = torch.tensor(self.cfg.sensor.realsense.fov, device=self.device)
        self.resolution_tensor = torch.tensor(self.cfg.sensor.realsense.resolution, device=self.device)

        self.fov_width = 2 * torch.tan(fov_rad[0] / 2)
        self.fov_height = 2 * torch.tan(fov_rad[1] / 2)
        self.center_x, self.center_y = self.resolution_tensor / 2

        # Next goal position of gripper
        # self.goal_pos = torch.rand_like(self.grasp_point_pos)

        all_obs_components = self.all_obs_components

        if getattr(self.cfg.sensor.realsense, "delay_action_obs", 0):
            self.current_target_latency = torch_rand_float(
                self.cfg.sensor.realsense.latency_range[0],
                self.cfg.sensor.realsense.latency_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()
            self.target_obs_buffer = torch.zeros((self.num_envs, self.cfg.sensor.realsense.buffer_length, 6), device=self.device)
            self.target_obs_last = torch.zeros((self.num_envs, 6), device=self.device)

        if getattr(self.cfg.control, "action_delay", False):
            assert hasattr(self.cfg.control, "action_delay_range") and hasattr(self.cfg.control, "action_delay_resample_time"), "Please specify action_delay_range and action_delay_resample_time in the config file."
            """ Used in pre-physics step """
            self.cfg.control.action_history_buffer_length = int((self.cfg.control.action_delay_range[1] + self.dt) / self.dt)
            self.actions_history_buffer = torch.zeros(
                (
                    self.cfg.control.action_history_buffer_length,
                    self.num_envs,
                    self.num_actions,
                ),
                dtype= torch.float32,
                device= self.device,
            )
            self.current_action_delay = torch_rand_float(
                self.cfg.control.action_delay_range[0],
                self.cfg.control.action_delay_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()
            self.action_delayed_frames = ((self.current_action_delay / self.dt) + 1).to(int)

        if "proprioception" in all_obs_components and hasattr(self.cfg.sensor, "proprioception"):
            """ Adding proprioception delay buffer """
            self.cfg.sensor.proprioception.buffer_length = int((self.cfg.sensor.proprioception.latency_range[1] + self.dt) / self.dt)
            self.proprioception_buffer = torch.zeros(
                (
                    self.num_envs,
                    self.cfg.sensor.proprioception.buffer_length,
                    self.get_num_obs_from_components(["proprioception"]),
                ),
                dtype= torch.float32,
                device= self.device,
            )
            self.current_proprioception_latency = torch_rand_float(
                self.cfg.sensor.proprioception.latency_range[0],
                self.cfg.sensor.proprioception.latency_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()
            self.proprioception_delayed_frames = ((self.current_proprioception_latency / self.dt) + 1).to(int)

        self.max_torques = torch.zeros_like(self.torques[..., 0])
        self.torque_exceed_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the torque exceeds the limit
        self.torque_exceed_count_envstep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of envsteps that the torque exceeds the limit
        self.out_of_dof_pos_limit_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the dof pos exceeds the limit
        
        self.success_times = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.num_returns_after_curriculum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # between every curriculum length

        self.obs_history_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reset_buffers(self, env_ids):
        if getattr(self.cfg.init_state, "zero_actions", False):
            self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.max_power_per_timestep[env_ids] = 0.
        self.gripper_close_time[env_ids] = 0.

        if hasattr(self, "actions_history_buffer"):
            self.actions_history_buffer[:, env_ids] = 0.
            self.action_delayed_frames[env_ids] = self.cfg.control.action_history_buffer_length
        if hasattr(self, "proprioception_buffer"):
            self.proprioception_buffer[env_ids, :, :] = 0.
            self.proprioception_delayed_frames[env_ids] = self.cfg.sensor.proprioception.buffer_length
            
        # self.goal_rel_robot = torch.rand_like(self.goal_position_rel, device=self.device)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_npc(self, env_handle, env_idx):
        """ create additional opponent for each environment such as static objects, random agents
        or turbulance.
        """
        torch.seed()
        npcs = dict()

        moving = False

        box_pose = gymapi.Transform()

        x = torch_rand_float(self.cfg.init_state.object_pos_x[0], self.cfg.init_state.object_pos_x[1], (1,1), device=self.device)
        y = torch_rand_float(self.cfg.init_state.object_pos_y[0], self.cfg.init_state.object_pos_y[1], (1,1), device=self.device)
        z = torch_rand_float(self.cfg.init_state.object_pos_z[0], self.cfg.init_state.object_pos_z[1], (1,1), device=self.device)
        box_pose.p = gymapi.Vec3(*self.env_origins[env_idx].clone()) + gymapi.Vec3(x, y, z)

        ### Task 0: fixed ball ###
        if not moving:
            asset_options = gymapi.AssetOptions()
            # asset_options.fix_base_link = not moving
            asset_options.fix_base_link = False
            asset_options.armature = True
            asset_options.density = self.cfg.asset.density
            asset_options.collapse_fixed_joints = False
            asset_options.disable_gravity = False
            asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity

            npc_asset = self.gym.create_sphere(self.sim, self.cfg.init_state.object_radius, asset_options)

            # Set object friction and rolling friction
            shape_props_asset = self.gym.get_asset_rigid_shape_properties(npc_asset)
            friction_range = self.cfg.domain_rand.npc_friction_range
            friction = torch_rand_float(friction_range[0], friction_range[1], (1,1), device=self.device)
            shape_props_asset[0].friction = friction
            shape_props_asset[0].rolling_friction = 0.1
            self.gym.set_asset_rigid_shape_properties(npc_asset, shape_props_asset)

            npcs["target_object"] = self.gym.create_actor(env_handle, npc_asset, box_pose, "target_object", env_idx,  0)

        #### task1: moving ball ####
        # moving = True
        if moving:
            asset_path = self.cfg.asset.target_object_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            asset_root = os.path.dirname(asset_path)
            asset_file = os.path.basename(asset_path)

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = False
            asset_options.armature = True
            asset_options.density = self.cfg.asset.density
            asset_options.collapse_fixed_joints = False
            asset_options.disable_gravity = False
            asset_options.max_linear_velocity = 200
            asset_options.max_angular_velocity = 200
            asset_options.linear_damping = 0.
            asset_options.angular_damping = 0.
            asset_options.enable_gyroscopic_forces = True
            

            object_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

            box_pose.p += gymapi.Vec3(0,0,0.5)
            
            npcs["target_object"] = self.gym.create_actor(env_handle, object_asset, box_pose, "moving_ball", env_idx,  0)


        # self.gym.set_rigid_body_color(env_handle, npcs["target_object"], 0, gymapi.MESH_VISUAL,
        #                                  gymapi.Vec3(1.,0.,0.))
        return npcs

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        # asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link 
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dof = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.npc_handles = [] # surrounding actors or objects or oppoents in each environment.
        self.actor_handles = []
        self.sensor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            sensor_handle_dict = self._create_sensors(env_handle, actor_handle)
            npc_handle_dict = self._create_npc(env_handle, i)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.sensor_handles.append(sensor_handle_dict)
            self.npc_handles.append(npc_handle_dict)

        ##### Get indices for each link #####
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        front_hip_names = getattr(self.cfg.asset, "front_hip_names", ["FR_hip_joint", "FL_hip_joint"])
        self.front_hip_indices = torch.zeros(len(front_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(front_hip_names):
            self.front_hip_indices[i] = self.dof_names.index(name)

        rear_hip_names = getattr(self.cfg.asset, "rear_hip_names", ["RR_hip_joint", "RL_hip_joint"])
        self.rear_hip_indices = torch.zeros(len(rear_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(rear_hip_names):
            self.rear_hip_indices[i] = self.dof_names.index(name)

        hip_names = front_hip_names + rear_hip_names
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)

        finger_grasp_point_name = "gripper_grasp_point_link"
        self.finger_grasp_point_indice = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], finger_grasp_point_name)

        finger_names = ["l_finger_link", "r_finger_link"]
        self.finger_indices = torch.zeros(len(finger_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(finger_names)):
            self.finger_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], finger_names[i])

        gripper_force_point_name = "gripper_apply_force_point_link"
        self.gripper_force_point_indice = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], gripper_force_point_name)

        realsense_name = "realsense_link"
        self.realsense_indice = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], realsense_name)

    def _create_terrain(self):
        mesh_type = getattr(self.cfg.terrain, "mesh_type", None)
        if mesh_type=='plane':
            self._create_ground_plane()
        else:
            terrain_cls = self.cfg.terrain.selected
            self.terrain = get_terrain_cls(terrain_cls)(self.cfg.terrain, self.num_envs)
            self.terrain.add_terrain_to_sim(self.gym, self.sim, self.device)
            self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if getattr(self.cfg.terrain, "mesh_type", None) is not None:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types] # (num_envs, 3)
            self.object_height_levels = torch.randint(0, self.cfg.init_state.max_init_object_pos_z_level+1, (self.num_envs,), device=self.device)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = copy(self.cfg.normalization.obs_scales)
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def apply_force_to_ball(self):
        pos = self.all_rigid_body_states[:,:,:3]
        pos_tensor = pos.clone()
        force_tensor = torch.zeros_like(pos, device=self.device)

        # apply force to balance gravity
        direction = torch.zeros_like(self.goal_pos)
        direction[:] = self.gravity_tensor
        force_tensor[:, -1] = direction
        
        # apply force to simulate catching ball
        close = (self.goal_rel_norm <= self.cfg.rewards.grasp_norm_threshold).squeeze() 
        close = close.unsqueeze(-1)
        env_ids = close.nonzero(as_tuple=False).flatten()
        direction = self.grasp_point_pos - self.object_pos
        force_tensor[env_ids, 37] = direction[env_ids] * 150. - 2. * self.object_states[env_ids, 7:10]

        open = (self.goal_rel_norm > self.cfg.rewards.grasp_norm_threshold)
        env_ids = open.nonzero(as_tuple=False).flatten()
        sim_rope = torch.zeros_like(self.object_pos)
        sim_rope[:, 2] = 0 #0.24
        direction = self.init_object_pos - self.object_pos + sim_rope
        # force_tensor[env_ids, 37] = direction[env_ids] * 1.3 - 0.1 * self.object_states[env_ids, 7:10]
        force_tensor[env_ids, 37] = direction[env_ids] * 6. - 0.2 * self.object_states[env_ids, 7:10]
        
        res = self.gym.apply_rigid_body_force_at_pos_tensors(self.sim,
                                                    gymtorch.unwrap_tensor(force_tensor), 
                                                    gymtorch.unwrap_tensor(pos_tensor),
                                                    gymapi.CoordinateSpace.ENV_SPACE)

    ##### draw debug vis and the sub functions #####
    def _draw_goals(self):
        radius = self.cfg.rewards.grasp_norm_threshold
        sphere_geom = gymutil.WireframeSphereGeometry(radius, 16, 16, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            x, y, z = self.goal_pos[i]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        radius = self.cfg.rewards.grasp_norm_threshold
        sphere_geom = gymutil.WireframeSphereGeometry(radius, 16, 16, None, color=(0, 1, 1))
        # tmp_goal_pos = quat_rotate(self.grasp_point_quat, self.target_obs_last[:,:3])
        tmp_goal_pos = quat_rotate(self.grasp_point_quat, self.obs_buf[:, 45:48])
        for i in range(self.num_envs):
            # x, y, z = self.goal_pos[i]
            pos = tmp_goal_pos[i] + self.grasp_point_pos[i]
            x, y, z = pos.squeeze()
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
            

    def _draw_debug_vis(self):
        # return_ = super()._draw_debug_vis()
        if hasattr(self, "forward_depth_output"):
            if self.num_envs == 1:
                import matplotlib.pyplot as plt
                forward_depth_np = self.forward_depth_output[0, 0].detach().cpu().numpy() # (H, W)
                plt.imshow(forward_depth_np, cmap= "gray", vmin= 0, vmax= 1)
                plt.pause(0.001)
            else:
                print("LeggedRobotNoisy: More than one robot, stop showing camera image")
        # return return_

    def _fill_extras(self, env_ids):
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["episode"]['rew_frame_' + key] = torch.nanmean(self.episode_sums[key][env_ids] / self.episode_length_buf[env_ids] / self.reward_scales[key])
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.commands.curriculum:
            self.extras["episode"]["object_height_z"] = torch.mean(self.object_pos[env_ids,2].float())
            self.extras["episode"]["object_height_level"] = torch.mean(self.object_height_levels.float())
            if len(env_ids) > 0:
                self.extras["episode"]["object_height_level_max"] = torch.max(self.object_height_levels[env_ids].float())
                self.extras["episode"]["object_height_level_min"] = torch.min(self.object_height_levels[env_ids].float())

                # record success rate for each difficulty
                for difficulty in range(self.cfg.init_state.object_pos_z_max_curriculum_level + 1):
                    height = self.cfg.init_state.object_pos_z[0]+difficulty*0.05
                    self.extras["episode"][f"num_height_{height:0.2f}"] = torch.sum((self.object_height_levels == difficulty)) # num of current difficulty in all envs

                    mask = (self.object_height_levels[env_ids] == difficulty).squeeze()
                    num_difficulty = torch.sum(mask.float())
                    if num_difficulty >= 1.:
                        self.extras["episode"][f"success_rate_in_difficulty_{height:0.2f}"] = torch.sum((self.gripper_close_time[env_ids][mask] > 1.)) / num_difficulty
                    else:
                        self.extras["episode"][f"success_rate_in_difficulty_{height:0.2f}"] = 0.

        # if self.cfg.terrain.curriculum:
        #     self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        #     if len(env_ids) > 0:
        #         self.extras["episode"]["terrain_level_max"] = torch.max(self.terrain_levels[env_ids].float())
        #         self.extras["episode"]["terrain_level_min"] = torch.min(self.terrain_levels[env_ids].float())
        # log power related info
        self.extras["episode"]["max_power_throughout_episode"] = self.max_power_per_timestep[env_ids].max().cpu().item()
        # log running range info
        pos_x = self.root_states[env_ids][:, 0] - self.env_origins[env_ids][:, 0]
        pos_y = self.root_states[env_ids][:, 1] - self.env_origins[env_ids][:, 1]
        pos_z = self.root_states[env_ids][:, 2] - self.env_origins[env_ids][:, 2]
        self.extras["episode"]["max_pos_x"] = torch.max(pos_x).cpu()
        self.extras["episode"]["min_pos_x"] = torch.min(pos_x).cpu()
        self.extras["episode"]["max_pos_y"] = torch.max(pos_y).cpu()
        self.extras["episode"]["min_pos_y"] = torch.min(pos_y).cpu()
        self.extras["episode"]["max_pos_z"] = torch.max(pos_z).cpu()
        self.extras["episode"]["min_pos_z"] = torch.min(pos_z).cpu()

        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # log whether the episode ends by timeout or dead, or by reaching the goal
        self.extras["episode"]["timeout_ratio"] = self.time_out_buf.float().sum() / self.reset_buf.float().sum()
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
        self.extras["episode"]["max_torques"] = self.max_torques[env_ids]
        self.max_torques[env_ids] = 0.
        self.extras["episode"]["torque_exceed_count_substeps_per_envstep"] = self.torque_exceed_count_substep[env_ids] / self.episode_length_buf[env_ids]
        self.torque_exceed_count_substep[env_ids] = 0
        self.extras["episode"]["torque_exceed_count_envstep"] = self.torque_exceed_count_envstep[env_ids]
        self.torque_exceed_count_envstep[env_ids] = 0
        self.extras["episode"]["out_of_dof_pos_limit_count_substep"] = self.out_of_dof_pos_limit_count_substep[env_ids] / self.episode_length_buf[env_ids]
        self.out_of_dof_pos_limit_count_substep[env_ids] = 0

        # self.extras["episode"]["n_obstacle_passed"] = 0.
        with torch.no_grad():
            pos_x = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
            self.extras["episode"]["pos_x"] = pos_x
            # if self.check_BarrierTrack_terrain():
            #     self.extras["episode"]["n_obstacle_passed"] = torch.mean(torch.clip(
            #         torch.div(pos_x, self.terrain.env_block_length, rounding_mode= "floor") - 1,
            #         min= 0.0,
            #     )).cpu()

        

    ############ Reward Functions: Grasp Object ############
            
    def _reward_gripper_open(self):
        rew = (self.goal_rel_norm >= self.cfg.rewards.grasp_norm_threshold).squeeze() * (self.gripper_open_dist_tensor <= 0.04).squeeze()
        rew *= (self.commands[:,0] > 0).squeeze()
        return rew
    
    def _reward_gripper_close(self):
        rew = (self.goal_rel_norm <= self.cfg.rewards.grasp_norm_threshold).squeeze() 
        rew *= (self.commands[:,0] > 0).squeeze()
        rew *= (self.gripper_close_time <= 1.).squeeze()
        return rew
    
    def _reward_tracking_goal_pos(self):
        rew = torch.exp(-self.goal_rel_norm.squeeze() / self.cfg.rewards.tracking_sigma_goal_pos) # not decrease rew whil goal vel disappear
        rew *= (self.goal_rel_norm.squeeze() <= self.cfg.rewards.xy_pos_norm_threshold).squeeze() 
        rew *= (self.commands[:,0] > 0).squeeze()
        return rew

    def _reward_tracking_goal_vel(self):
        cur_vel = self.root_states[:, 7:9]
        goal_vec_norm = self.goal_position_rel[:,[0,2]] / (self.goal_rel_xy_norm + 1e-5)
        rew = torch.minimum(torch.sum(cur_vel* goal_vec_norm, dim=-1), self.commands[:,0]) / (self.commands[:,0]+1e-5) 
        mask = (self.goal_rel_xy_norm <= self.cfg.rewards.xy_vel_norm_threshold).squeeze()
        
        rew[mask] = 1.
        rew *= (self.commands[:,0] > 0).squeeze() # avoid very large negative rewards while commands[0]=0 and vel < 0
        return rew
    
    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.goal_yaw - self.yaw)/self.cfg.rewards.tracking_sigma)
        mask = (self.goal_rel_xy_norm <= self.cfg.rewards.xy_vel_norm_threshold).squeeze()
        rew[mask] = 1.
        rew *= (self.commands[:,0] > 0).squeeze()
        return rew
    
    # def _reward_in_sight(self):
        # rew = self.object_confidence[:]
        # rew *= (self.commands[:,0] > 0).squeeze()
        # return rew
    
    def _reward_tracking_pitch(self):
        ang_pitch_error = torch.square(self.commands[:, 4] - self.pitch)
        rew = torch.exp(-ang_pitch_error/self.cfg.rewards.tracking_sigma)
        rew *= (self.commands[:,4] >= 0.01).squeeze()
        return rew

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, 1:3] - self.base_lin_vel[:, :2]), dim=1)
        rew = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        rew *= (torch.norm(self.commands[:,1:3], dim=-1) >= 0.05).squeeze()
        rew *= (rew >= 0.01).squeeze()
        return rew

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_yaw_error = torch.square(self.commands[:, 3] - self.base_ang_vel[:, 2])
        ang_vel_error = ang_yaw_error
        rew = torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
        # rew *= (self.commands[:, 3] >= 0.1).squeeze()
        rew *= (rew >= 0.01).squeeze()
        return rew
    ##############   Default Reward Functions   ##############

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        rew = torch.square(self.base_lin_vel[:, 2])
        rew *= (self.goal_rel_xy_norm > self.cfg.rewards.xy_vel_norm_threshold).squeeze()        
        return rew
    
    def _reward_base_height(self):
        # Penalize base height away from target
        rew = torch.square(self.root_states[:, 2] - self.cfg.rewards.base_height_target)
        rew *= (self.goal_rel_xy_norm > self.cfg.rewards.xy_vel_norm_threshold).squeeze()  
        return rew
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        rew = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        rew *= (self.goal_rel_xy_norm > self.cfg.rewards.xy_vel_norm_threshold).squeeze()
        return rew
    
    def _reward_ang_vel_z(self):
        # Penalize xy axes base angular velocity
        rew = torch.abs(self.base_ang_vel[:, 2])
        rew *= (self.goal_rel_xy_norm > self.cfg.rewards.xy_vel_norm_threshold).squeeze()
        return rew
    
    def _reward_object_pitch(self):
        # Penalize xy axes base angular velocity
        rew = torch.abs(self.goal_rel_robot[:, 1])
        rew *= (self.goal_rel_xy_norm > self.cfg.rewards.xy_vel_norm_threshold).squeeze()
        return rew
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        rew = torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
        return rew
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        rew = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return rew
    
    def _reward_action_rate_grasp(self):
        # Penalize changes in actions
        rew = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        rew *= (self.goal_rel_norm.squeeze() <= 0.5).squeeze()
        return rew
    
    def _reward_action_rate_boolean(self):
        diff = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        rew = (diff > self.cfg.rewards.action_rate_th).squeeze()
        return rew
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        # TODO: l1_norm, ratio to soft dof vel limit
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_exceed_dof_vel_l1norm(self):
        exceeded_vel = torch.abs(self.substep_dof_vel) - (self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit)
        exceeded_vel[exceeded_vel < 0.] = 0.
        rew = torch.norm(exceeded_vel, p=1, dim=-1).sum(dim=1)
        # rew *= (self.goal_rel_norm.squeeze() <= 0.5).squeeze()
        return rew
    
    def _reward_exceed_dof_vel_ratio_limits(self):
        exceeded_vel = torch.abs(self.substep_dof_vel) / (self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit) - 1.
        rew = exceeded_vel.clip(min=0.).sum(dim=-1).sum(dim=-1)
        # rew *= (self.goal_rel_norm.squeeze() <= 0.5).squeeze()
        return rew

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.desired_dof_pos), dim=1) * (torch.norm(self.commands, p=1, dim=-1) < 0.1) 
    
    def _reward_stand_still_catch(self):
        # Penalize motion at zero commands
        l_contact = torch.norm(self.contact_forces[:, self.finger_indices[0], :], dim=-1)
        r_contact = torch.norm(self.contact_forces[:, self.finger_indices[1], :], dim=-1)
        contact = torch.minimum(l_contact, r_contact)
        close = (self.goal_rel_norm <= self.cfg.rewards.grasp_norm_threshold).squeeze() * (contact >= 0.01).squeeze() * (self.gripper_open_dist_tensor >= 0.0001).squeeze()
        rew = torch.sum(torch.abs(self.dof_pos - self.desired_dof_pos), dim=1).squeeze() * (self.commands[:, 0] > 0.).squeeze()
        rew *= close
        return rew

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
        
    def _reward_alive(self):
        return 1.

    def _reward_legs_energy_substeps(self):
        # (n_envs, n_substeps, n_dofs) 
        # square sum -> (n_envs, n_substeps)
        # mean -> (n_envs,)
        return torch.mean(torch.sum(torch.square(self.substep_torques * self.substep_dof_vel), dim=-1), dim=-1)
    
    def _reward_hip_pos(self):
        rew = torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.desired_dof_pos[:, self.hip_indices]), dim=1)
        return rew 

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.desired_dof_pos), dim=1)
        return dof_error
    
    def _reward_dof_error_front(self):
        dof_error = torch.sum(torch.square(self.dof_pos[:,:6] - self.desired_dof_pos[:,:6]), dim=1)
        return dof_error
    
    def _reward_rear_hip_dof_error(self):
        dof_error_FR = self.dof_pos[:,self.rear_hip_indices[0]] > self.desired_dof_pos[:,self.rear_hip_indices[0]]
        dof_error_FL = self.dof_pos[:,self.rear_hip_indices[1]] < self.desired_dof_pos[:,self.rear_hip_indices[1]]
        dof_error = dof_error_FL + dof_error_FR
        dof_error *= (self.goal_rel_norm.squeeze() <= self.cfg.rewards.xy_pos_norm_threshold*2.).squeeze() 
        return dof_error
    
    def _reward_front_hip_dof_error(self):
        # "FR_hip_joint", "FL_hip_joint"
        dof_error_FR = self.dof_pos[:,self.front_hip_indices[0]] > self.desired_dof_pos[:,self.front_hip_indices[0]]
        dof_error_FL = self.dof_pos[:,self.front_hip_indices[1]] < self.desired_dof_pos[:,self.front_hip_indices[1]]
        dof_error = dof_error_FL + dof_error_FR
        dof_error *= (self.goal_rel_norm.squeeze() <= self.cfg.rewards.xy_pos_norm_threshold*2.).squeeze() 
        return dof_error
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()
    
    def _reward_exceed_torque_limits_i(self):
        """ Indicator function """
        max_torques = torch.abs(self.substep_torques).max(dim= 1)[0]
        exceed_torque_each_dof = max_torques > (self.torque_limits*self.cfg.rewards.soft_torque_limit)
        exceed_torque = exceed_torque_each_dof.any(dim= 1)
        return exceed_torque.to(torch.float32)
    
    def _reward_exceed_torque_limits_square(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - (self.torque_limits*self.cfg.rewards.soft_torque_limit)
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.square(exceeded_torques).sum(dim= 1).sum(dim= 1)
    
    def _reward_exceed_torque_limits_l1norm(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - (self.torque_limits*self.cfg.rewards.soft_torque_limit)
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.norm(exceeded_torques, p= 1, dim= -1).sum(dim= 1)
    
    def _reward_exceed_dof_pos_limits(self):
        return self.substep_exceed_dof_pos_limits.to(torch.float32).sum(dim= -1).mean(dim= -1)

    ###############   Test Functions   ###############

    def test(self):
        actor_idx = torch.tensor([0], device=self.device) 
        actor_idx_int32 = actor_idx.to(dtype=torch.int32)

        self.root_states[:, :3] = self.object_states[:, :3] - (self.grasp_point_pos[:] - self.root_states[:,:3])
        self.root_states[:, 7:10] = torch.tensor([0,0,0], device=self.device)
        self.root_states[:, 10:13] = torch.tensor([0,0,0], device=self.device)
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(actor_idx_int32), len(actor_idx_int32))

        

