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

from legged_gym import LEGGED_GYM_ROOT_DIR
from collections import OrderedDict
import os
import json
import atexit
import subprocess
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

import numpy as np
import torch

from isaacgym import gymtorch, gymapi, gymutil




def create_recording_camera(gym, env_handle,
        resolution= (1920, 1080),
        h_fov= 86,
        actor_to_attach= None,
        transform= None, # related to actor_to_attach
    ):
    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = resolution[0]
    camera_props.height = resolution[1]
    camera_props.horizontal_fov = h_fov
    camera_handle = gym.create_camera_sensor(env_handle, camera_props)
    if actor_to_attach is not None:
        gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_to_attach,
            transform,
            gymapi.FOLLOW_POSITION,
        )
    elif transform is not None:
        gym.set_camera_transform(
            camera_handle,
            env_handle,
            transform,
        )
    return camera_handle

def exit_handler():
    elapsed_time = time.time() - start_time
    if elapsed_time > 12:
        print("Program is exiting. Running bash script...")
        subprocess.call(["bash","/home/duanxin/Documents/Legged_robot/doggy_bots/legged_gym/video.sh", args.task, args.load_run])
        print("Bash script execution completed.")
    else:
        print(f"Program is exiting. Elapsed time is less than {elapsed_time} seconds.")


class PlayNode:
    def __init__(self, args):
        self.args = args
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        self.env_cfg, self.train_cfg = env_cfg, train_cfg
        self.override_env_cfg()
        self.override_train_cfg()

        # prepare environment
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=self.env_cfg)
        env.reset()
        self.env = env

        self.register_keyboard_event()

        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            args=args,
            train_cfg=train_cfg,
            save_cfg= False,
            )
        agent_model = ppo_runner.alg.actor_critic
        policy = ppo_runner.get_inference_policy(device=env.device)
        
        self.train_cfg = train_cfg
        self.ppo_runner = ppo_runner
        self.agent_model = agent_model
        self.policy = policy

        if RECORD_FRAMES:
            transform = gymapi.Transform()
            transform.p = gymapi.Vec3(*env_cfg.viewer.pos)
            transform.r = gymapi.Quat.from_euler_zyx(0., 0., -np.pi/2)
            recording_camera = create_recording_camera(
                env.gym,
                env.envs[0],
                transform= transform,
            )

        logger = Logger(env.dt)
        self.logger = logger

        self.robot_index = 0 # which robot is used for logging
        self.camera_follow_id = 0 # only effective when CAMERA_FOLLOW
        self.stop_state_log = args.plot_time # number of steps before plotting states
        self.img_idx = 0

    def override_env_cfg(self):
        env_cfg = self.env_cfg
        # override some parameters for testing
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
        # env_cfg.env.episode_length_s = 60
        # env_cfg.sim.dt = 0.006

        env_cfg.terrain.terrain_length = 8
        env_cfg.terrain.terrain_width = 8
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.TerrainPerlin_kwargs["zScale"] = 0.01
        # env_cfg.terrain.TerrainPerlin_kwargs["frequency"] = 10

        # env_cfg.viewer.debug_viz = False # in a1_distill, setting this to true will constantly render the egocentric depth view.
        env_cfg.viewer.draw_volume_sample_points = False
        env_cfg.viewer.debug_viz = True

        env_cfg.env.episode_length_s = 20# int(1e10)
        env_cfg.env.episode_length_s = 20#2500
        env_cfg.commands.resampling_time = int(1e16)
        env_cfg.commands.resampling_time = 20#250
        env_cfg.domain_rand.push_robots = False
        
        # env_cfg.sim.physx.num_position_iterations = 32
        # env_cfg.sim.physx.num_velocity_iterations = 1
        
        # env_cfg.env.use_lin_vel = False
        # env_cfg.control.decimation = 10
        # env_cfg.sim.dt = 0.002
        # env_cfg.sim.gravity = [0., 0. ,-6.81]
        # env_cfg.control.stiffness['joint'] = 37.5 
    

        # env_cfg.init_state.object_pos_x = [1.4, 1.4]
        # env_cfg.init_state.object_pos_y = [-0., 0.]
        env_cfg.init_state.object_pos_z = [0.9, 0.9]
        env_cfg.commands.curriculum = False
        env_cfg.init_state.max_init_object_pos_z_level = 0

        # env_cfg.domain_rand.init_dof_pos_ratio_range = [1.0, 1.0]
        # env_cfg.domain_rand.init_base_rot_range = dict(
        #     roll= [-0.0, 0.0],
        #     pitch= [-0.0, 0.0], 
        # )
        env_cfg.domain_rand.com_range.x = [0.1, 0.1]
        env_cfg.domain_rand.com_range.y = [0.05, 0.05]
        env_cfg.domain_rand.com_range.z = [0., 0.]
        env_cfg.domain_rand.friction_range = [0.8, 0.8]
        env_cfg.domain_rand.randomize_motor = True
        env_cfg.domain_rand.leg_motor_strength_range = [1.] * 2
        # env_cfg.domain_rand.randomize_com = True
        # env_cfg.domain_rand.com_range.x = [0., 0.]
        # env_cfg.domain_rand.randomize_base_mass = True
        # env_cfg.domain_rand.added_mass_range = [1.0] * 2
        # env_cfg.domain_rand.friction_range = [1.] * 2
        env_cfg.domain_rand.init_dof_vel_range = [-0., 0.]
        env_cfg.domain_rand.init_base_rot_range = dict(
            roll=[-0., 0.],
            pitch=[-0., 0.],
        )
        env_cfg.domain_rand.init_base_vel_range = dict(
            x=[-0., 0.],
            y=[-0., 0.],
            z=[-0., 0.],
            roll=[-0.,0.],
            pitch=[-0.,0.],
            yaw=[-0.,0.],
        )
        env_cfg.domain_rand.init_base_pos_range = dict(
            x=[-0., 0.],
            y=[-0., 0.],
        )
        # if self.args.no_throw:
            # env_cfg.domain_rand.init_base_vel_range = [0., 0.]
        # elif isinstance(env_cfg.domain_rand.init_base_vel_range, dict):
            # print("init_base_vel_range 'x' value is set to:", env_cfg.domain_rand.init_base_vel_range["x"])
        # else:
            # print("init_base_vel_range not set, remains:", env_cfg.domain_rand.init_base_vel_range)
        
        # env_cfg.noise.noise_scales.depth = [1., 1.56]
        # env_cfg.noise.noise_scales.depth_vel_threshold = 2.
        # env_cfg.noise.add_noise = False
        env_cfg.asset.terminate_after_contacts_on = []
        env_cfg.termination.termination_terms = []
        # env_cfg.termination.timeout_at_finished = False

        # env_cfg.sensor.realsense.latency_range = [0.03, 0.3]
        # env_cfg.sensor.realsense.refresh_duration = 0.08
        # env_cfg.sensor.realsense.only_update_insight = True
        env_cfg.sensor.proprioception.buffer_length = 6
        # env_cfg.sensor.proprioception.latency_range = [0.0, 0.0]
        
        
        self.env_cfg = env_cfg

    def override_train_cfg(self):
        train_cfg = self.train_cfg
        train_cfg.runner.resume = True
        # train_cfg.runner.policy_class_name = "ActorCriticRecurrent"
        # train_cfg.policy.rnn_num_layers = 1
        # train_cfg.policy.rnn_hidden_size = 512
        # train_cfg.policy.rnn_num_layers = 6
        # train_cfg.policy.rnn_hidden_size = 256
        train_cfg.policy.mu_activation = None
        # train_cfg.policy.mu_activation = "elu"
        
        self.train_cfg = train_cfg

    def register_keyboard_event(self):
        env = self.env
        # register debugging options to manually trigger disruption
        # env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_P, "push_robot")
        # env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_L, "press_robot")
        # env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_J, "action_jitter")
        # env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_Q, "exit")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_R, "agent_full_reset")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_U, "full_reset")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_C, "resample_commands")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_W, "forward")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_S, "backward")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_A, "leftward")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_D, "rightward")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_Q, "leftturn")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_E, "rightturn")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_B, "pitch_up")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_M, "pitch_down")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_X, "stop")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_P, "set_to_object")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_G, "grasp")
        # env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_H, "test_open_gripper")
        # env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_F, "apply_force")

    def act_event(self, actions):
        env_cfg = self.env_cfg
        env = self.env
        agent_model = self.agent_model
        ppo_runner = self.ppo_runner

        for ui_event in env.gym.query_viewer_action_events(env.viewer):
            if ui_event.action == "push_robot" and ui_event.value > 0:
                # manully trigger to push the robot
                env._push_robots()
            if ui_event.action == "press_robot" and ui_event.value > 0:
                env.root_states[:, 9] = torch_rand_float(-env.cfg.domain_rand.max_push_vel_xy, 0, (env.num_envs, 1), device=env.device).squeeze(1)
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "action_jitter" and ui_event.value > 0:
                # assuming wrong action is taken
                obs, critic_obs, rews, dones, infos = env.step(actions + torch.randn_like(actions) * 0.2)
            if ui_event.action == "exit" and ui_event.value > 0:
                print("exit")
                exit(0)
            if ui_event.action == "agent_full_reset" and ui_event.value > 0:
                print("agent_full_reset")
                agent_model.reset()
                # env.create_npc()
            if ui_event.action == "full_reset" and ui_event.value > 0:
                print("full_reset")
                agent_model.reset()
                if hasattr(ppo_runner.alg, "teacher_actor_critic"):
                    ppo_runner.alg.teacher_actor_critic.reset()
                print(env._get_terrain_curriculum_move([self.robot_index]))
                obs, _ = env.reset()
            if ui_event.action == "resample_commands" and ui_event.value > 0:
                print("resample_commands")
                # env._resample_commands(torch.arange(env.num_envs, device= env.device))
                env._resample_npc_pose(torch.arange(env.num_envs, device= env.device))
            if ui_event.action == "stop" and ui_event.value > 0:
                env.commands[:, :] = 0
            if ui_event.action == "forward" and ui_event.value > 0:
                # env.test_remote_control([1,0])
                env.commands[:] = 0.
                env.commands[:, 1] = 1.
            if ui_event.action == "backward" and ui_event.value > 0:
                # env.test_remote_control([-1,0])
                env.commands[:] = 0.
                env.commands[:, 1] = -1.
                # if env.commands[:, 1] > env_cfg.commands.ranges.lin_vel_x[0]:
                #     env.commands[:, 1] -= 0.1
                # env.command_ranges["lin_vel_x"] = [env_cfg.commands.ranges.lin_vel_x[0], env_cfg.commands.ranges.lin_vel_x[0]]
            if ui_event.action == "leftward" and ui_event.value > 0:
                # env.test_remote_control([0,1])
                env.commands[:] = 0.
                env.commands[:, 2] = 1.
                # if env.commands[:, 2] < env_cfg.commands.ranges.lin_vel_y[1]:
                #     env.commands[:, 2] += 0.1
                # env.command_ranges["lin_vel_y"] = [env_cfg.commands.ranges.lin_vel_y[1], env_cfg.commands.ranges.lin_vel_y[1]]
            if ui_event.action == "rightward" and ui_event.value > 0:
                # env.test_remote_control([0,-1])
                env.commands[:] = 0.
                env.commands[:, 2] = -1.
                # if env.commands[:, 2] > env_cfg.commands.ranges.lin_vel_y[0]:
                    # env.commands[:, 2] -= 0.1
                # env.command_ranges["lin_vel_y"] = [env_cfg.commands.ranges.lin_vel_y[0], env_cfg.commands.ranges.lin_vel_y[0]]
            if ui_event.action == "leftturn" and ui_event.value > 0:
                env.commands[:] = 0.
                env.commands[:, 3] = 1.
                # env.command_ranges["ang_vel_yaw"] = [env_cfg.commands.ranges.ang_vel_yaw[1], env_cfg.commands.ranges.ang_vel_yaw[1]]
            if ui_event.action == "rightturn" and ui_event.value > 0:
                env.commands[:] = 0.
                env.commands[:, 3] = -1.
                # env.command_ranges["ang_vel_yaw"] = [env_cfg.commands.ranges.ang_vel_yaw[0], env_cfg.commands.ranges.ang_vel_yaw[0]]
            if ui_event.action == "pitch_up" and ui_event.value > 0:
                env.commands[:] = 0.
                env.commands[:, 4] = 0.2
                # env.root_states[:, 7:10] += quat_rotate(env.base_quat, torch.tensor([[0., 0.5, 0.]], device= env.device))
                # env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "pitch_down" and ui_event.value > 0:
                env.commands[:] = 0.
                env.commands[:, 4] = -0.2
                # env.root_states[:, 7:10] += quat_rotate(env.base_quat, torch.tensor([[0., -0.5, 0.]], device= env.device))
                # env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "set_to_object" and ui_event.value > 0:
                env.test()
            if ui_event.action == "grasp" and ui_event.value > 0:
                ## test change the remote control mode ##
                env_cfg.commands.remote_control = not env_cfg.commands.remote_control
                env.commands[:] = 0.
                env.commands[:, 0] = env_cfg.commands.ranges.grasp[1]
                #########################################
                ## Test apply force to ball ##
                env.apply_force_to_ball()
            if ui_event.action == "test_open_gripper" and ui_event.value > 0:
                # env.test_open_gripper()   
                env.cfg.commands.gripper_open_test = True
            if ui_event.action == "apply_force" and ui_event.value > 0:
                env.set_gripper_by_force()

    def record_frames(self):
        env = self.env
        if RECORD_FRAMES:
                # os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
                path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.train_cfg.runner.experiment_name, self.args.load_run, 'images')
                filename = os.path.join( path, f"{self.img_idx:04d}.png")
                if not os.path.exists(path):
                    os.mkdir(path)
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                self.img_idx += 1
                
    def set_camera(self):
        env_cfg = self.env_cfg
        env = self.env

        camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        camera_vel = np.array([0.6, 0., 0.])
        camera_direction = np.array(env_cfg.viewer.lookat)
        # camera_direction = np.array(env_cfg.viewer.pos)

        if MOVE_CAMERA:
            if CAMERA_FOLLOW:
                camera_position = np.array(env_cfg.viewer.move_pos, dtype=np.float64)
                # camera_position = np.array(env_cfg.viewer.move_pos, dtype=np.float64)
                camera_position[:2] += env.root_states[self.camera_follow_id, :2].cpu().numpy()
                target_position = env.root_states[self.camera_follow_id, :3].cpu().numpy()
                target_position[2] = env_cfg.viewer.move_pos[2]
            else:
                camera_position += camera_vel * env.dt
                target_position = camera_position + camera_direction
            
            ## forward camera: need to set MOVE_CAMERA=True ##
            # camera_position = env.realsense_pos[:]
            # target_position = quat_apply(env.realsense_quat, torch.tensor([[1.,0.,0.]], device=env.device)) + camera_position
            # env.set_camera(camera_position[0], target_position[0])
            ####################
            env.set_camera(camera_position, target_position)

    def log_info(self, actions, teacher_actions, rews, infos, i):
        robot_index = self.robot_index
        env = self.env
        logger = self.logger

        joint_index = 9 # which joint is used for logging
        stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

        if (abs(env.substep_torques[robot_index]) > 35.).any():
            exceed_idxs = torch.where(abs(env.substep_torques[robot_index]) > 35.)
            print("substep_torques:", exceed_idxs[1], env.substep_torques[robot_index][exceed_idxs[0], exceed_idxs[1]])
        if i < self.stop_state_log:
            # log obs
            obs_buf = env.obs_buf[robot_index]
            log_obs_dicts = {
                'base_ang_vel_0': obs_buf[3].item(),
                'base_ang_vel_1': obs_buf[4].item(),
                'base_ang_vel_2': obs_buf[5].item(),

                'projected_gravity_0': obs_buf[6].item(),
                'projected_gravity_1': obs_buf[7].item(),
                'projected_gravity_2': obs_buf[8].item(),

                'cmd_0': obs_buf[9].item(),
                'cmd_1': obs_buf[10].item(),
                'cmd_2': obs_buf[11].item(),

                'goal_rel_robot_0': obs_buf[51].item(),
                'goal_rel_robot_1': obs_buf[52].item(),
                'goal_rel_robot_2': obs_buf[53].item(),

                'goal_yaw_robot': obs_buf[54].item(),
                'object_confidence': obs_buf[55].item(),
            }
            logger.log_obs(log_obs_dicts)

            dof_names = env.dof_names
            angles_dict = {}
            angles_cmd_dict = {}
            vel_dict = {}
            for i, name in enumerate(dof_names):
                # angles_dict[name] = env.obs_buf[robot_index, 12:26][i].item()
                angles_dict[name] = env.dof_pos[0,i].item()
                cmd_angle = actions[robot_index, i] * env.cfg.control.action_scale + env.default_dof_pos[robot_index, i]
                angles_cmd_dict[name] = cmd_angle.item()
                vel_dict[name] = env.substep_dof_vel[robot_index,:, i].max().item()
                # vel_dict[name] = env.torques[robot_index, i].item()
            logger.log_angles(angles_dict, angles_cmd_dict, vel_dict)


            if torch.is_tensor(env.cfg.control.action_scale):
                action_scale = env.cfg.control.action_scale.detach().cpu().numpy()[joint_index]
            else:
                action_scale = env.cfg.control.action_scale
            base_roll = get_euler_xyz(env.base_quat)[0][robot_index].item()
            base_pitch = get_euler_xyz(env.base_quat)[1][robot_index].item()
            if base_pitch > torch.pi: base_pitch -= torch.pi * 2
            def reward_removed_term(term):
                return {"reward_removed_" + term: rews[robot_index].item() - (getattr(env, ("_reward_" + term))() * env.reward_scales[term])[robot_index].item()}
            log_states_dicts = {
                    # 'dof_pos_target': env.actions_scaled_torque_clipped[robot_index, joint_index].item(),
                    'dof_pos_target': env.actions[robot_index, joint_index].item() * action_scale,
                    'dof_pos': (env.dof_pos - env.default_dof_pos)[robot_index, joint_index].item(),
                    'dof_vel': env.substep_dof_vel[robot_index, 0, joint_index].max().item(),
                    'dof_torque': torch.mean(env.substep_torques[robot_index, :, joint_index]).item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_pitch': base_pitch,
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    # 'max_torques': torch.abs(env.substep_torques).max().item(),
                    'max_vel': torch.abs(env.substep_dof_vel).max().item(),
                    'max_torque_motor': torch.argmax(torch.max(torch.abs(env.substep_torques[robot_index]), dim= -2)[0], dim= -1).item() % 3, # between hip, thigh, calf
                    'max_torque_leg': int(torch.argmax(torch.max(torch.abs(env.substep_torques[robot_index]), dim= -2)[0], dim= -1).item() / 4), # between hip, thigh, calf
                    "student_action": actions[robot_index, joint_index].item(),
                    "teacher_action": teacher_actions[robot_index, joint_index].item(),
                    "reward": rews[robot_index].item(),
                    "power": torch.max(torch.sum(env.substep_torques * env.substep_dof_vel, dim= -1), dim= -1)[0][robot_index].item(),

                }
            logger.log_states(log_states_dicts)
        elif i==self.stop_state_log:
            logger.plot_states()
            # import matplotlib.pyplot as plt
            # plt.plot(logger.state_log["dof_torque"], label="robot{}joint{}torque".format(robot_index, joint_index))
            # plt.legend()
            # plt.show()
            env._get_terrain_curriculum_move(torch.tensor([0], device= env.device))
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

    def check_done(self, dones):
        if dones.any():
            self.agent_model.reset(dones)
            if hasattr(self.ppo_runner.alg, "teacher_actor_critic"):
                self.ppo_runner.alg.teacher_actor_critic.reset(dones)
            if self.env.time_out_buf[dones].any():
                print("env dones, because has timeout")
            else:
                print("env dones, because has fallen")

    @torch.no_grad()
    def play(self):
        args = self.args
        env = self.env
        ppo_runner, policy = self.ppo_runner, self.policy

        obs = env.get_observations()
        critic_obs = env.get_privileged_observations()

        env.commands[:,:] = 0. # to beter compare action rate of different checkpoints.
        # env.commands[0,3] = -0.2
        
        REAL2SIM = False
        if REAL2SIM: # for real robot debug
            # file = open('actions.txt', 'w')
            # real_file = open('real_actions.txt', 'r').readlines()
            file_pth = os.path.dirname(os.path.realpath(__file__))
            # obs_lines = open(f'{file_pth}/../../logs/{args.task}/{args.load_run}/{args.load_run}_obs.txt', 'r').readlines()
            obs_lines = open(f'{file_pth}/../../logs/{args.task}/{args.load_run}/{args.load_run}actions.txt', 'r').readlines()
        
        for i in range(10*int(env.max_episode_length)):
            epoch_start_time = time.monotonic()
            if args.slow > 0:
                time.sleep(args.slow)
            
            if False:
                idx = i%len(obs_lines)
                if idx == 0:
                    print("start")
                line = eval(obs_lines[idx].strip())
                # real_obs = torch.tensor([line], device=env.device)
                # env.goal_rel_robot_obs[0] = real_obs[0,51:54]
                # obs[0, 51:] = real_obs[0,51:]
                # obs[0, :51] = real_obs[0,:51]
                # obs = real_obs
                # env.obs_buf[0, 51:] = real_obs[0,51:]
                # env.obs_buf[0, :] = real_obs[0,:]
                # obs[0, :12] = torch.zeros_like(obs[0, :12])
                # obs[0, 12:26] = torch.zeros_like(env.dof_pos)
                # obs[0, 26:40] = torch.zeros_like(env.dof_vel)
                # obs[0, 40:54] = torch.zeros_like(env.actions)
                # obs[0, 54:60] = torch.zeros_like(obs[0, 54:60])
                # obs[0, 12:26] = ((env.dof_pos - env.default_dof_pos) * env.obs_scales.dof_pos).cpu()
                # obs[0, 26:40] = (env.dof_vel * env.obs_scales.dof_vel).cpu()
                # print(obs[0, 51:56])
                tmp_goal_pos = quat_rotate(env.grasp_point_quat, obs[:, 51:54])
                sphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0, 1, 0))
                pos = tmp_goal_pos[0] + env.grasp_point_pos[0]
                x, y, z = pos.squeeze()
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[0], sphere_pose)
                
            if i >= 20:
                actions = policy(obs.detach())
            else:
                actions = torch.zeros_like(env.prev_actions)
            
            if REAL2SIM:
                idx = i%len(obs_lines)
                if idx == 0:
                    print("start")
                line = eval(obs_lines[idx].strip())
                actions = torch.tensor([line], device=env.device)
            
            # self.agent_model.reset()
            teacher_actions = actions
            if not args.no_teacher and "distill" in args.task:
                teacher_actions = ppo_runner.alg.teacher_actor_critic.act_inference(critic_obs.detach())
            
            if False:
                line = real_file[i].strip()
                actions = torch.tensor([eval(line)], device=env.device)

            if False:
                file.write(f"{env.obs_buf[0].detach().cpu().numpy().tolist()}\n")
            
            if args.show_teacher:
                obs, critic_obs, rews, dones, infos = env.step(teacher_actions.detach())
            else:
                obs, critic_obs, rews, dones, infos = env.step(actions.detach())
            
            # print(env.obs_buf[0], env.goal_rel_robot, env.object_confidence)
            # print(env.prev_actions)
            # print(env.obs_buf[:, 45:])
            # print(env.commands)
            # print(env.goal_rel_robot_obs)
            # print(env.contact_forces[:, 27])
            # print(env.episode_length_buf)
            # print(env.base_lin_vel)
            # print(torch.mean(env.episode_sums["gripper_close"][0]))
            # print("y:", obs[0, 46])
            # print(env.target_obs_last)
            # print()
            
            self.record_frames()            
            self.set_camera()
            self.act_event(actions)
            self.log_info(actions, teacher_actions, rews, infos, i)
            self.check_done(dones)
            
            end_time = time.monotonic()
            # print(end_time - epoch_start_time)


EXPORT_POLICY = False
RECORD_FRAMES = True
MOVE_CAMERA = True
CAMERA_FOLLOW = MOVE_CAMERA
start_time = time.time()
    

    
args = get_args([
    dict(name= "--slow", type= float, default= 0.0, help= "slow down the simulation by sleep secs (float) every frame"),
    dict(name= "--show_teacher", action= "store_true", default= False, help= "show teacher actions"),
    dict(name= "--no_teacher", action= "store_true", default= False, help= "whether to disable teacher policy when running the script"),
    # dict(name= "--zero_act_until", type= int, default= 0., help= "zero action until this step"),
    dict(name= "--sample", action= "store_true", default= False, help= "sample actions from policy"),
    dict(name= "--plot_time", type= int, default= 100, help= "plot states after this time"),
    dict(name= "--no_throw", action= "store_true", default= False),
])

# auto trans from imges to video

if RECORD_FRAMES:
    atexit.register(exit_handler)

if __name__ == '__main__':    
    node = PlayNode(args)
    node.play()
