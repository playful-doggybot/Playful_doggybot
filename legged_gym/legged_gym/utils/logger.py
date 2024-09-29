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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.obs_log = defaultdict(list)
        self.angles_log = defaultdict(list)
        self.angles_cmd_log = defaultdict(list)
        self.vel_dict = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def log_obs(self, dict):
        for key, value in dict.items():
            self.obs_log[key].append(value)

    def log_angles(self, dict, cmd_dict, vel_dict):
        for key, value in dict.items():
            self.angles_log[key].append(value)
        for key, value in cmd_dict.items():
            self.angles_cmd_log[key].append(value)
        for key, velue in vel_dict.items():
            self.vel_dict[key].append(velue)

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 4
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base pitch
        a = axs[0, 2]
        if log["base_pitch"]:
            a.plot(time, log["base_pitch"], label='measured')
            a.plot(time, [-0.75] * len(time), label= 'thresh')
        # if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang [rad]', title='Base pitch')
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # # plot torque/vel curves
        # a = axs[2, 1]
        # if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        # a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        # a.legend()
        # plot power curves
        a = axs[2, 1]
        if log["power"]!=[]: a.plot(time, log["power"], label='power [W]')
        a.set(xlabel='time [s]', ylabel='Power [W]', title='Power')
        # plot torques
        a = axs[2, 2]
        if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        # plot rewards
        a = axs[3, 0]
        if log["max_torques"]: a.plot(time, log["max_torques"], label='max_torques')
        if log["max_torque_motor"]: a.plot(time, log["max_torque_motor"], label='max_torque_motor')
        if log["max_torque_leg"]: a.plot(time, log["max_torque_leg"], label='max_torque_leg')
        if log["max_vel"]: a.plot(time, log["max_vel"], label='max_vel')
        # a.set(xlabel='time [s]', ylabel='max_torques [Nm]', title='max_torques')
        a.legend(fontsize= 5)
        # plot customed data
        a = axs[3, 1]
        if log["student_action"]:
            a.plot(time, log["student_action"], label='s')
            a.plot(time, log["teacher_action"], label='t')
        a.legend()
        a.set(xlabel='time [s]', ylabel='value before step()', title='student/teacher action')
        a = axs[3, 2]
        a.plot(time, log["reward"], label='rewards')
        for i in log["mark"]:
            if i > 0:
                a.plot(time, log["mark"], label='user mark')
                break
        for key in log.keys():
            if "reward_removed_" in key:
                a.plot(time, log[key], label= key)
        a.set(xlabel='time [s]', ylabel='', title='rewards')
        # a.set_ylim([-0.12, 0.1])
        a.legend(fontsize = 5)
        
        self._plot_obs(time)
        self._plot_angles(time)
        # self._plot_mean_dof_vel()

        plt.show()

    def _plot_mean_dof_pos(self, a, log, time):
        # plot joint targets and measured positions
        # a = axs[1, 0]
        if log["dof_pos"]: 
            a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]: 
            a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()

    def _plot_mean_dof_vel(self, a, log, time):
        if log["dof_vel"]: 
            a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: 
            a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()

    def _plot_obs(self, time):
        nb_rows, nb_cols = 2, 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        log = self.obs_log

        a = axs[0, 0]
        if log["base_ang_vel_0"]:
            a.plot(time, log["base_ang_vel_0"], label='0')
            a.plot(time, log["base_ang_vel_1"], label='1')
            a.plot(time, log["base_ang_vel_2"], label='2')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Base ang vel')
        a.legend()

        a = axs[0, 1]
        if log["projected_gravity_0"]:
            a.plot(time, log["projected_gravity_0"], label='0')
            a.plot(time, log["projected_gravity_1"], label='1')
            a.plot(time, log["projected_gravity_2"], label='2')
        a.set(xlabel='time [s]', ylabel='gravity', title='Projected gravity')
        a.legend()

        a = axs[0, 2]
        if log["cmd_0"]:
            a.plot(time, log["cmd_0"], label='grasp')
            a.plot(time, log["cmd_1"], label='lin_x')
            a.plot(time, log["cmd_2"], label='lin_y')
        a.set(xlabel='time [s]', ylabel='Commands [m/s]', title='Commands')
        a.legend()
        
        a = axs[1, 0]
        if log["goal_rel_robot_0"]:
            a.plot(time, log["goal_rel_robot_0"], label='0')
            a.plot(time, log["goal_rel_robot_1"], label='1')
            a.plot(time, log["goal_rel_robot_2"], label='2')
        a.set(xlabel='time [s]', ylabel='pos [m]', title='Goal rel pos')
        a.legend()

        a = axs[1, 1]
        if log["goal_yaw_robot"]:
            a.plot(time, log["goal_yaw_robot"], label='yaw')
        a.set(xlabel='time [s]', ylabel='yaw [rad]', title='Goal yaw robot')
        a.legend()

        a = axs[1, 2]
        if log["object_confidence"]:
            a.plot(time, log["object_confidence"], label='')
        a.set(xlabel='time [s]', ylabel='percentage', title='object confidence')
        a.legend()

    def _plot_angles(self, time):
        log = self.angles_log
        cmd_log = self.angles_cmd_log
        vel_log = self.vel_dict
        dof_names = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'l_finger_joint', 'r_finger_joint']

        nb_rows = 4
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for i in range(nb_rows):
            for j in range(nb_cols):
                a = axs[i, j]
                dof_name = dof_names[i*3 + j]
                if dof_name in log.keys():
                    a.plot(time, cmd_log[dof_name], label='cmd')
                    a.plot(time, log[dof_name], label='measured')
                    a.plot(time, vel_log[dof_name], label='torque')
                    a.set(xlabel='time [s]', ylabel='dof pos', title=dof_name)
                    a.legend()
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
            