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
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

action_scale = 0.5

GO2_CONST_DOF_RANGE = dict(  # copied from go1 config, cannot find in unitree sdk.
            hip_max=1.047, # 1.047
            hip_min=-1.047, 
            thigh_max=1.9, # 2.966
            thigh_min=-0.663,
            calf_max=-0.837,
            calf_min=-2.721, # -2.7
            finger_min = -0.4,
            finger_max = 0.4   
        )

JOINT_ANGLES = {  # = target angles [rad] when action = 0.0
            'FR_hip_joint': -0.1,  # [rad]
            'FL_hip_joint': 0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]

            'l_finger_joint': -0.01,
            'r_finger_joint': -0.01,
        }
DESIRED_JOINT_ANGLES = {  # = target angles [rad] when action = 0.0
            'FR_hip_joint': -0.1,  # [rad]
            'FL_hip_joint': 0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]

            'FL_thigh_joint': 0.45,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.45,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.3,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.3,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]

            'l_finger_joint': -0.01,
            'r_finger_joint': -0.01,
        }

class Go2MixCmdCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        zero_actions = False
        default_joint_angles = JOINT_ANGLES
        desired_dof_pos = DESIRED_JOINT_ANGLES

        # define the properties of the target ball to catch
        object_radius = 0.02

        object_pos_x = [1.3, 1.8] 
        object_pos_y = [-0.1, 0.1]
        object_pos_z = [0.55, 0.55]

        object_pos_z_curriculum_increment = 0.05
        object_pos_z_max_curriculum_level = 5
        max_init_object_pos_z_level = 3
        curriculum_length = 100

        curriculum_up = 1.
        curriculum_down = -1.
 
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2g_description_vertical.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2g_description.urdf'
        target_object_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/hanging_ball.urdf'
        name = "go2"
        foot_name = "foot"
        sdk_dof_range = GO2_CONST_DOF_RANGE
        
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        density = 1000 # TODO

        penalize_contacts_on = ["base", "thigh", "calf", "realsense_link", "radar", "gripper_link", "l_finger_link", "r_finger_link"] # , "gripper_link",
        # penalize_contacts_on = []
        # terminate_after_contacts_on = ["base", "imu", "gripper_link", "l_finger_link", "r_finger_link"]
        terminate_after_contacts_on = ["base", "imu"] # 
        # terminate_after_contacts_on = []
        front_hip_names = ["FR_hip_joint", "FL_hip_joint"]
        rear_hip_names = ["RR_hip_joint", "RL_hip_joint"]
        
        joint_max_velocity = 40.

        replace_cylinder_with_capsule = False
        
    class sim(LeggedRobotCfg.sim):
        dt =  0.005
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        body_measure_points = { # transform are related to body frame
            "base": dict(
                x= [i for i in np.arange(-0.2, 0.31, 0.03)],
                y= [-0.08, -0.04, 0.0, 0.04, 0.08],
                z= [i for i in np.arange(-0.061, 0.071, 0.03)],
                transform= [0., 0., 0.005, 0., 0., 0.],
            ),
            "thigh": dict(
                x= [
                    -0.16, -0.158, -0.156, -0.154, -0.152,
                    -0.15, -0.145, -0.14, -0.135, -0.13, -0.125, -0.12, -0.115, -0.11, -0.105, -0.1, -0.095, -0.09, -0.085, -0.08, -0.075, -0.07, -0.065, -0.05,
                    0.0, 0.05, 0.1,
                ],
                y= [-0.015, -0.01, 0.0, -0.01, 0.015],
                z= [-0.03, -0.015, 0.0, 0.015],
                transform= [0., 0., -0.1,   0., 1.57079632679, 0.],
            ),
            "calf": dict(
                x= [i for i in np.arange(-0.13, 0.111, 0.03)],
                y= [-0.015, 0.0, 0.015],
                z= [-0.015, 0.0, 0.015],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }
        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8 # 16 TODO 
            num_velocity_iterations = 0 
            contact_offset = 0.001  # [m]
            rest_offset = 0.00   # [m]
            bounce_threshold_velocity = 0.1 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 9*1024**2 # max is 50*1024**2 ;2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 8
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class noise( LeggedRobotCfg.noise ):
        add_noise = True # disable internal uniform +- 1 noise, and no noise in proprioception
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            forward_depth = 0.1
            base_pose = 1.0
            goal = 0.01 # 0.0
            depth = [0.0, 0.002]
            depth_vel_threshold = 1.9
            # goal = 0.02
            not_insight = 0.01
            dof_pos = 0.03
            dof_pos = 0.08
            dof_vel = 0.05
            ang_vel = 0.05
            imu = 0.2
            forward_camera = 0.02
            commands = 0.

    class viewer( LeggedRobotCfg.viewer ):
        pos = [9., 5., 0.6]
        lookat = [6., 7., 0.44]  # [m]
        move_pos = [1.9, -2.57, 0.9] # [m]
        draw_volume_sample_points = False
        debug_viz = False

    class termination:
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            # "ball_fly_away",
         
        ]

        roll_kwargs = dict(
            threshold= 2., # [rad]
            # tilt_threshold= 1.5,
            # jump_threshold= 1.5,
        )
        pitch_kwargs = dict(
            threshold= 1.6, # [rad] # for leap, jump
            # jump_threshold= 1.6,
            # leap_threshold= 1.5,
        )
        z_low_kwargs = dict(
            threshold= 0.05, # [m]
        )
        z_high_kwargs = dict(
            threshold= 1.5, # [m]
        )
        out_of_track_kwargs = dict(
            threshold= 1., # [m]
        )
        ball_fly_away_kwargs = dict(
            threshold= (0.1, 1.5),
        )
        catch_ball_kwargs = dict(
            threshold = 1.5
        )


    class curriculum:
        no_moveup_when_fall = False
        # chosen heuristically, please refer to `LeggedRobotField._get_terrain_curriculum_move` with fixed body_measure_points

    class env(LeggedRobotCfg.env):
        num_envs = 8192 # try 5120
        num_envs = 4096
        # num_envs = 64
        episode_length_s = 12
        obs_components = [
            "proprioception",  # 51
            "target", # 6, 1 redundant
            "commands", # 5, [grasp, lin_x, lin_y, yaw, pitch] + 3 redundant
            # "robot_config" 
        ]
        use_lin_vel = False # TODO
        privileged_use_lin_vel = False
        # num_actions = 14 # 1 dummy action
        num_actions = 12 # 1 dummy action
        num_observations = 81 # history memory dims 65 + 16
        num_state_chunck = 4 # if not use transformer, set num_state_chunck = 1
        num_state_chunck = 5
        # num_state_chunck = 1

    class control( LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # torque_limits = [30., 30., 25.] * 4 + [30.]*2 
        torque_limits = [30., 30., 25.] * 4  # num_actions
        stiffness = {'joint':35., 'gripper':10000}  # [N*m/rad]
        damping = {'joint': 0.6, 'gripper':2000}     # [N*m*s/rad]
        # damping = {'joint': 1., 'gripper':2000}

        computer_clip_torque = False
        motor_clip_torque = False

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4


    class commands(LeggedRobotCfg.commands):
        num_commands = 5 # 0:grasp(+), 1:forward(+)/backward(-), 2:left(-)/right(+)
        heading_command = False
        resampling_time = 10 # [s]
        curriculum = True
        remote_control = False
        commands_probabilities = [1., 0.] # [0., 1.] for walk policy

        class ranges(LeggedRobotCfg.commands.ranges):
            grasp = [0., 0.5]
            # grasp = [0., 1.]
            lin_vel_x = [-1, +1.]
            lin_vel_y = [-1., 1.]
            yaw = [-1., 1.]
            pitch = [-0.4, 0.4]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_com = True
        class com_range: # base mass center
            x = [-0.1, 0.15]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]

        randomize_motor = True
        leg_motor_strength_range = [0.8, 1.2]

        randomize_base_mass = True
        added_mass_range = [1.0, 3.0]
        
        randomize_friction = True
        friction_range = [-0., 1.2] 
        npc_friction_range = [0.5, 0.55]
        
        push_robots = True 
        max_push_vel_xy = 0.5 # [m/s]
        push_interval_s = 2.5

        init_base_pos_range = dict(
            x=[-0.1, 0.1],
            y=[-0.1, 0.1],
        )
        init_base_rot_range = dict(
            roll=[-0.8, 0.8],
            pitch=[-0.8, 0.8],
        )
        init_base_vel_range = dict(
            x=[-0.5, 0.8],
            y=[-0.8, 0.8],
            z=[-0.2, 0.2],
            roll=[-0.7,0.7],
            pitch=[-0.7,0.7],
            yaw=[-0.8,0.8],
        )

        init_dof_vel_range = [-2, 2]

    class terrain( LeggedRobotCfg.terrain):
        mesh_type = "trimesh" # Don't change. "none", "plane", "heightfield" or "trimesh"
        num_rows = 6
        num_cols = 6
        selected = "TerrainPerlin" # "BarrierTrack" or "TerrainPerlin"; "TerrainPerlin" can be used for training a walk policy.
        max_init_terrain_level = 0
        border_size = 5
        slope_treshold = 5.
        static_friction = 0.5
        dynamic_friction = 0.2

        curriculum = False # for walk
        horizontal_scale = 0.025 # [m]
        # vertical_scale = 1. # [m]
        pad_unavailable_info = True
        measure_heights = True

        TerrainPerlin_kwargs = dict(
            zScale=0.07, 
            frequency=20,
        )


    class sensor:
        class forward_camera:
            resolution = [480, 848]
            ##### forward camera ########
            # position = [0.26, 0., 0.03] # position in base_link
            # rotation = [0., 0., 0.] # ZYX Euler angle in base_link

            ##### for realsense #########
            position = [0.3, 0.3, 0.1] # [0.285 0 0.01] + [0.055 0. 0.15]
            rotation = [0, 0.75, 0]
            
            latency_range = [0., 0.]
            refresh_duration = 1

        class proprioception:
            delay_action_obs = True
            latency_range = [0.03, 0.08] # TODO: check buffer length
            latency_resample_time = 2.0 # [s]
            refresh_duration = 0.02
            buffer_length = 5

        class realsense:
            fov = [1.0, 0.533] # degree:[69., 42.]=>[1.204, 0.733](1920x1080); [0.53, 0.32] [848, 480]
            horizontal_fov = 69
            resolution = [848, 480] # [1920, 1080]
            camera_angle = 0.
            refresh_duration = 0.12
            refresh_duration = 0.034
            refresh_duration = 0.067
            refresh_duration = 0.04
            # refresh_duration = 0.083
            delay_action_obs = True
            latency_range = [0.02, 0.06]
            # latency_range = [0.02, 0.02]
            buffer_length = 4
            latency_resample_time = 2.0 # [s]
            orin = False
            only_update_insight = True
            position = [0.06, 0., 0.18]
            rotation = [0., 0.3, 0.]
            use_absolute_height = True
            no_absolute_height_hint = False
            
        class viewer_camera:
            position = [0.9, 1.1, 0.4]
            rotation = [0., 0.3, -1.79]
            resolution = [640, 320]
            horizontal_fov = 100

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            forward_depth = 1.
            base_pose = [0., 0., 0., 1., 1., 1.]
            engaging_block = 1.
            dof_pos = 1.0 
            dof_vel = 0.05

        dof_pos_redundancy = 0.2
        clip_actions_method = "hard"
        clip_actions_low = []
        clip_actions_high = []
        for sdk_joint_name, sim_joint_name in zip(
            # ["Finger"]*2 + ["Hip", "Thigh", "Calf"] * 4,
            ["hip", "thigh", "calf"] * 4,
            [ # in the order as simulation
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                # "l_finger_joint", "r_finger_joint",
            ],
        ):
            clip_actions_low.append( (GO2_CONST_DOF_RANGE[sdk_joint_name + "_min"] + dof_pos_redundancy - JOINT_ANGLES[sim_joint_name]) / action_scale)
            clip_actions_high.append( (GO2_CONST_DOF_RANGE[sdk_joint_name + "_max"] - dof_pos_redundancy - JOINT_ANGLES[sim_joint_name]) / action_scale)
        del dof_pos_redundancy, sdk_joint_name, sim_joint_name

        # cannot calculate finger using dof_pos_redundancy


    class rewards(LeggedRobotCfg.rewards): # adjusted, can run
        class scales:
            ###### tracking rewards ######
            tracking_goal_vel = 0.9
            tracking_yaw = 0.3
            # tracking_yaw = 0.5
            tracking_goal_pos = 2.
            tracking_goal_pos = 3.

            # tracking_lin_vel = 0.8
            # tracking_ang_vel = 0.4
            # tracking_pitch = 0.1

            ###### gripper action rewards ######
            gripper_close = 30.0

            ###### regularization rewards ######
            # action_rate = -0.4 # no use [-0.1, -0.25, -0.5]
            action_rate = -0.1
            # action_rate = -0.08 # too small will be no use
            ang_vel_xy = -0.03
            ang_vel_xy = -0.05
            # ang_vel_xy = -0.1
            # ang_vel_z = -0.8
            # object_pitch = -0.08
            alive = 1.
            
            collision = -0.05
            # delta_torques = -1.e-9

            dof_acc = -1.e-7
            dof_error = -0.15
            # dof_error_front = -0.2
            # rear_hip_dof_error = -0.001
            front_hip_dof_error = -1.
            # dof_vel = -1.e-4 # penalize dof velocity

            exceed_dof_pos_limits = -0.2
            # exceed_torque_limits_l1norm = -5.e-2
            # exceed_dof_vel_l1norm = -0.2
            # exceed_dof_vel_ratio_limits = -0.08
            # exceed_torque_limits_square = -1.e-4
            # exceed_torque_limits_i = -1.e-4
            hip_pos = -0.1 # can jump, jump walk: -5.e-08, -5.e-07, -1.e-06; only walk -1e-04 ok
            legs_energy_substeps = -1.e-08 # -6 too large
            stand_still = -0.05 # based on cmds
            # stand_still = -0.8
            torques = -1.0e-4 # should be < -1.e-4
            lin_vel_z = -0.1

            base_height = -0.1
            
            # orientation = -0.5
            termination = -60.
                        
        cam_target_norm_th = 0.15
        action_rate_th = 1.
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.4 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_goal_pos = 0.2 # more smooth
        soft_dof_pos_limit = 0.7 # percentage of urdf limits, values above this limit are penalized, not smaller than 0.6
        soft_dof_pos_limit = 0.6
        # soft_dof_pos_limit = 0.4
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.8
        base_height_target = 0.28
        max_contact_force = 50. # forces above this value are penalized
        
        ###### grasp threshold ######
        lin_vel_z_xy_norm_threshold = 0.25
        xy_vel_norm_threshold = 0.2 # tracking goal vel of x and y
        xy_vel_norm_threshold = 0.35
        xy_pos_norm_threshold = xy_vel_norm_threshold
        grasp_norm_threshold = 0.03 # when to close gripper
        pos_norm_threshold = 0.05 # when to start apply force and give pos bool rew, cannot smaller than 0.05
        no_hip_pos_penalty_while_jump = False # Don't penalize hip pos while close object and try to jump


class Go2MixCmdCfgPPO(LeggedRobotCfgPPO):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.05 # TODO, 0.2 before
        learning_rate = 1e-4
        # desired_kl = 5e-3 # 1e-3 may be too small
        desired_kl = 5e-3
        # desired_kl = 1e-2 # from scracth
        # desired_kl = 2e-3
        # desired_kl = 1e-3

    class policy( LeggedRobotCfgPPO.policy ):
        rnn_type = 'gru'
        # mu_activation = "elu"
        mu_activation = None
        # rnn_hidden_size = 512
        rnn_hidden_size = 256
        rnn_num_layers = 1
        # rnn_num_layers = 3
        # rnn_num_layers = 6
        # rnn_hidden_size = 64
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        # actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [512, 256, 128]
        # actor_hidden_dims = [1024, 512, 256]
        # critic_hidden_dims = [1024, 512, 256]
        hist_num = 16

        transformer_hidden_size = 256
        # transformer_hidden_size = 512
        transformer_num_layers = 3
        # transformer_num_layers = 1

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticTransformer" # ActorCriticHistory, ActorCriticRecurrent, ActorCritic, ActorCriticTransformer
        run_name = ''
        experiment_name =  'go2_mix_cmd' #'go2_mix_cmd'
        resume = True
        load_run = None
        load_run = "Apr03_23-08-42_highhighProb_rAngTrack0.4_rLinTrack0.8_noResume"
        load_run = "Sep12_16-25-35_ActorCriticTransformer__crclLth100_trackingGoalPos-3e+00_gClose-4e+01_fromSep02_16-31-41"

        # MLP
        # load_run = "Aug22_10-51-41_ActorCritic_crclLth100_trackingGoalPos-1e+01_gClose-1e+01_noResume"

        run_name = "".join([
                            (policy_class_name), # Transformer: 256x1, 256x3, 512x1, 512x3, 512x4; MLP:1024
                            ("_"),
                            # ("_hint0.4" if not Go2MixCmdCfg.sensor.realsense.no_absolute_height_hint else "_noH0.4"),
                            # ("trackingYawSigma2.e_"),
                            # ("_useLinVel" if Go2MixCmdCfg.env.use_lin_vel else ""),
                            # ("_crcl" if getattr(Go2MixCmdCfg.commands, "curriculum", False) else ""),
                            ("_crclLth" +str(Go2MixCmdCfg.init_state.curriculum_length) if getattr(Go2MixCmdCfg.init_state, "curriculum_length", "") else ""),
                            # ("_crclUp" +str(Go2MixCmdCfg.init_state.curriculum_up) if getattr(Go2MixCmdCfg.init_state, "curriculum_up", "") else ""),
                            # ("_crclDown" +str(Go2MixCmdCfg.init_state.curriculum_down) if getattr(Go2MixCmdCfg.init_state, "curriculum_down", "") else ""),
                            # ("_graspTh"+ str(Go2MixCmdCfg.rewards.grasp_norm_threshold)),
                            # ("_onlyPstRew" if getattr(Go2MixCmdCfg.rewards, "only_positive_rewards", False) else ""),
                            # ("_FPS"+str(Go2MixCmdCfg.sensor.realsense.refresh_duration) if getattr(Go2MixCmdCfg.sensor.realsense, "refresh_duration", "") else ""),
                            # ("_pAngVelXy" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.ang_vel_xy,trim="-", exp_digits=1) if getattr(Go2MixCmdCfg.rewards.scales, "ang_vel_xy", 0.) != 0. else ""),
                            # ("_pgOpen" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.gripper_open,trim="-", exp_digits=1) if getattr(Go2MixCmdCfg.rewards.scales, "gripper_open", 0.) != 0. else ""),
                            # ("_pActRate" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.action_rate,trim="-", exp_digits=1) if getattr(Go2MixCmdCfg.rewards.scales, "action_rate", 0.) != 0. else ""),
                            # ("_pActRateB" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.action_rate_boolean,trim="-", exp_digits=1) if getattr(Go2MixCmdCfg.rewards.scales, "action_rate_boolean", 0.) != 0. else ""),
                            # ("_rAlive{:.1f}".format(Go2MixCmdCfg.rewards.scales.alive) if getattr(Go2MixCmdCfg.rewards.scales, "alive", 0.) != 0. else ""),
                            # ("_pclls{:.0e}".format(-Go2MixCmdCfg.rewards.scales.collision) if getattr(Go2MixCmdCfg.rewards.scales, "collision", 0.) != 0. else ""),
                            # ("_pDeltaTorq" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.delta_torques, trim= "-", exp_digits= 1) if getattr(Go2MixCmdCfg.rewards.scales, "delta_torques", 0.) != 0. else ""),
                            # ("_pDofAcc{:.0e}".format(-Go2MixCmdCfg.rewards.scales.dof_acc) if getattr(Go2MixCmdCfg.rewards.scales, "dof_acc", 0.) != 0. else ""),
                            # ("_pDofErr{:.0e}".format(-Go2MixCmdCfg.rewards.scales.dof_error) if getattr(Go2MixCmdCfg.rewards.scales, "dof_error", 0.) != 0. else ""),
                            # ("_pDofErrFrt{:.0e}".format(-Go2MixCmdCfg.rewards.scales.dof_error_front) if getattr(Go2MixCmdCfg.rewards.scales, "dof_error_front", 0.) != 0. else ""),
                            # ("_pFrontDofErr{:.0e}".format(-Go2MixCmdCfg.rewards.scales.front_hip_dof_error) if getattr(Go2MixCmdCfg.rewards.scales, "front_hip_dof_error", 0.) != 0. else ""),
                            # ("_prearHipDofErr{:.0e}".format(-Go2MixCmdCfg.rewards.scales.rear_hip_dof_error) if getattr(Go2MixCmdCfg.rewards.scales, "rear_hip_dof_error", 0.) != 0. else ""),
                            # ("_pDofVel{:.0e}".format(-Go2MixCmdCfg.rewards.scales.dof_vel) if getattr(Go2MixCmdCfg.rewards.scales, "dof_vel", 0.) != 0. else ""),

                            # ("_pExcdDofPos" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.exceed_dof_pos_limits, trim= "-", exp_digits= 1) if getattr(Go2MixCmdCfg.rewards.scales, "exceed_dof_pos_limits", 0.) != 0. else ""),
                            # ("_pExcdTorqL1" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.exceed_torque_limits_l1norm, trim="-",exp_digits=1) if getattr(Go2MixCmdCfg.rewards.scales, "exceed_torque_limits_l1norm",0.) != 0. else ""),
                            # ("_pExcdTorqSq" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.exceed_torque_limits_square, trim="-",exp_digits=1) if getattr(Go2MixCmdCfg.rewards.scales, "exceed_torque_limits_square",0.) != 0. else ""),
                            # ("_pExcdTorq_i" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.exceed_torque_limits_i, trim="-",exp_digits=1) if getattr(Go2MixCmdCfg.rewards.scales, "exceed_torque_limits_i",0.) != 0. else ""),
                            # ("_pExcdVelL1" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.exceed_dof_vel_l1norm, trim="-",exp_digits=1) if getattr(Go2MixCmdCfg.rewards.scales, "exceed_dof_vel_l1norm",0.) != 0. else ""),
                            # ("_pExcdVelR" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.exceed_dof_vel_ratio_limits, trim="-",exp_digits=1) if getattr(Go2MixCmdCfg.rewards.scales, "exceed_dof_vel_ratio_limits",0.) != 0. else ""),

                            # ("_pHipPos" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.hip_pos, trim="-",exp_digits=1) if getattr(Go2MixCmdCfg.rewards.scales, "hip_pos",0.) != 0. else ""),
                            # ("_pEngSub{:.0e}".format(-Go2MixCmdCfg.rewards.scales.legs_energy_substeps) if getattr(Go2MixCmdCfg.rewards.scales, "legs_energy_substeps", 0.) != 0. else ""),

                            # ("_trackingYaw{:.0e}".format(
                            #     -Go2MixCmdCfg.rewards.scales.tracking_yaw) if getattr(
                            #     Go2MixCmdCfg.rewards.scales, "tracking_yaw", 0.) != 0. else ""),
                            
                            # ("_trackingGoalVel{:.0e}".format(-Go2MixCmdCfg.rewards.scales.tracking_goal_vel) if getattr(Go2MixCmdCfg.rewards.scales, "tracking_goal_vel", 0.) != 0. else ""),
                            ("_trackingGoalPos{:.0e}".format(-Go2MixCmdCfg.rewards.scales.tracking_goal_pos) if getattr(Go2MixCmdCfg.rewards.scales, "tracking_goal_pos", 0.) != 0. else ""),
                            ("_gClose{:.0e}".format(-Go2MixCmdCfg.rewards.scales.gripper_close) if getattr(Go2MixCmdCfg.rewards.scales, "gripper_close", 0.) != 0. else ""),


                            # ("_rAngTrack{:.1f}".format(Go2MixCmdCfg.rewards.scales.tracking_ang_vel) if getattr(Go2MixCmdCfg.rewards.scales, "tracking_ang_vel", 0.) != 0. else ""),
                            # ("_rLinTrack{:.1f}".format(Go2MixCmdCfg.rewards.scales.tracking_lin_vel) if getattr(Go2MixCmdCfg.rewards.scales, "tracking_lin_vel", 0.) != 0. else ""),

                            # ("z_scale{:.2f}").format(Go2MixCmdCfg.terrain.TerrainPerlin_kwargs["zScale"]),
                            
                            # ("_pTorq" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.torques, trim= "-", exp_digits= 1) if getattr(Go2MixCmdCfg.rewards.scales, "torques", 0.) != 0. else ""),
                            # ("_pOrient" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.orientation, trim= "-", exp_digits= 1) if getattr(Go2MixCmdCfg.rewards.scales, "orientation", 0.) != 0. else ""),
                            # ("_pStdCtch" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.stand_still_catch, trim= "-", exp_digits= 1) if getattr(Go2MixCmdCfg.rewards.scales, "stand_still_catch", 0.) != 0. else ""),
                            # ("_pStd" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.stand_still, trim= "-", exp_digits= 1) if getattr(Go2MixCmdCfg.rewards.scales, "stand_still", 0.) != 0. else ""),
                            # ("_pLinVelZ" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.lin_vel_z, trim= "-", exp_digits= 1) if getattr(Go2MixCmdCfg.rewards.scales, "lin_vel_z", 0.) != 0. else ""),
                            # ("_pbaseH" + np.format_float_scientific(-Go2MixCmdCfg.rewards.scales.base_height, trim= "-", exp_digits= 1) if getattr(Go2MixCmdCfg.rewards.scales, "base_height", 0.) != 0. else ""),
                            
                            # ("_softDof{:.1f}".format(Go2MixCmdCfg.rewards.soft_dof_pos_limit) if Go2MixCmdCfg.rewards.soft_dof_pos_limit != 0.9 else ""),
                            # ("_kp{:d}".format(int(Go2MixCmdCfg.control.stiffness["joint"])) if Go2MixCmdCfg.control.stiffness["joint"] != 50 else ""),
                            # ("_kd{:.1f}".format(Go2MixCmdCfg.control.damping["joint"]) if Go2MixCmdCfg.control.damping["joint"] != 1. else ""),
                            # ("_noTanh"),
                            # ("_zeroResetAction" if Go2MixCmdCfg.init_state.zero_actions else ""),
                            # ("_EntropyCoef0.01"),
                            # ("_actionClip" + Go2MixCmdCfg.normalization.clip_actions_method if getattr(
                            #     Go2MixCmdCfg.normalization, "clip_actions_method", None) is not None else ""),
                            # ("_gravity" + str(Go2MixCmdCfg.sim.gravity[2]) if getattr(
                                # Go2MixCmdCfg.sim, "gravity", None) is not None else ""),
                            ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
                            ])
        max_iterations = 60000
        save_interval = 1000



