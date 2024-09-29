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
            Hip_max=1.047,
            Hip_min=-1.047,
            Thigh_max=2.966,
            Thigh_min=-0.663,
            Calf_max=-0.837,
            Calf_min=-2.721,
            Finger_min = -0.4,
            Finger_max = 0.4   
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


class Go2RoughCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        zero_actions = False
        default_joint_angles = JOINT_ANGLES

        # define the properties of the target ball to catch
        object_radius = 0.02
        object_pos_x = [0, 3]
        object_pos_y = [-1.5, 1.5]
        object_pos_z = [0.2, 0.3]

        object_pos_z_curriculum_increment = 0.02
        object_pos_z_max_curriculum_level = 15
        curriculum_length = 100


    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2g_description_vertical.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2g_description.urdf'
        target_object_file = '{LEGGED_GYM_ROOT_DIR}/resources/objects/hanging_ball.urdf'
        name = "go2"
        foot_name = "foot"
        sdk_dof_range = GO2_CONST_DOF_RANGE
        
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        density = 1000

        penalize_contacts_on = ["base", "thigh", "calf", "gripper_link", "l_finger_link", "r_finger_link"]
        # penalize_contacts_on = []
        # terminate_after_contacts_on = ["base", "imu", "gripper_link", "l_finger_link", "r_finger_link"]
        terminate_after_contacts_on = ["base", "imu"]
        # terminate_after_contacts_on = []
        front_hip_names = ["FR_hip_joint", "FL_hip_joint"]
        rear_hip_names = ["RR_hip_joint", "RL_hip_joint"]

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
            num_position_iterations = 16  
            num_velocity_iterations = 0 
            contact_offset = 0.01  # [m]
            rest_offset = 0.001   # [m]
            bounce_threshold_velocity = 0.1 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 9*1024**2 # max is 50*1024**2 ;2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 128
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class noise( LeggedRobotCfg.noise ):
        add_noise = True # disable internal uniform +- 1 noise, and no noise in proprioception
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            forward_depth = 0.1
            base_pose = 1.0
            goal = 0.1
            not_insight = 0.1
            dof_pos = 0.01
            dof_vel = 0.15
            ang_vel = 0.3
            imu = 0.2
            forward_camera = 0.02

    class viewer( LeggedRobotCfg.viewer ):
        pos = [9., 4., 0.6]
        lookat = [3., 4., 0.34]  # [m]
        move_pos = [2., 1., 0.7] # [m]
        draw_volume_sample_points = False
        debug_viz = True

    class termination:
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            # "out_of_track", # for barrier track
        ]

        roll_kwargs = dict(
            threshold= 2., # [rad]
            # tilt_threshold= 1.5,
            # jump_threshold= 1.5,
        )
        pitch_kwargs = dict(
            threshold= 2., # [rad] # for leap, jump
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


    class curriculum:
        no_moveup_when_fall = False
        # chosen heuristically, please refer to `LeggedRobotField._get_terrain_curriculum_move` with fixed body_measure_points


    class env(LeggedRobotCfg.env):
        num_envs = 5120 # try 5120
        episode_length_s = 20
        obs_components = [
            "proprioception",  # 54
            "target", # 6, including 1 redundant zeros
            # "robot_config" 
            # "height_measurements", # 187
            # "forward_depth",
            # "forward_color",
            # "base_pose",
            # "engaging_block",
            # "sidewall_distance",
        ]
        # privileged_obs_components = [
        #     "proprioception",
        #     "robot_config",x
        # ]
        use_lin_vel = False
        privileged_use_lin_vel = False
        num_actions = 14 # 1 dummy action


    class control( LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        torque_limits = [30., 30., 25.] * 4 + [30.]*2 
        stiffness = {'joint':40., 'gripper':10000}  # [N*m/rad]
        damping = {'joint': 0.6, 'gripper':2000}     # [N*m*s/rad]

        computer_clip_torque = False
        motor_clip_torque = False

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4


    class commands(LeggedRobotCfg.commands):
        num_commands = 4 # 0:grasp(+), 1:forward(+)/backward(-), 2:left(-)/right(+)
        heading_command = False
        resampling_time = 10 # [s]
        curriculum = False

        remote_control = False
        class ranges(LeggedRobotCfg.commands.ranges):
            grasp = [0., 1.]
            lin_vel_x = [-1, +1.]
            lin_vel_y = [-1., 1.]
            # ang_vel_yaw = [-1., 1.]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_com = True
        class com_range: # base mass center
            x = [-0.01, 0.2]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]

        randomize_motor = True
        leg_motor_strength_range = [0.8, 1.2]

        randomize_base_mass = True
        added_mass_range = [1.0, 3.0]
        
        randomize_friction = True
        friction_range = [0.5, 2.5]
        npc_friction_range = [0.5, 0.55]
        
        push_robots = True 
        max_push_vel_xy = 0.5 # [m/s]
        push_interval_s = 8

        init_base_pos_range = dict(
            x=[-0.1, 0.1],
            y=[-0.1, 0.1],
        )
        init_base_rot_range = dict(
            roll=[-0.4, 0.4],
            pitch=[-0.4, 0.4],
        )
        init_base_vel_range = dict(
            x=[-0.2, 0.8],
            y=[-0.2, 0.8],
            z=[-0.2, 0.2],
            roll=[-0.2,0.2],
            pitch=[-0.2,0.2],
            yaw=[-0.2,0.2]
        )

        init_dof_vel_range = [-5, 5]

    class terrain( LeggedRobotCfg.terrain):
        mesh_type = "trimesh" # Don't change. "none", "plane", "heightfield" or "trimesh"
        num_rows = 6
        num_cols = 6
        selected = "TerrainPerlin" # "BarrierTrack" or "TerrainPerlin"; "TerrainPerlin" can be used for training a walk policy.
        max_init_terrain_level = 0
        border_size = 5
        slope_treshold = 20.

        curriculum = False # for walk
        horizontal_scale = 0.025 # [m]
        # vertical_scale = 1. # [m]
        pad_unavailable_info = True
        measure_heights = True

        TerrainPerlin_kwargs = dict(
            zScale=0.06, 
            frequency=10,
        )


    class sensor:
        class forward_camera:
            resolution = [460, 840]
            ##### forward camera ########
            # position = [0.26, 0., 0.03] # position in base_link
            # rotation = [0., 0., 0.] # ZYX Euler angle in base_link

            ##### for realsense #########
            position = [0.34, 0., 0.16] # [0.285 0 0.01] + [0.055 0. 0.15]
            rotation = [0, 0.55, 0]
            
            latency_range = [0., 0.]
            refresh_duration = 1

        class proprioception:
            delay_action_obs = True
            latency_range = [0.02, 0.06] # TODO: check buffer length
            latency_resample_time = 2.0 # [s]

        class realsense:
            fov = [1.204, 0.733] # degree:[69., 42.]
            resolution = [1920., 1080.]
            refresh_duration = 0.01
            delay_action_obs = True
            latency_range = [0., 0.025]
            buffer_length = 5
            latency_resample_time = 2.0 # [s]
            orin = False
            

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
            ["Hip", "Thigh", "Calf"] * 4,
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

        clip_actions_low.append(GO2_CONST_DOF_RANGE["Finger_min"])
        clip_actions_low.append(GO2_CONST_DOF_RANGE["Finger_min"])
        clip_actions_high.append(GO2_CONST_DOF_RANGE["Finger_max"])
        clip_actions_high.append(GO2_CONST_DOF_RANGE["Finger_max"])


    class rewards(LeggedRobotCfg.rewards): # adjusted, can run
        class scales:
            ###### tracking rewards ######
            tracking_goal_vel = 0.5
            tracking_yaw = 0.3
            tracking_goal_pos = 0.5

            ###### gripper action rewards ######
            gripper_open = -0.15
            gripper_close = 1.0
            # in_sight = 0.1 # local opt

            ###### regularization rewards ######
            action_rate = -0.06 # no use [-0.1, -0.25, -0.5]
            ang_vel_xy = -0.01 
            # # # alive = 1
            collision = -0.1 # -0.5 not move, -0.1 collide, -0.05 ok
            delta_torques = -1.e-8
            delta_torques = -1.e-5

            dof_acc = -1.e-7 # penalize dof acc, -6 too large now.
            dof_error = -0.225 # penalize if far from default pos
            # dof_error = -0.25 # penalize if far from default pos
            # dof_error = -0.275 # penalize if far from default pos
            dof_vel = -1.e-5 # penalize dof velocity

            exceed_dof_pos_limits = -0.05
            exceed_dof_pos_limits = -0.01
            # exceed_torque_limits_l1norm = -0.005
            # exceed_torque_limits_square = -1.e-5
            exceed_torque_limits_i = -1.e-4
            exceed_torque_limits_i = -1.e-3
            exceed_torque_limits_i = -1.e-2
            # exceed_torque_limits_l1norm
            # hip_pos = -5.0e-4 # can jump, jump walk: -5.e-08, -5.e-07, -1.e-06; only walk -1e-04 ok
            # legs_energy_substeps = -7.e-07 # -1.e-07, -5.e-07, -1.e-06 ok , 1e-05 too small cannot move
            legs_energy_substeps = -1.e-06 # -1.e-07, -5.e-07, -1.e-06 ok , 1e-05 too small cannot move
            # legs_energy_substeps = -7.e-07 # -1.e-07, -5.e-07, -1.e-06 ok , 1e-05 too small cannot move
            stand_still = -0.1 # based on cmds
            torques = -7.0e-5 # should be < -1.e-4
            # torques = -9.0e-5 # should be < -1.e-4
            torques = -1.0e-4 # should be < -1.e-4
            torques = -5.0e-4 # should be < -1.e-4
            # torques = -1.0e-3 # should be < -1.e-4
            stand_still_catch = -0.05
            # feet_air_time = -0.1
            lin_vel_z = -1.
            # lin_vel_z = -0.5 
            # lin_vel_z = -0.25
            # orientation = -1. 
            # orientation = -0.5
            # feet_stumble = -1.
            # feet_edge = -0.1
            
        action_rate_th = 0.1
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_goal_pos = 0.5 # more smooth
        soft_dof_pos_limit = 0.7 # percentage of urdf limits, values above this limit are penalized, not smaller than 0.6
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.7
        base_height_target = 1.
        max_contact_force = 50. # forces above this value are penalized
        
        ###### grasp threshold ######
        xy_vel_norm_threshold = 0.1 # tracking goal vel of x and y
        xy_pos_norm_threshold = xy_vel_norm_threshold
        grasp_norm_threshold = 0.05 # when to close gripper
        pos_norm_threshold = 0.05 # when to start apply force and give pos bool rew, cannot smaller than 0.05
        no_hip_pos_penalty_while_jump = False # Don't penalize hip pos while close object and try to jump


class Go2RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.2

    class policy( LeggedRobotCfgPPO.policy ):
        rnn_type = 'gru'
        # mu_activation = "tanh"
        mu_activation = None
        rnn_hidden_size = 256
        rnn_num_layers = 2

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        run_name = 'simple'
        experiment_name = 'go2'
        resume = True
        load_run = None
        load_run = "Apr01_23-11-08__pDofErr2e-01_pEngSub7e-07_fromMar26_22-36-09"
        # load_run = "Mar26_22-36-09__pEngSub1e-08_pDofErr5e-02_pDofAcc1e-09_pclls5e-02_pTorq2e-7_pActRate1e-3_pExcdDof1e-4_pStd3e-2_fromMar26_15-59-58"
        # load_run = "Mar28_22-26-18_no_crcl_pEngSub5e-07_pDofErr2e-01_pDofAcc1e-08_pDofVel1e-08_pclls3e-01z_scale0.06_pTorq5e-6_pActRate5e-2_pExcdDof1e-3_pExcdTorqL11e-4_pStd5e-2_noResume"
        # load_run = "Mar27_22-50-19_no_crcl_pEngSub8e-07_pDofErr2e-01_pDofAcc1e-07_pclls3e-01_pTorq1e-4_pActRate1e-1_pExcdDof1e-3_pExcdTorqL11e-4_pStd5e-2_fromMar27_17-26-46" # cannot stand
        # load_run = "Mar27_17-26-46_no_crcl_pEngSub8e-07_pDofErr2e-01_pDofAcc1e-07_pclls4e-01_pTorq5e-6_pActRate5e-2_pExcdDof1e-3_pStd2e-2_fromMar25_15-05-28"
        # load_run = "Mar26_15-59-27__pEngSub1e-08_pDofErr1e-01_pDofAcc1e-09_pclls2e-01_pTorq2e-6_pActRate1e-3_pExcdDof1e-3_pStd3e-2_noResume"
        # load_run = "Mar25_15-05-28_no_crcl_pEngSub1e-07_pDofErr2e-01_pDofAcc1e-09_pclls4e-01_pTorq2e-6_pActRate1e-3_pExcdDof1e-3_pStd2e-2_noResume"
        # load_run = "Mar24_13-41-36_no_crcl_pEngSub1e-07_pDofErr1e-03_noResume"
        # load_run = "Mar22_23-22-53__pEngSub5e-07_pDofAcc1e-10_pDofVel5e-08_pclls1e-02_pTorq1e-4_pActRate1e-2_noResume"
        # load_run = "Mar19_08-15-55_Grasp_pEnergySubsteps1e-06_pDeltaTorq1e-5_pTorq1e-4_pActRate1e-3_pExceedDof1e-2_pExceedTorqueL11e-3_pStand1e-2_softDof0.7_fromMar18_15-31-52"
        # load_run = "Mar20_22-25-04__pDofAcc1e-07_pDofVel5e-06_pclls1e-01_pExcdTorqL11e-4_fromMar19_22-15-24" # sqz
        # load_run = "Mar19_14-14-56_Grasp_softDof0.7_fromJan30_17-19-49"
        # load_run = "Mar18_15-31-52_Grasp_pEnergySubsteps1e-06_pDeltaTorq1e-4_pTorq1e-4_pActRate1e-1_pExceedDof1e-1_pExceedTorqueL11e-3_pStand1e-1_softDof0.6_fromMar18_13-59-27"
        # load_run = "Mar17_15-30-28_Grasp_pEnergySubsteps1e-06_pDeltaTorq1e-4_pTorq1e-4_pActRate2e-1_pExceedDof1e-1_pExceedTorqueL11e-3_pStand1e-1_softDof0.6_fromMar12_07-47-59"
        # load_run = "Mar14_20-32-17_Grasp_pEnergySubsteps8e-06_pDeltaTorq1e-4_pTorq1e-4_pActRate2.8e-1_pExceedDof1e-2_pExceedTorqueL11e-3_pStand1e-3_softDof0.6_fromMar14_17-22-21"
        # load_run = "Mar14_17-22-21_Grasp_pEnergySubsteps5e-06_pDeltaTorq1e-4_pTorq1e-5_pActRate2.7e-1_pExceedDof1e-2_pExceedTorqueL11e-3_pStand1e-3_softDof0.6_fromMar14_09-27-44"
        # load_run = "Mar14_09-27-44_Grasp_pEnergySubsteps4e-06_pDeltaTorq1e-4_pTorq1e-5_pActRate2.7e-1_pExceedDof1e-2_pExceedTorqueL11e-3_pStand1e-2_softDof0.6_fromMar13_15-22-50"
        # load_run = "Mar14_11-12-23_Grasp_pEnergySubsteps4e-06_pDeltaTorq1e-4_pTorq1e-5_pActRate2.55e-1_pExceedDof1e-2_pExceedTorqueL11e-3_pStand1e-3_softDof0.6_fromMar13_15-22-50"
        # load_run = "Mar13_15-22-50_Grasp_pEnergySubsteps5e-06_pDofErr3e-1_pHipPos1e-4_pExceedDof1e-1_pExceedTorqueL11e-3_pStand1e-1_pActRate2.5e-1_softDof0.6_fromMar12_07-47-59"
        # load_run = "Mar12_07-47-59_Grasp_pEnergySubsteps5e-06_pDofErr3e-1_pHipPos1e-3_pExceedDof1e-1_pExceedTorqueL11e-3_pStand1e-1_pActRate2e-1_softDof0.6_fromMar09_17-08-05"
        # load_run = "Mar12_07-47-20_Grasp_pEnergySubsteps5e-06_pDofErr3e-1_pHipPos1e-3_pExceedDof1e-1_pExceedTorqueL11e-3_pStand1e-1_pActRate2e-1_softDof0.6_fromMar09_17-08-05"
        # load_run = "Mar10_08-59-34_Grasp_pEnergySubsteps2e-05_pDofErr3e-1_pHipPos1e-3_pExceedDof1e-1_pExceedTorqueL11e-3_pStand2e-1_pActRate2.5e-1_softDof0.6_fromMar09_23-02-09"
        # load_run = "Mar09_23-02-09_Grasp_pEnergySubsteps2e-05_pDofErr3e-1_pHipPos1e-3_pExceedDof1e-1_pExceedTorqueL11e-3_pStand2e-1_pActRate2.5e-1_softDof0.6_fromMar09_15-32-00"
        # load_run = "Mar09_15-32-00_Grasp_pEnergySubsteps1e-05_pDofErr3e-1_pHipPos1e-3_pExceedDof1e-1_pExceedTorqueL11e-3_pStand2e-1_pActRate2e-1_softDof0.6_fromMar08_12-47-09"
        # load_run = "Mar09_17-08-05_Grasp_pEnergySubsteps5e-06_pDofErr3e-1_pHipPos1e-3_pExceedDof1e-1_pExceedTorqueL11e-3_pStand1e-1_pActRate2e-1_softDof0.6_fromMar09_15-25-00"
        # load_run = "Mar09_15-25-00_Grasp_pEnergySubsteps1e-06_pDofErr2.5e-1_pHipPos1e-3_pExceedDof1e-1_pExceedTorqueL11e-3_pStand2e-1_pActRate1e-1_softDof0.6_fromMar08_12-47-09"
        # load_run = "Mar08_12-47-09_Grasp_pEnergySubsteps5e-07_pDofErr1e-1_pHipPos1e-3_pExceedDof1e-1_pExceedTorqueL11e-3_pStand1e-1_pActRate1e-2_softDof0.7_fromMar07_14-43-36"
        # load_run = "Mar07_14-43-36_Grasp_pEnergySubsteps1e-06_pDofErr1e-3_pHipPos1e-3_pExceedDof1e-1_pExceedTorqueL11e-4_pActRate1e-2_softDof0.7_fromJan30_17-19-49"
        # load_run = "Mar02_18-28-40_Grasp_pEnergySubsteps1e-06_pDofErr2e-1_pHipPos1e-4_pExceedDof1e-1_softDof0.7_fromFeb22_22-42-49"
        # load_run = "Feb22_22-42-49_Grasp_pEnergySubsteps1e-07_pDofErr1e-1_pCollide1e-2_pHipPos1e-7_softDof1.0_kp40_actionCliphard_gravity-9.81_fromJan30_17-19-49"
        # load_run = "Feb23_15-42-16_Grasp_pEnergySubsteps1e-06_pDofErr2e-1_pCollide5e-2_pHipPos1e-7_softDof1.0_kp40_actionCliphard_gravity-9.81_fromFeb23_10-29-06"
        # load_run = "Feb23_10-29-06_Grasp_pEnergySubsteps1e-06_pDofErr2e-1_pCollide2e-2_pHipPos1e-7_softDof1.0_kp40_actionCliphard_gravity-9.81_fromJan30_17-19-49"
        # load_run = "Feb22_22-43-59_Grasp_pEnergySubsteps5e-07_pDofErr1e-1_pCollide1e-2_pHipPos1e-7_softDof1.0_kp40_actionCliphard_gravity-9.81_fromJan30_17-19-49"
        # load_run = "../go2/Feb21_18-03-16_Grasp_trackingGoalVel-9e-01_trackingYaw-4e-01_pEnergySubsteps5e-07_rAlive1.0z_scale0.05_pCollide5e-2_softDof0.8_kp40_noTanh_EntropyCoef0.01_actionCliphard_gravity-9.81_fromFeb19_11-40-52"
        # load_run = "Feb19_11-40-52_Grasp_trackingGoalVel-9e-01_trackingYaw-4e-01_pEnergySubsteps1e-15_rAlive1.0z_scale0.05_softDof1.0_kp40_noTanh_EntropyCoef0.01_actionCliphard_gravity-9.81_fromJan30_17-19-49"
        # load_run = "Jan30_17-19-49_Grasp_trackingGoalVel-6e-01_trackingYaw-4e-01_rAlive1.0z_scale0.05_pDofAcc2.5e-9_pCollide1e-4_softDof1.0_kp40_noTanh_EntropyCoef0.01_actionCliphard_gravity-9.81_fromJan29_23-40-34"
        # load_run = "Jan29_23-40-34_Grasp_trackingGoalVel-5e-01_trackingYaw-2e-01_rAlive1.0z_scale0.05_softDof1.0_kp40_noTanh_EntropyCoef0.01_actionCliphard_gravity-9.81_fromJan29_21-33-57"
        # load_run = "Jan29_21-33-57_Grasp_trackingGoalVel-5e-01_trackingYaw-2e-01_rAlive1.0z_scale0.05_softDof1.0_kp40_noTanh_EntropyCoef0.01_actionCliphard_gravity-9.81_fromJan10_17-54-59"
        # load_run = "Jan10_17-54-59_Grasp_trackingGoalVel-1e+00_trackingYaw-5e-01_pEnergySubsteps6e-06_rAlive1.0z_scale0.10_pStand1e-2_pActRate1e-2_softDof1.0_kp20_kd0.5_noTanh_EntropyCoef0.01_actionCliphard_fromJan08_12-16-42"
        # load_run = "Jan08_12-16-42_Grasp_trackingGoalVel-1e+00_trackingYaw-5e-01_pEnergySubsteps6e-06_rAlive1.0z_scale0.10_pCollide5e-1_pStand1e-2_pActRate1e-2_softDof1.0_kp20_kd0.5_noTanh_EntropyCoef0.01_actionCliphard_fromJan04_16-11-54"
        # load_run = "Jan04_16-11-54_Catch_trackingGoalVel-7e-01_trackingYaw-5e-01_pEnergySubsteps6e-06_rAlive1.0z_scale0.10_pCollide5e-1_pStand1e-2_pActRate1e-2_softDof1.0_kp20_kd0.5_noTanh_EntropyCoef0.01_actionCliphard_fromJan03_17-55-00"
        # load_run = "Jan03_17-55-00_Catch_trackingGoalVel-7e-01_trackingYaw-5e-01_pEnergySubsteps6e-06_rAlive1.0z_scale0.10_pCollide5e-1_pStand1e-2_pActRate1e-2_softDof1.0_kp20_kd0.5_noTanh_EntropyCoef0.01_actionCliphard_noResume"
        # load_run = "Jan02_20-12-13_Catch_pEnergySubsteps6e-07_trackingGoalVel-1e+00_trackingYaw-1e+00_rAlive1.0z_scale0.10_pCollide5e-1_softDof0.0_actionCliphard_fromJan02_18-38-15"
        # load_run = "Dec28_10-14-58_WalkByRemote_pEnergySubsteps6e-06_rAlive1.0z_scale0.10_softDof0.0_actionCliphard_fromDec25_23-15-30"
        # load_run = "Dec25_23-15-30_WalkByRemote_pEnergySubsteps6e-08_rAlive0.1z_scale0.10_softDof0.0_actionCliphard_noResume"
        # load_run = "Dec12_13-23-19_WalkByRemote_pEnergySubsteps6e-08_rAlive0.1z_scale0.10_softDof0.0_actionCliphard_noResume"
        # load_run = "Dec10_19-32-25_WalkByRemote_pEnergySubsteps6e-09_rAlive0.1z_scale0.10_softDof0.0_actionCliphard_fromDec09_22-11-49"
        # load_run = "Dec09_22-11-49_WalkByRemote_pEnergySubsteps6e-08z_scale0.10_softDof0.0_actionCliphard_noResume"
        # load_run = "Dec08_13-15-08_WalkByRemote_pEnergySubsteps6e-08z_scale0.10_softDof0.0_actionCliphard_fromDec08_12-38-51"
        run_name = "".join([
                            # ("friction" + str(Go2RoughCfg.domain_rand.friction_range)),
                            ("_crcl" + str(Go2RoughCfg.commands.curriculum) if getattr(Go2RoughCfg.commands, "curriculum", False) else ""),
                            ("_onlyPstRew" if getattr(Go2RoughCfg.rewards, "only_positive_rewards", False) else ""),

                            # ("_pAngVelXy" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.ang_vel_xy,trim="-", exp_digits=1) if getattr(Go2RoughCfg.rewards.scales, "ang_vel_xy", 0.) != 0. else ""),
                            # ("_pActRate" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.action_rate,trim="-", exp_digits=1) if getattr(Go2RoughCfg.rewards.scales, "action_rate", 0.) != 0. else ""),
                            # ("_rAlive{:.1f}".format(Go2RoughCfg.rewards.scales.alive) if getattr(Go2RoughCfg.rewards.scales, "alive", 0.) != 0. else ""),
                            # ("_pclls{:.0e}".format(-Go2RoughCfg.rewards.scales.collision) if getattr(Go2RoughCfg.rewards.scales, "collision", 0.) != 0. else ""),
                            # ("_pDeltaTorq" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.delta_torques, trim= "-", exp_digits= 1) if getattr(Go2RoughCfg.rewards.scales, "delta_torques", 0.) != 0. else ""),
                            # ("_pDofAcc{:.0e}".format(-Go2RoughCfg.rewards.scales.dof_acc) if getattr(Go2RoughCfg.rewards.scales, "dof_acc", 0.) != 0. else ""),
                            # ("_pDofErr{:.0e}".format(-Go2RoughCfg.rewards.scales.dof_error) if getattr(Go2RoughCfg.rewards.scales, "dof_error", 0.) != 0. else ""),
                            # ("_pDofVel{:.0e}".format(-Go2RoughCfg.rewards.scales.dof_vel) if getattr(Go2RoughCfg.rewards.scales, "dof_vel", 0.) != 0. else ""),

                            # ("_pExcdDofPos" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.exceed_dof_pos_limits, trim= "-", exp_digits= 1) if getattr(Go2RoughCfg.rewards.scales, "exceed_dof_pos_limits", 0.) != 0. else ""),
                            # ("_pExcdTorqL1" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.exceed_torque_limits_l1norm, trim="-",exp_digits=1) if getattr(Go2RoughCfg.rewards.scales, "exceed_torque_limits_l1norm",0.) != 0. else ""),
                            # ("_pExcdTorqSq" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.exceed_torque_limits_square, trim="-",exp_digits=1) if getattr(Go2RoughCfg.rewards.scales, "exceed_torque_limits_square",0.) != 0. else ""),
                            # ("_pExcdTorq_i" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.exceed_torque_limits_i, trim="-",exp_digits=1) if getattr(Go2RoughCfg.rewards.scales, "exceed_torque_limits_i",0.) != 0. else ""),

                            # ("_pHipPos" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.hip_pos, trim="-",exp_digits=1) if getattr(Go2RoughCfg.rewards.scales, "hip_pos",0.) != 0. else ""),
                            # ("_pEngSub{:.0e}".format(-Go2RoughCfg.rewards.scales.legs_energy_substeps) if getattr(Go2RoughCfg.rewards.scales, "legs_energy_substeps", 0.) != 0. else ""),

                            # ("_trackingYaw{:.0e}".format(
                            #     -Go2RoughCfg.rewards.scales.tracking_yaw) if getattr(
                            #     Go2RoughCfg.rewards.scales, "tracking_yaw", 0.) != 0. else ""),
                            
                            # ("_trackingGoalVel{:.0e}".format(
                            #     -Go2RoughCfg.rewards.scales.tracking_goal_vel) if getattr(
                            #     Go2RoughCfg.rewards.scales, "tracking_goal_vel", 0.) != 0. else ""),
                            
                            
                            # tracking_ang_vel 
                            # ("_rAngTrack{:.1f}".format(Go2RoughCfg.rewards.scales.tracking_ang_vel) if getattr(
                            #     Go2RoughCfg.rewards.scales, "tracking_ang_vel", 0.) != 0. else ""),
                            # ("_rLinTrack{:.1f}".format(Go2RoughCfg.rewards.scales.tracking_lin_vel) if getattr(
                            #     Go2RoughCfg.rewards.scales, "tracking_lin_vel", 0.) != 0. else ""),
                            # ("_pLinVelL2{:.1f}".format(-Go2RoughCfg.rewards.scales.lin_vel_l2norm) if getattr(
                            #     Go2RoughCfg.rewards.scales, "lin_vel_l2norm", 0.) != 0. else ""),
                            
                            # ("z_scale{:.2f}").format(Go2RoughCfg.terrain.TerrainPerlin_kwargs["zScale"]),
                            
                            ("_pTorq" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.torques, trim= "-", exp_digits= 1) if getattr(Go2RoughCfg.rewards.scales, "torques", 0.) != 0. else ""),
                            # ("_pOrient" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.orientation, trim= "-", exp_digits= 1) if getattr(Go2RoughCfg.rewards.scales, "orientation", 0.) != 0. else ""),
                            # ("_pStdCtch" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.stand_still_catch, trim= "-", exp_digits= 1) if getattr(Go2RoughCfg.rewards.scales, "stand_still_catch", 0.) != 0. else ""),
                            # ("_pStd" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.stand_still, trim= "-", exp_digits= 1) if getattr(Go2RoughCfg.rewards.scales, "stand_still", 0.) != 0. else ""),
                            # ("_pLinVelZ" + np.format_float_scientific(-Go2RoughCfg.rewards.scales.lin_vel_z, trim= "-", exp_digits= 1) if getattr(Go2RoughCfg.rewards.scales, "lin_vel_z", 0.) != 0. else ""),
                            
                            # ("_softDof{:.1f}".format(Go2RoughCfg.rewards.soft_dof_pos_limit) if Go2RoughCfg.rewards.soft_dof_pos_limit != 0.9 else ""),
                            # ("_kp{:d}".format(int(Go2RoughCfg.control.stiffness["joint"])) if Go2RoughCfg.control.stiffness["joint"] != 50 else ""),
                            # ("_kd{:.1f}".format(Go2RoughCfg.control.damping["joint"]) if Go2RoughCfg.control.damping["joint"] != 1. else ""),
                            # ("_noTanh"),
                            # ("_zeroResetAction" if Go2RoughCfg.init_state.zero_actions else ""),
                            # ("_EntropyCoef0.01"),
                            # ("_actionClip" + Go2RoughCfg.normalization.clip_actions_method if getattr(
                            #     Go2RoughCfg.normalization, "clip_actions_method", None) is not None else ""),
                            # ("_gravity" + str(Go2RoughCfg.sim.gravity[2]) if getattr(
                                # Go2RoughCfg.sim, "gravity", None) is not None else ""),
                            ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
                            ])
        max_iterations = 20000
        save_interval = 500



