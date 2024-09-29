import os
import sys
import time
import json
import atexit
from datetime import datetime
import os.path as osp
from collections import OrderedDict

# import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from unitree_go.msg import (
    WirelessController,
    LowState,
    MotorState,
    IMUState,
    LowCmd,
    MotorCmd,
)

sys.path.append(osp.dirname(osp.abspath(__file__)))
from crc_module import get_crc
from hardware import Hardware, train_cfg_dict 
from deploy_node import *

sys.path.append('../../legged_gym')
# from utils import messageLogger
from rsl_rl.runners import build_runner

REAL_TO_SIM = True
LEG_DOF = 12
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

@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


class DeployTwoPolicyNode(Node):
    def __init__(self, load_run, checkpoint = -1):
        super().__init__("deploy_node")
        # init logger
        self.load_run = load_run
        self.checkpoint = checkpoint

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self._init_config() # Load config from json file.
        self._init_properties()
        self._init_subscribers()
        self._init_publisher()
        self._init_policy()

        # TODO: ROS2 bag record messages
        # ROSDataLogger
        file_pth = os.path.dirname(os.path.realpath(__file__))
        if REAL_TO_SIM:
            print(os.getcwd())
            log_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_path = os.path.join("./logs", log_path)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            self.sim2real_file = open(os.path.join(log_path, load_run["catch"] + '_obs.txt'), 'w')
            self.sim2real_file_actions = open(os.path.join(log_path, load_run["catch"] + 'actions.txt'), 'w')

        # start
        self.start_time = time.monotonic()
        self.get_logger().info("Press L2 to start policy")
        self.get_logger().info("Press L1 for emergent stop")

    def _init_config(self):
        checkpoint_path = osp.join("models", self.load_run["catch"])
        with open(osp.join(checkpoint_path, "config.json"), "r") as f:
            config_dict = json.load(f, object_pairs_hook= OrderedDict)
        self.train_cfg_dict_catch = config_dict

        self.max_kp = 40. #config_dict['control']['stiffness']['joint'] # 
        self.clip_actions_high = torch.tensor(config_dict['normalization']['clip_actions_high'], device=self.device)
        self.clip_actions_low = torch.tensor(config_dict['normalization']['clip_actions_low'], device=self.device)
        self.commands_ranges = config_dict['commands']['ranges']
        self.num_commands = config_dict['commands']['num_commands']
        self.num_state_chunck = config_dict['env']['num_state_chunck']
        self.num_obs = config_dict['env']['num_observations']
        self.num_obs = 56*self.num_state_chunck # TODO
        self.num_actions = config_dict['env']['num_actions']

        self.obs_scales = config_dict['normalization']['obs_scales']
        self.commands_scale = torch.ones(self.num_commands, device=self.device, requires_grad=False)

        # For build runner
        self.num_privileged_obs = None # Need to set None.
        self.num_envs = 1

        # walk policy config
        checkpoint_path = osp.join("models", self.load_run["walk"])
        with open(osp.join(checkpoint_path, "config.json"), "r") as f:
            config_dict = json.load(f, object_pairs_hook= OrderedDict)
        
        self.train_cfg_dict_walk = config_dict

    def _init_properties(self):
        self.dof_map = [ # from isaacgym simulation joint order to URDF order
                3, 4, 5,
                0, 1, 2,
                9, 10,11,
                6, 7, 8,
            ]
        # urdf order
        self.dof_names = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',\
                    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
        self.default_joint_angles_tensor = torch.zeros(LEG_DOF, device=self.device)
        self.default_joint_angles_tensor_all = torch.zeros(self.num_actions, device=self.device)
        self.dof_ranges = {}
        for i in range(LEG_DOF):
            name = self.dof_names[i]
            angle = self.train_cfg_dict_catch['init_state']['default_joint_angles'][name]
            self.default_joint_angles_tensor[i] = angle
            self.default_joint_angles_tensor_all[i] = angle
            type = name.split('_')[1]
            self.dof_ranges[name] = [GO2_CONST_DOF_RANGE[type+'_min'], GO2_CONST_DOF_RANGE[type+'_max']]
        # self.dof_ranges['RL_thigh_joint'][1] = RARE_THIGH_MAX
        # self.dof_ranges['RR_thigh_joint'][1] = RARE_THIGH_MAX
        
        self.projected_gravity = torch.zeros((1, 3), dtype= torch.float32, device=self.device)
        self.projected_gravity[:, 2] = -1

        self.joint_pos = torch.zeros(self.num_actions, device=self.device)
        self.joint_vel = torch.zeros(self.num_actions, device=self.device)
        self.joint_torq = torch.zeros(self.num_actions, device=self.device)
        self.kp = 0.
        self.kd = 0.

        self.goal_position_rel = torch.zeros(3, device=self.device)
        self.goal_rel_robot = torch.zeros_like(self.goal_position_rel)
        self.target_coordinate = torch.zeros_like(self.goal_position_rel)
        self.goal_yaw = torch.zeros(1, device=self.device)
        self.object_confidence = torch.zeros(1, device=self.device)
        self.commands = torch.zeros(self.num_commands, dtype=torch.float, device=self.device, requires_grad=False) 
        self.obs = torch.zeros(self.num_obs, device=self.device)
        self.obs_buf = torch.zeros(self.num_obs, dtype=torch.float, device=self.device, requires_grad=False)
        self.angles = torch.zeros(self.num_actions, device=self.device)
        self.prev_action = torch.zeros(self.num_actions, device=self.device)

        self.start_policy = False
        self.policy_refresh_time = 0.02

        self.sim_obs_buf = None
        self.camera_init = False

    def _init_subscribers(self):
        self.joy_stick_sub = self.create_subscription(WirelessController, "/wirelesscontroller", self.joy_stick_cb, 1)
        self.lowlevel_state_sub = self.create_subscription(LowState, "/lowstate", self.lowlevel_state_cb, 1)
        self.camera_sub = self.create_subscription(Float32MultiArray, "/camera", self.camera_cb, 2) 
        self.sim_obs_sub = self.create_subscription(Float32MultiArray, "/sim_logs/obs", self.sim_obs_cb, 2) 

    def _init_publisher(self):
        # Dog leg motor control
        self.motor_pub = self.create_publisher(LowCmd, "/lowcmd", 1)
        self.cmd_msg = LowCmd()
        self.low_state = LowState()
        self.motor_cmd = [
            MotorCmd(q=0., dq=0., tau=0.0, kp=0.0, kd=0.0, mode=0x01)
            for _ in range(20)
        ]
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

        # Log info
        self.obs_log = self.create_publisher(Float32MultiArray, "/logs/obs", 1)
        self.deploy_config_log = self.create_publisher(Float32MultiArray, "/logs/deploy_config", 1)

    def _init_policy(self):
        # file_pth = os.path.dirname(os.path.realpath(__file__))
        self.runner_catch = build_runner(
            "OnPolicyRunner",
            self, # env
            self.train_cfg_dict_catch,
            "logs",
            device=self.device,
        )

        load_run = os.path.join("models", self.load_run["catch"])
        checkpoint = self.checkpoint
        if checkpoint==-1:
            models = [file for file in os.listdir(load_run) if 'model' in file]
            models.sort(key=lambda m: '{0:0>15}'.format(m))
            model = models[-1]
        else:
            model = "model_{}.pt".format(checkpoint) 
        
        load_path = os.path.join(load_run, model)

        checkpoint = torch.load(load_path, map_location = self.device)

        self.runner_catch.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        
        self.agent_model_catch = self.runner_catch.alg.actor_critic
        self.agent_model_catch.to(self.device)
        self.policy_catch = self.runner_catch.get_inference_policy(device=self.device)
        print(f"Using device: {self.device}")
        
        _ = self.policy_catch(self.obs.detach().unsqueeze(0))  # first inference takes longer time


        ############# walk policy #################
        self.runner_walk = build_runner(
            "OnPolicyRunner",
            self, # env
            self.train_cfg_dict_walk,
            "logs",
            device=self.device,
        )

        load_run = os.path.join("models", self.load_run["walk"])
        checkpoint = self.checkpoint
        if checkpoint==-1:
            models = [file for file in os.listdir(load_run) if 'model' in file]
            models.sort(key=lambda m: '{0:0>15}'.format(m))
            model = models[-1]
        else:
            model = "model_{}.pt".format(checkpoint) 
        
        load_path = os.path.join(load_run, model)

        checkpoint = torch.load(load_path, map_location = self.device)

        self.runner_walk.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        
        self.agent_model_walk = self.runner_walk.alg.actor_critic
        self.agent_model_walk.to(self.device)
        self.policy_walk = self.runner_walk.get_inference_policy(device=self.device)
        print(f"Using device: {self.device}")
        
        _ = self.policy_walk(self.obs.squeeze()[:int(self.num_obs/self.num_state_chunck)].detach())  # first inference takes longer time

        self.policy = self.policy_walk
        self.policy_name = "walk"

    ########################################################
    ################## Callback Functions ##################
    ########################################################
    def camera_cb(self, msg):
        self.target_coordinate[0] = msg.data[0] 
        self.target_coordinate[1] = msg.data[1] 
        self.target_coordinate[2] = msg.data[2] 
        # print("Received:", self.target_coordinate)
        if not torch.isnan(self.target_coordinate).any():
            if torch.abs(self.target_coordinate[0] - self.goal_position_rel[0]) < 0.6 or (not self.camera_init):
                self.goal_position_rel[:] = self.target_coordinate[:]
                self.object_confidence[:] = 1.
                self.camera_init = True

    def joy_stick_cb(self, msg):        
        if msg.keys == 2:  # L1: emergency stop
            print("L2 pressed.")
            self.emergency_stop()

        if msg.keys == 32:  # L2: start policy
            if self.stand_up and not self.start_policy:
                self.get_logger().info("Start policy")
                self.start_policy = True
                 
            else:
                print("Wait for standing up first")

        cmd_grasp = self.commands[0] # comment this if want to start catch without press any button
        if msg.keys == 1: 
            # cmd_grasp = min(cmd_grasp-0.1, self.commands_ranges['grasp'][0])
            cmd_grasp = self.commands_ranges['grasp'][0]
            self.agent_model_catch.reset()
            self.agent_model_walk.reset()
            self.policy = self.policy_walk
            self.policy_name = "walk"
            self.goal_position_rel[:] = 0.
            

        if msg.keys == 16:  # R2: increase(+) tracking goal velocity
            # cmd_grasp = max(cmd_grasp+0.1, self.commands_ranges['grasp'][1])
            cmd_grasp = self.commands_ranges['grasp'][1] 
            self.agent_model_catch.reset()
            self.agent_model_walk.reset()
            self.policy = self.policy_catch
            self.policy_name = "catch"

        # Keep cmd_yaw as 0 when the joystick value is low, 
        # and gradually increase cmd_yaw in a linear manner as the joystick value increases.
        if msg.ly > 0.1: # Left y: forward
            cmd_vx = 1.5 * msg.ly
        elif msg.ly < -0.1:
            cmd_vx = 1. * msg.ly
        else:
            cmd_vx = 0.

        if msg.lx > 0.1: # 
            cmd_vy = -1.5 * msg.lx
        elif msg.lx < -0.1:
            cmd_vy = -1.5 * msg.lx
        else:
            cmd_vy = 0.

        if msg.rx > 0.1:
            cmd_yaw = -1.5 * msg.rx
        elif msg.rx < -0.1:
            cmd_yaw = -1.5 * msg.rx
        else:
            cmd_yaw = 0.

        if msg.ry > 0.1:
            cmd_pitch = 1.5 * msg.ry
        elif msg.ry < -0.1:
            cmd_pitch = 1.5 * msg.ry
        else:
            cmd_pitch = 0.

        # cmd_grasp = self.commands_ranges['grasp'][1]
        self.commands[0] = cmd_grasp
        self.commands[1] = cmd_vx
        self.commands[2] = cmd_vy
        self.commands[3] = cmd_yaw
        self.commands[4] = cmd_pitch

    def lowlevel_state_cb(self, msg: LowState):
        # imu data
        imu_data = msg.imu_state
        self.roll, self.pitch, self.yaw = imu_data.rpy
        self.obs_ang_vel = torch.tensor(imu_data.gyroscope, device=self.device)
        # self.obs_imu = torch.tensor([self.roll, self.pitch])

        self.base_quat = torch.tensor(imu_data.quaternion, device=self.device)
        self.projected_gravity_obs = quat_rotate_inverse(
            self.base_quat.unsqueeze(0), # batch function
            self.projected_gravity,
        )
        self.projected_gravity_obs = self.projected_gravity_obs.squeeze()

        # motor data
        for i in range(LEG_DOF):
            self.joint_pos[i] = msg.motor_state[self.dof_map[i]].q
            self.joint_vel[i] = msg.motor_state[self.dof_map[i]].dq
            self.joint_torq[i] = msg.motor_state[self.dof_map[i]].tau_est

        # Emergency stop if the joint velocity is too large. Order not matter here with velocity.
        if self.out_of_limit_vel() or self.out_of_limit_torque():
            self.emergency_stop()

    def sim_obs_cb(self, msg:Float32MultiArray):
        self.sim_obs_buf = msg.data
        
        
    ########################################################
    ################## Publish Functions ###################
    ########################################################

    def set_gains(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd
        for i in range(LEG_DOF):
            self.motor_cmd[i].kp = kp
            self.motor_cmd[i].kd = kd

    def set_motor_position(self, q: torch.Tensor):
        # Only control 12 leg motors.
        for i in range(LEG_DOF):
            idx = self.dof_map[i]
            self.motor_cmd[idx].q = float(q[i])
            self.motor_cmd[idx].dq = 0.
            self.motor_cmd[idx].tau = 0.
            self.motor_cmd[idx].kp = self.kp
            self.motor_cmd[idx].kd = self.kd
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.cmd_msg.crc = get_crc(self.cmd_msg)
        self.motor_pub.publish(self.cmd_msg)

    ########################################################
    #################### Help Functions ####################
    ########################################################
    def reset(self):
        # For build runner
        return None, None

    def emergency_stop(self):
        print("Emergency stop")
        self.set_gains(0.0,0.5)
        self.motor_pub.publish(self.cmd_msg)
        # self.logger.plot_states()
        raise SystemExit

    def exit_handler(self):
        print("exit")
        self.emergency_stop()

    def out_of_limit_torque(self):
        out_of_limit = False
        torques = self.joint_torq
        for idx, dof_name in enumerate(self.dof_names):
            min_val, max_val = -55., 55. # TODO
            if min_val > torques[idx] or max_val < torques[idx]:
                print(f"Out of limit **torque**: {dof_name} {torques[idx]}, which should in [{min_val}, {max_val}]")
                out_of_limit = True
            torques[idx] = float(torch.clip(torques[idx], min_val, max_val))
        return out_of_limit

    def out_of_limit_vel(self):
        out_of_limit = False
        velocities = self.joint_vel
        for idx, dof_name in enumerate(self.dof_names):
            min_val, max_val = -40., 40. # TODO
            if min_val > velocities[idx] or max_val < velocities[idx]:
                print(f"Out of limit **vel**: {dof_name} {velocities[idx]}, which should in [{min_val}, {max_val}]")
                out_of_limit = True
            velocities[idx] = float(torch.clip(velocities[idx], min_val, max_val))
        return out_of_limit
        
    def check_limit_pos(self, angles):
        print("angles:", angles)
        out_of_limit = False
        for idx, dof_name in enumerate(self.dof_names):
            min_val, max_val = self.dof_ranges[dof_name]
            if min_val > angles[idx] or max_val < angles[idx]:
                print(f"Out of limit: {dof_name} {angles[idx]}, which should in [{min_val}, {max_val}]")
                out_of_limit = True
            angles[idx] = float(torch.clip(angles[idx], min_val, max_val))
        return out_of_limit, angles

    def compute_angle(self, actions):
        action_scale = torch.tensor(self.train_cfg_dict_catch['control']['action_scale'], device= self.device)
        angles = actions*action_scale + self.default_joint_angles_tensor_all
        _, angles = self.check_limit_pos(angles) # clip angles.
        return torch.tensor(angles, device=self.device)

    def compute_observations(self):
        self.obs_proprio = self._get_proprioception_obs()
        self.obs_target = self._get_target_obs()
        self.obs_commands = self._get_commands_obs()
        current_obs = torch.cat([self.obs_proprio, self.obs_target, self.obs_commands], dim=-1)

        self.obs_buf = torch.cat(
            [
                current_obs.unsqueeze(dim=0),
                torch.reshape(self.obs_buf, (self.num_state_chunck, -1))[:-1,:]
            ],
            dim=0
        )

        return self.obs_buf.flatten()

    def _get_proprioception_obs(self):
        self.obs_joint_pos = (self.joint_pos - self.default_joint_angles_tensor_all) * self.obs_scales['dof_pos']
        self.obs_joint_vel = self.joint_vel * self.obs_scales['dof_vel']
        obs_proprio = torch.cat(
                    (
                        torch.zeros(3, device=self.device),
                        self.obs_ang_vel *self.obs_scales['ang_vel'],
                        self.projected_gravity_obs, 
                        self.obs_joint_pos,
                        self.obs_joint_vel,
                        self.prev_action
                    )
                )
        return obs_proprio
    
    def _get_target_obs(self):
        if torch.abs(self.commands[0]) > 0.:
            self.goal_yaw[0] = torch.atan2(self.goal_position_rel[1], self.goal_position_rel[0] + 1e-5)
            
            # Already robot frame, change order according to quat_rotate_inverse(roll, pitch, yaw).
            self.goal_rel_robot = self.goal_position_rel.clone()
            self.goal_rel_robot[1] = self.goal_position_rel[2]# TODO
            self.goal_rel_robot[2] = self.goal_position_rel[1]  
        else:
            self.goal_yaw[0] = 0.
        target_obs = torch.cat(
                    (
                        self.goal_rel_robot,
                        self.goal_yaw,
                        self.object_confidence.clone(),
                        torch.zeros(1, device=self.device),
                    )
                )
        self.object_confidence[:] *= 0.9

        if self.policy_name == "catch":
            target_obs[-1] = 0.9

        return target_obs
    
    def _get_commands_obs(self):
        return self.commands * self.commands_scale
    
    def clip_actions(self, actions):
        actions = torch.clip(actions, self.clip_actions_low, self.clip_actions_high)
        return actions

    ########################################################
    #################### Deploy Policy #####################
    ########################################################
    def set_stand_up(self):
        self.stand_up = True
        while not self.start_policy:
            cmd_start_time = time.monotonic()  
            stand_kp = 40
            stand_kd = 0.6
            stand_up_time = 2.0

            if time.monotonic() - self.start_time < stand_up_time:
                time_ratio = (time.monotonic() - self.start_time) / stand_up_time
                self.set_gains(kp=time_ratio * stand_kp, kd=time_ratio * stand_kd)
                self.set_motor_position(
                    q=self.default_joint_angles_tensor.cpu().detach()
                )
            elif time.monotonic() - self.start_time < stand_up_time * 2:
                pass
            elif time.monotonic() - self.start_time < stand_up_time * 3:
                time_ratio = (
                    time.monotonic() - self.start_time - stand_up_time * 2
                ) / stand_up_time
                kp = (1 - time_ratio) * stand_kp + time_ratio * float(30.)
                kd = (1 - time_ratio) * stand_kd + time_ratio * float(0.6)
                self.set_gains(kp=kp, kd=kd)
            self.cmd_msg.crc = get_crc(self.cmd_msg)
            self.motor_pub.publish(self.cmd_msg)
            rclpy.spin_once(self)

            print(f"Standing up...", end="\r")
            print(f"{time.monotonic() - self.start_time}", end="\r")
            cmd_end_time = time.monotonic()
            time.sleep(max(self.policy_refresh_time-(cmd_end_time - cmd_start_time),0))
    
    @torch.no_grad()
    def main_loop(self):
        idx = 0
        self.kp = self.max_kp
        self.kd = 0.6
        
        while rclpy.ok():
            if self.start_policy:   
                cmd_start_time = time.monotonic()  
                
                ##### Publish joint position/torque/velocity #####
                self.obs = self.compute_observations()
                if self.policy_name == "walk":
                    actions = self.policy(self.obs.squeeze()[:int(self.num_obs/self.num_state_chunck)]).squeeze()
                    
                else:
                    actions = self.policy(self.obs.unsqueeze(0)).squeeze()

                self.prev_action = actions.clone().detach()
                actions = self.clip_actions(actions)
                self.angles = self.compute_angle(actions)

                self.set_motor_position(self.angles.cpu())  # publish lowlevel cmd message         
 
                rclpy.spin_once(self)

                # Sleep left policy refresh time.
                self.log() # publish log messages to save mcap file
                if REAL_TO_SIM:
                    self.sim2real_file.write(f"{self.obs.detach().cpu().numpy().tolist()}\n")
                    self.sim2real_file_actions.write(f"{actions.detach().cpu().numpy().tolist()}\n")

                cmd_end_time = time.monotonic()
                time.sleep(max(self.policy_refresh_time-(cmd_end_time - cmd_start_time),0))


    def log(self):
        obs_msg = Float32MultiArray(data = self.obs.cpu().tolist())
        self.obs_log.publish(obs_msg)

        deploy_config = Float32MultiArray(
            data = [
                self.kp, self.kd,
            ] + self.joint_torq.detach().cpu().numpy().tolist()
        )
        self.deploy_config_log.publish(deploy_config)


if __name__ == "__main__":
    checkpoint = -1
    load_run = {
        "walk": 
            # "Jul21_17-04-59__pActRate2e-1_pDofAcc5e-07_pDofErr1e-01_gClose-1e+01_kd0.6_fromJul19_23-32-12",
            "Jul25_11-06-30__pActRate1e-1_pDofAcc2e-07_pDofErr2e-01_kd0.6_fromJul25_00-27-13",
        "catch": 
            # "Jul31_17-54-49__crcl_crclLth200_pDofAcc2e-07_pDofErr1e-01_trackingGoalPos-1e+01_gClose-1e+02_fromJul22_00-11-59",
            # transformer
            "Sep14_12-00-14_ActorCriticTransformer__crclLth100_trackingGoalPos-3e+00_gClose-4e+01_fromSep12_16-25-35",

    }

    rclpy.init()

    dp_node = DeployTwoPolicyNode(load_run, checkpoint)
    atexit.register(dp_node.exit_handler)
    dp_node.set_stand_up() # Keep stand up pose first

    dp_node.main_loop()
    rclpy.shutdown()