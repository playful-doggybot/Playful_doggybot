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

import rosbag2_py
from rclpy.clock import Clock
from rclpy.duration import Duration
from rclpy.serialization import serialize_message
from rclpy.serialization import deserialize_message
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from legged_gym.scripts.play import PlayNode, exit_handler

import cv2
from cv_bridge import CvBridge

class ROSPlayNode(Node):
    def __init__(self, args):
        super().__init__("sim_node")
        self.player = PlayNode(args)
        self.real_obs = None

        self._init_publishers()
        
        if RECORD_FRAMES:
            self._init_rosbag_writers()

        if REPLAY:
            self._init_rosbag_reader(args.rosbag)

    def _init_publishers(self):
        self.sim_obs_log = self.create_publisher(Float32MultiArray, "/sim_logs/obs", 1)
        self.sim_reward_log = self.create_publisher(Float32MultiArray, "/sim_logs/rewards", 1)
        self.sim_first_view_log = self.create_publisher(Image, "/sim_logs/first_view", 1)
        self.sim_third_view_log = self.create_publisher(Image, "/sim_logs/third_view", 1)
        self.tmp_log = self.create_publisher(Float32MultiArray, "/sim_logs/tmp", 1)
        self.bridge = CvBridge()

    def _init_rosbag_writers(self):
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.player.train_cfg.runner.experiment_name, self.player.args.load_run, self.player.args.load_run.split('__')[0]+time.strftime("_%b-%d_%H-%M-%S", time.localtime()))        

        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py._storage.StorageOptions(
            uri=path,
            storage_id='mcap')
        converter_options = rosbag2_py._storage.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)

        topic_info = rosbag2_py._storage.TopicMetadata(
            name='/sim_logs/obs',
            type='std_msgs/msg/Float32MultiArray',
            serialization_format='cdr')
        self.writer.create_topic(topic_info)

        topic_info = rosbag2_py._storage.TopicMetadata(
            name='/sim_logs/rewards',
            type='std_msgs/msg/Float32MultiArray',
            serialization_format='cdr')
        self.writer.create_topic(topic_info)

        topic_info = rosbag2_py._storage.TopicMetadata(
            name='/sim_logs/first_view',
            type='sensor_msgs/msg/Image',
            serialization_format='cdr')
        self.writer.create_topic(topic_info)

        topic_info = rosbag2_py._storage.TopicMetadata(
            name='/sim_logs/third_view',
            type='sensor_msgs/msg/Image',
            serialization_format='cdr')
        self.writer.create_topic(topic_info)

        topic_info = rosbag2_py._storage.TopicMetadata(
            name='/sim_logs/tmp',
            type='std_msgs/msg/Float32MultiArray',
            serialization_format='cdr')
        self.writer.create_topic(topic_info)

        self.action_file = open(os.path.join(path, 'actions.txt'), 'w')
        self.obs_file = open(os.path.join(path, 'obs.txt'), 'w')

        

    def _init_rosbag_reader(self, input_bag):
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.player.train_cfg.runner.experiment_name, self.player.args.load_run, input_bag)
        print("init reader:", path)
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr"
            )
        )
        self.reader = reader

    def get_viewer(self):
        env = self.player.env
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        env.gym.start_access_image_tensors(env.sim)
    
        forward_image_ = env.gym.get_camera_image_gpu_tensor(
            env.sim,
            env.envs[0],
            env.sensor_handles[0]['forward_camera'],
            gymapi.IMAGE_COLOR
        )
        forward_image = gymtorch.wrap_tensor(forward_image_).cpu().numpy()
        
        viewer_image_ = env.gym.get_camera_image_gpu_tensor(
            env.sim,
            env.envs[0],
            env.sensor_handles[0]['viewer_camera'],
            gymapi.IMAGE_COLOR
        )
        viewer_image = gymtorch.wrap_tensor(viewer_image_).cpu().numpy()
        
        showed_image = cv2.cvtColor(forward_image, cv2.COLOR_RGB2BGR)
        showerd_viewer_image = cv2.cvtColor(viewer_image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("forward image", cv2.WINDOW_NORMAL)
        cv2.imshow("forward image", showed_image)
        cv2.imshow("viewer image", showerd_viewer_image)
        cv2.waitKey(1)
        
        env.gym.end_access_image_tensors(env.sim)

        image_msg = self.bridge.cv2_to_imgmsg(cv2.resize(forward_image, dsize=(106, 60), interpolation=cv2.INTER_CUBIC), "rgba8")
        viewer_image_msg = self.bridge.cv2_to_imgmsg(viewer_image, "rgba8")
        return image_msg, viewer_image_msg

    @torch.no_grad()
    def play(self, args):
        # self.player.play()
        args = self.player.args
        env = self.player.env
        ppo_runner, policy = self.player.ppo_runner, self.player.policy

        obs = env.get_observations()

        env.commands[:,:] = 0. # to beter compare action rate of different checkpoints.
        
        i = 0
        for key in env.reward_names:
            print(f"{i}: {key}")
            i += 1
        time_stamp = Clock().now()
        env.commands[:,0] = 0.
        
        idx = 0
        while rclpy.ok():
            start_time = time.monotonic()
            # env.commands[:,0] = 0.5
            
            if False: # set ball position
                real_pos = torch.tensor([-0.2, 0.2, 0.], device=self.player.env.device)
                real_pos = quat_rotate(self.player.env.grasp_point_quat, real_pos.unsqueeze(0))
                target = self.player.env.grasp_point_pos[0, :3] + real_pos
                env.object_pos[0, :3] = target
                env.init_object_pos = env.object_pos.clone()
                env.goal_pos = env.object_pos.clone() # override [0] dosen't work
                
                
                # print(obs[0,45:45+6])
                # env.target_obs_last[0,:] = torch.tensor(data[45:45+6], device=self.player.env.device)
                # env.goal_rel_robot_obs[0,:] = torch.tensor(data[45:45+3], device=self.player.env.device)
                    
                print(env.goal_pos, env.object_pos)
                actor_idx_int32 = torch.zeros(1, device=self.player.env.device).to(dtype=torch.int32)
                self.player.env.gym.set_actor_root_state_tensor_indexed(self.player.env.sim,
                                                            gymtorch.unwrap_tensor(self.player.env.all_root_states),
                                                            gymtorch.unwrap_tensor(actor_idx_int32), len(actor_idx_int32))
            
            
            if REPLAY:
                while self.reader.has_next():
                    topic, data, timestamp = self.reader.read_next()
                    
                    # if topic == "/sim_logs/obs":
                    if topic == "/logs/obs":
                        # print(timestamp*1.e-9)
                        msg = deserialize_message(data, Float32MultiArray)
                        data = msg.data
                        ##### Publish joint position/torque/velocity #####
                        if idx <= 9999:
                            idx += 1
                            real_pos = torch.tensor(data[45:45+3], device=self.player.env.device)
                            real_pos = quat_rotate(self.player.env.grasp_point_quat, real_pos.unsqueeze(0))
                            target = self.player.env.grasp_point_pos[0, :3] + real_pos
                        env.object_pos[0, :3] = target
                        env.init_object_pos = env.object_pos.clone()
                        env.goal_pos = env.object_pos.clone() # override [0] dosen't work
                        
                        
                        # print(obs[0,45:45+6])
                        # env.target_obs_last[0,:] = torch.tensor(data[45:45+6], device=self.player.env.device)
                        # env.goal_rel_robot_obs[0,:] = torch.tensor(data[45:45+3], device=self.player.env.device)
                        env.commands[0,:] = torch.tensor(data[51:], device=self.player.env.device)
                        # obs[0,:] = torch.tensor(data, device=self.player.env.device)
                        # obs[0,9:9+12] = 0.
                        # obs[0,9+12:9+12*2] = 0.
                        # obs[0,9+12*2:9+12*3] = 0.
                            
                        # print(env.goal_pos, env.object_pos)
                        actor_idx_int32 = torch.zeros(1, device=self.player.env.device).to(dtype=torch.int32)
                        self.player.env.gym.set_actor_root_state_tensor_indexed(self.player.env.sim,
                                                                    gymtorch.unwrap_tensor(self.player.env.all_root_states),
                                                                    gymtorch.unwrap_tensor(actor_idx_int32), len(actor_idx_int32))
                        # env.compute_observations()
                        # # obs[0,45:45+6] = torch.tensor(data[45:45+6], device=self.player.env.device)
                        # print(obs[0,45:45+6], data[45:45+6])
                        actions = policy(obs.detach())
                        if USE_ACTIONS:
                            actions[0,:] = torch.tensor(data[33:33+12], device=self.player.env.device)
                        break
                else:
                    break
            else:
                actions = policy(obs.detach())
            actions = torch.reshape(actions, (1, actions.shape[-1]))
            obs, critic_obs, rews, dones, infos = env.step(actions.detach())
            # print(env.commands)
            # print(env.gripper_close_time)
            # print(env.num_obs)
            # print(env.obs_history_buf.shape)
            
            self.player.record_frames()            
            self.player.set_camera()
            self.player.act_event(actions)
            # self.player.log_info(actions, teacher_actions, rews, infos, i)
            self.player.check_done(dones)
            
            image_msg,viewer_image_msg =  self.get_viewer()

            ######### Publish Message #############
            # self.sim_obs_log.publish(Float32MultiArray(data=obs[0].cpu().tolist()))
            
            rew_buf = []
            for i in env.reward_names:
                rew = env.rew_buf_msg[i] # / env.reward_scales[env.reward_names[i]]
                rew_buf.append(rew)
            # self.sim_reward_log.publish(Float32MultiArray(data = rew_buf))

            ######### Write Message to Bag File #############
            if RECORD_FRAMES:
                self.action_file.write(f"{actions[0].detach().cpu().numpy().tolist()}\n")
                self.obs_file.write(f"{obs[0].detach().cpu().numpy().tolist()}\n")
                self.writer.write(
                    "/sim_logs/obs",
                    serialize_message(Float32MultiArray(data=obs[0].cpu().tolist())),
                    time_stamp.nanoseconds
                )
                self.writer.write(
                    "/sim_logs/rewards",
                    serialize_message(Float32MultiArray(data=rew_buf)),
                    time_stamp.nanoseconds
                )
                self.writer.write(
                    "/sim_logs/first_view",
                    serialize_message(image_msg),
                    time_stamp.nanoseconds
                )
                self.writer.write(
                    "/sim_logs/third_view",
                    serialize_message(viewer_image_msg),
                    time_stamp.nanoseconds
                )
                # For temperal debugging.
                self.writer.write(
                    "/sim_logs/tmp",
                    serialize_message(Float32MultiArray(data=
                        env.torques[0].cpu().tolist()
                        # env.yaw,
                        # env.goal_yaw
                    )),
                    time_stamp.nanoseconds
                )
                time_stamp += Duration(seconds = 0.02)
            
            # if REPLAY:
            #     rclpy.spin_once(self)
                
            end_time = time.monotonic()
            if args.slow > 0:
                # print(end_time - start_time)
                time.sleep(max(args.slow - (end_time - start_time), 0.))


if __name__ == '__main__':
    RECORD_FRAMES = False
    REPLAY = False
    USE_ACTIONS = False
    args = get_args([
        dict(name= "--slow", type= float, default= 0.02, help= "slow down the simulation by sleep secs (float) every frame"),
        dict(name= "--show_teacher", action= "store_true", default= False, help= "show teacher actions"),
        dict(name= "--no_teacher", action= "store_true", default= False, help= "whether to disable teacher policy when running the script"),
        # dict(name= "--zero_act_until", type= int, default= 0., help= "zero action until this step"),
        dict(name= "--sample", action= "store_true", default= False, help= "sample actions from policy"),
        dict(name= "--plot_time", type= int, default= 100, help= "plot states after this time"),
        dict(name= "--no_throw", action= "store_true", default= False),
        # dict(name="--rosbag", type=str, default="", help="If you have a ros bag mcao file to replay.")
    ])
    
    args.rosbag = "0719_field_Jul18_22-53-08_01/0719_field_Jul18_22-53-08_01_0.mcap"

    # auto trans from imges to video
    start_time = time.time()
    if RECORD_FRAMES:
        atexit.register(exit_handler)
    
    rclpy.init()
    node = ROSPlayNode(args)
    node.play(args)
    rclpy.shutdown()
