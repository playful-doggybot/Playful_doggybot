# Hardware deployment #

### Installation ###
- install unitree_sdk:  
    https://github.com/unitreerobotics/unitree_sdk2  
    
- install unitree_ros2:  
    https://support.unitree.com/home/en/developer/ROS2_service  

- install robotis sdk for the gripper(set USE_GRIPPPER=True in play_hardware.py):  
    https://www.youtube.com/watch?v=E8XPqDjof4U  

```bash
conda create -n doggy python=3.8
conda activate doggy
```

- install nvidia-jetpack

- install torch==1.11.0 torchvision==0.12.0:  
    https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048   
    https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html  

```bash
conda activate doggy
pip install "numpy<1.24" opencv-python
```

### Usage ###

1. turn off sport mode: (needed only once after turning on the robot). Run the following command or turn off it with the unitree app
    - `./stand_up_go2 eth0`

2. save the checkpoint folder to `models` and run the policy with:  
    - `python deploy_node.py`


3. to set up the Realsense D435i camera,
    - `python camera_node.py`

4. (optional) you can install mcap to save the experiment details as `.mcap` file, run the following command. You can also use Foxglove Studio to visualize the whole process.
    - `ros2 bag -s mcap record <topic_name> -o <file_name>`

**joystick commands:**
- L1: emergency stop
- L2: start playing policy
- R1: catch object mode.
- R2: walk mode.
- left stick: linear movement
- right stick: yaw & pitch