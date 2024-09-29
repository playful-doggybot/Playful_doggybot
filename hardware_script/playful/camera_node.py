import sys
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
import math
import logging
from datetime import datetime
import time
import subprocess

import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

import torch

RECORD_FRAMES = False
DETECTOR_TYPE = "color" # color, yolo, groundingdino

if DETECTOR_TYPE == "yolo":
    sys.path.append('/home/unitree/duanxin/YOLO-World')
    from det_seg_track import myDetector
    RECORD_FRAMES = True
elif DETECTOR_TYPE == "groundingdino":
    sys.path.append('/home/unitree/duanxin/GroundingDINO/demo')
    from det_seg_track import myDetector
else:
    print("Using color filter.")

# resolution = [640, 360]
resolution = [848, 480] # [640, 480]
fx, fy = 915.4009, 917.5076
cx, cy = 660.06, 370.85
dist_coeffs = np.array([1.39392759e-01, -4.33120259e-01, 3.00636079e-03, 1.46124676e-04, 4.10163016e-01])
# depth_scale = 1000
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])


def get_center(contour, image):
    M = cv2.moments(contour)
    center_x = int(M["m10"] / (M["m00"]+1e-5))
    center_y = int(M["m01"] / (M["m00"]+1e-5))

    # Draw the contour and centroid on the image
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    cv2.circle(image, (center_x, center_y), 5, (0, 225, 255), -1)
    
    return center_x, center_y
class CameraNode(Node):
    def __init__(self, log_path = None):
        super().__init__("camera_node")
        self._setup_logger(log_path)

        self.pipeline = self.initialize_pipeline()
        self.pub = self.create_publisher(Float32MultiArray, "/camera", 2)
        self.msg = Float32MultiArray(data = [0 for _ in range(4)])

        self.image_pub = self.create_publisher(Image, "/camera/image", 1)
        self.depth_pub = self.create_publisher(Image, "/camera/depth_image", 1)

        self.bridge = CvBridge()

        if DETECTOR_TYPE != "color":
            self.detector = myDetector(self.log_path)

        self.idx = 0
        self.gripper_offset = [-0.44, 0.0, +0.16] # new cam rack -0.35, 0.0, +0.095
        self.gripper_offset = [-0.41, 0.0, +0.15] # skd, 3D printed rack, 0.6m
        self.gripper_offset = [-0.40, 0.0, +0.15] # skd, 3D printed rack, 0.75m, Jul31_17-54-49
        self.theta = -0. # the angle of xy-plane between realsense and gripper
        self.last_goal_x_cam_frame = 999

        self.start_time = time.monotonic()

    def _setup_logger(self, log_path):
        if not log_path:
            log_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_path = os.path.join("logs_cam", log_path)
        file_name = "camera.txt"

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        logger = logging.getLogger(file_name)
        logger.setLevel(logging.INFO)

        log_name = os.path.join(log_path, file_name)
        file_handler = logging.FileHandler(log_name)
        logger.addHandler(file_handler)

        img_path = os.path.join(log_path, "images")
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        self.log_path = log_path
        self.logger = logger
        self.img_path = img_path
    
    def initialize_pipeline(self):
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            self.logger.info("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, 60)
        config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 60)
        pipeline.start(config)

        # disable auto exposure time, set upper exp limit
        sensors = device.query_sensors() 

        for sensor in sensors:  
            if sensor.supports(rs.option.auto_exposure_priority):  
                #print('Start setting AE priority.')  
                aep = sensor.get_option(rs.option.auto_exposure_priority)  
                print(str(sensor),' supports AEP.')  
                print('Original AEP = %d' %aep)  

                # Disable auto exp, set exposure time = 70
                aep = sensor.set_option(rs.option.auto_exposure_priority, 0)  
                ep = sensor.set_option(rs.option.exposure, 180)   

                # Auto exp, this will follow the config last set!
                # aep = sensor.set_option(rs.option.auto_exposure_priority, 1) 

                aep = sensor.get_option(rs.option.auto_exposure_priority)  
                print('New AEP = %d' %aep)  

                sensor.set_option(rs.option.enable_auto_white_balance, False)
                wb_value = sensor.get_option(rs.option.white_balance)
                wb_value = 4600
                print(f"Current white balance value={wb_value}")
                sensor.set_option(rs.option.white_balance, wb_value)

        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        return pipeline
    
    def detect_max_red_object(self, image):
        # Define lower and upper bounds for red color
        # lower_red = np.array([160, 120, 60])  # lower bound for red hue
        # upper_red = np.array([165, 255, 255])  # upper bound for red hue
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # red_mask_purple = cv2.inRange(hsv_image, lower_red, upper_red)

        # # orange red
        # lower_red = np.array([0, 120, 65])  # lower bound for red hue
        # upper_red = np.array([4, 255, 255])
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # red_mask_orange = cv2.inRange(hsv_image, lower_red, upper_red)

        # red_mask = red_mask_purple | red_mask_orange
        # red_mask = red_mask_purple

        ###### green ball ########
        lower_red = np.array([35, 60, 60])  # lower bound for red hue
        upper_red = np.array([82, 255, 255])  # upper bound for red hue
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

        ###### purple ball ######
        # lower_red = np.array([110, 70, 70])  # lower bound for red hue
        # upper_red = np.array([135, 255, 255])  # upper bound for red hue
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # Find contours of the red regions
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if not contours: 
            # print("Not detected red object.")
            return
        
        # print(f"Detected {len(contours)} red objects.")
        max_contour = contours[0]
        
        if len(max_contour) < 15:
            return
        
        center_x, center_y = get_center(max_contour, image)

        z = self.depth_frame.get_distance(center_x, center_y) # geometric center
        z = self.get_median_depth_color(max_contour) # median depth value of contour
        z = self.get_min_depth_color(max_contour) # min depth value of contour

        return center_x, center_y, z
        
    def get_median_depth_color(self, contour):
        # Convert camera coordinates to 3D coordinates
        depth_values = []
        for point in contour:
            x, y = point[0]
            # depth = self.depth_image[x,y]
            depth = self.depth_frame.get_distance(x, y)
            if depth != 0 and not math.isnan(depth):
                depth_values.append(depth)

        return np.median(depth_values)
        
    def get_min_depth_color(self, contour):
        # Convert camera coordinates to 3D coordinates
        depth_values = []
        for point in contour:
            x, y = point[0]
            # depth = self.depth_image[x,y]
            depth = self.depth_frame.get_distance(x, y)
            if depth != 0 and not math.isnan(depth):
                depth_values.append(depth)

        # return np.median(depth_values)
        if len(depth_values) > 0:
            depth_values = sorted(depth_values)[:math.floor(len(depth_values)/2)]
            # return np.min(depth_values)
            return np.median(depth_values)
        else:
            return 0.
    
    def get_gripper_coord(self, center_row, center_col, depth):
        u, v, depth = center_row, center_col, depth
        intrin = self.color_frame.profile.as_video_stream_profile().intrinsics

        res = rs.rs2_deproject_pixel_to_point(intrin, [u, v], depth)
        x, y, z = res[2], res[0], -res[1] # -res[0] for yolo
        # if DETECTOR_TYPE == "color":
        #     y = res[0]

        # Transfer to gripper frame
        x = x * np.cos(self.theta) - z * np.sin(self.theta)
        z = z * np.cos(self.theta) + x * np.sin(self.theta)

        x += self.gripper_offset[0]
        y += self.gripper_offset[1]
        z += self.gripper_offset[2]

        return x, y, z
    
    def get_pixel_coor(self, mask, color_image):
        masked_image = color_image.copy()
        masked_image[mask] = 0
        image = masked_image
        mask = mask.transpose((1,0))

        nonzero_coords = np.nonzero(mask)
        if not np.any(nonzero_coords):
            return None
        row_coords, col_coords = nonzero_coords
        center_row = int(np.median(row_coords))
        center_col = int(np.median(col_coords))

        depth = self.get_median_depth(row_coords, col_coords)
        return (center_row, center_col, depth)

    def main_loop(self):
        log_info = ""
        depth_buffer = []

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align = rs.align(rs.stream.color)

        # 0 - fill_from_left - Use the value from the left neighbor pixel to fill the hole
        # 1 - farest_from_around - Use the value from the neighboring pixel which is furthest away from the sensor
        # 2 - nearest_from_around - Use the value from the neighboring pixel closest to the sensor
        hole_filling = rs.hole_filling_filter(2)

        try:
            while True:
                if self.idx % 10 == 0:
                    self.logger.info(log_info)
                    mean_depth = np.mean(depth_buffer)
                    self.logger.info(f"Mean depth: {mean_depth} {depth_buffer}\n")
                    log_info = ""
                    depth_buffer = []
                    
                frames = self.pipeline.wait_for_frames() # Wait for a coherent pair of frames: depth and color
                aligned_frames = align.process(frames) # Align the depth frame to color frame
                self.depth_frame = aligned_frames.get_depth_frame()
                self.color_frame = aligned_frames.get_color_frame()

                if not self.depth_frame or not self.color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.array(self.color_frame.get_data())
                # depth_image = np.array(hole_filling.process(self.depth_frame).get_data(), dtype=np.float64)
                depth_image = np.array(self.depth_frame.get_data(), dtype=np.float64)

                self.depth_image = depth_image * self.depth_scale

                image_msg = self.bridge.cv2_to_imgmsg(cv2.resize(color_image, dsize=(212, 120), interpolation=cv2.INTER_CUBIC), "bgr8")
                depth_msg = self.bridge.cv2_to_imgmsg(cv2.resize(depth_image, dsize=(212, 120), interpolation=cv2.INTER_CUBIC).astype(np.ushort))
                

                if RECORD_FRAMES:
                    filename = os.path.join(self.img_path, f"{self.idx:04d}.png")
                    cv2.imwrite(filename, color_image)
                self.idx += 1

                if DETECTOR_TYPE == "color":
                    # clip by depth
                    coor = self.detect_max_red_object(color_image)
                else:
                    if DETECTOR_TYPE == "groundingdino":
                        mask = self.detector.track(color_image)
                    else:
                        mask = self.detector.track(filename)
                    if not np.any(mask):
                        self.image_pub.publish(image_msg)
                        self.depth_pub.publish(depth_msg) # ,  encoding= "16UC1"
                        continue
                    coor = self.get_pixel_coor(mask, color_image)

                if coor is None:
                    self.image_pub.publish(image_msg)
                    self.depth_pub.publish(depth_msg) # ,  encoding= "16UC1"
                    continue
                center_row, center_col, depth = coor

                goal_x_cam_frame, goal_y_cam_frame, goal_z_cam_frame = self.get_gripper_coord(center_row, center_col, depth)

                # if goal_x_cam_frame - self.last_goal_x_cam_frame > 0.2:
                #     goal_x_cam_frame = self.last_goal_x_cam_frame
                    
                self.last_goal_x_cam_frame = goal_x_cam_frame

                text = f'camera coordinates: ({goal_x_cam_frame:.02f},{goal_y_cam_frame:.02f},{goal_z_cam_frame:.02f})'
                log_info += text
                depth_buffer.append(goal_x_cam_frame)
                print(text)

                color_image = cv2.resize(color_image, dsize=(212, 120), interpolation=cv2.INTER_CUBIC)
                
                self.msg.data[0] = goal_x_cam_frame
                self.msg.data[1] = goal_y_cam_frame
                self.msg.data[2] = goal_z_cam_frame
                self.msg.data[3] = 1.

                image_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
                self.image_pub.publish(image_msg)
                self.depth_pub.publish(depth_msg) # ,  encoding= "16UC1"
                self.pub.publish(self.msg)

                if RECORD_FRAMES:
                    filename = os.path.join(self.img_path, f"{self.idx-1:04d}.png")
                    cv2.imwrite(filename, color_image)
                
                # # Show images
                # cv2.namedWindow("RealSense", cv2.WINDOW_NORMAL)
                # cv2.imshow('RealSense', color_image)
                # cv2.waitKey(1)



        finally:
            self.pipeline.stop() # Stop streaming
            if RECORD_FRAMES:
                # subprocess.call(["bash","./video.sh", self.log_path])
                pass

if __name__ == "__main__":
    log_path = None
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    rclpy.init()

    dp_node = CameraNode(log_path)
    dp_node.main_loop()
    
    rclpy.shutdown()
