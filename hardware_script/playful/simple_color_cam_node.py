import sys
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
import math
import logging
from datetime import datetime
import time
import matplotlib.pyplot as plt
import subprocess

import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

# sys.path.append('/home/unitree/duanxin/YOLO-World')
# from det_seg_track import myDetector





RECORD_FRAMES = False
USE_DETECTOR = False

font = cv2.FONT_HERSHEY_SIMPLEX  #
font_scale = 0.6  
color = (255, 255, 0)  
thickness = 2

fx, fy = 915.4009, 917.5076
cx, cy = 660.06, 370.85
resolution = [848, 480]
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


def draw_origin(image):
    x, y = 645, 356
    length = 20
    points = np.array([[x-length, y], [x+length,y], [x,y], [x, y - length], [x, y + length]], dtype=np.float32)
    points = points.astype(np.int32)
    cv2.polylines(image, [points], False, (0, 0, 255), 2)


class CameraNode(Node):
    def __init__(self, log_path = None):
        super().__init__("camera_node")
        self._setup_logger(log_path)

        self.pipeline = self.initialize_pipeline()
        self.pub = self.create_publisher(Float32MultiArray, "/camera", 1)
        self.msg = Float32MultiArray(data = [0 for _ in range(4)])

        self.image_pub = self.create_publisher(Image, "/camera/image", 1)
        self.depth_pub = self.create_publisher(Image, "/camera/depth_image", 1)

        self.bridge = CvBridge()

        # if USE_DETECTOR:
        #     self.detector = myDetector(self.log_path)

        self.idx = 0
        self.gripper_offset = [-0.33, 0.0, +0.08]

        self.depth_hist = [] 
        self.time_hist = []
        self.x_hist = []
        self.y_hist = []
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
                ep = sensor.set_option(rs.option.exposure, 150)   

                # Auto exp, this will follow the config last set!
                # aep = sensor.set_option(rs.option.auto_exposure_priority, 1) 

                aep = sensor.get_option(rs.option.auto_exposure_priority)  
                print('New AEP = %d' %aep)  

        return pipeline
    
    def detect_max_red_object(self, image):
        # Define lower and upper bounds for red color
        #lower_red = np.array([0, 0, 150])  # lower bound for red hue
        #upper_red = np.array([100, 100, 255])  # upper bound for red hue
        # purple red
        lower_red = np.array([168, 80, 80])  # lower bound for red hue
        upper_red = np.array([180, 255, 255])  # upper bound for red hue

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask_purple = cv2.inRange(hsv_image, lower_red, upper_red)

        # orange red
        lower_red = np.array([0, 80, 80])  # lower bound for red hue
        upper_red = np.array([12, 255, 255])

        # Convert BGR to HSV. Create a mask for red color
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask_orange = cv2.inRange(hsv_image, lower_red, upper_red)

        red_mask = red_mask_purple | red_mask_orange

        # Find contours of the red regions
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if not contours: return
        if len(contours) <= 20: return
        max_contour = contours[0]
        
        center_x, center_y = get_center(max_contour, image)
        
        z = self.get_median_depth(max_contour)

        return center_x, center_y, z
        
    def get_median_depth(self, contour):
        # Convert camera coordinates to 3D coordinates
        depth_values = []
        for point in contour:
            x, y = point[0]
            depth = self.depth_frame.get_distance(x, y)
            if depth != 0 and not math.isnan(depth):
                depth_values.append(depth)
    
        return np.median(depth_values)
    
    def main_loop(self):
        log_info = ""
        depth = []
        align = rs.align(rs.stream.color)
        try:
            while True:
                
                if self.idx % 10 == 0:
                    self.logger.info(log_info)
                    mean_depth = np.mean(depth)
                    self.logger.info(f"Mean depth: {mean_depth} {depth}\n")
                    # print(depth)
                    log_info = ""
                    depth = []
                    
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                self.depth_frame = aligned_frames.get_depth_frame()
                self.color_frame = aligned_frames.get_color_frame()
                if not self.depth_frame or not self.color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(self.color_frame.get_data())
                depth_image = np.asanyarray(self.depth_frame.get_data())

                # undistored image
                image = color_image
                image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
                self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_image))

                # if RECORD_FRAMES:
                #     filename = os.path.join(self.img_path, f"{self.idx:04d}.png")
                #     cv2.imwrite(filename, image)
                #     if USE_DETECTOR:
                #         res = self.detector.track(filename, show=True)
                #         print(res)
                #         cv2.imshow('RealSense', image*res)

                self.idx += 1
        
                coor = self.detect_max_red_object(image)
                if not coor:
                    self.image_pub.publish(image_msg)
                    continue
                u, v, z = coor

                intrin = self.color_frame.profile.as_video_stream_profile().intrinsics
                res = rs.rs2_deproject_pixel_to_point(intrin, [u, v], z)

                goal_x_cam_frame, goal_y_cam_frame, goal_z_cam_frame = res[2], -res[0], -res[1]
                goal_x_cam_frame += self.gripper_offset[0]
                goal_y_cam_frame += self.gripper_offset[1]
                goal_z_cam_frame += self.gripper_offset[2]

                text = f'camera coordinates: ({goal_x_cam_frame:.02f},{goal_y_cam_frame:.02f},{goal_z_cam_frame:.02f})\n'
                log_info += text
                depth.append(goal_x_cam_frame)
                position = (u, v)  
                
                cv2.putText(image, text, position, font, font_scale, color, thickness)

                self.msg.data[0] = goal_x_cam_frame
                self.msg.data[1] = goal_y_cam_frame
                self.msg.data[2] = goal_z_cam_frame
                self.msg.data[3] = 1.

                image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
                self.image_pub.publish(image_msg)
                self.pub.publish(self.msg)
            
                # draw_origin(image)

                # Show images
                # cv2.imshow('RealSense', image)
                # cv2.waitKey(1)
                
                # if RECORD_FRAMES:
                #     filename = os.path.join(self.img_path, f"{self.idx-1:04d}.png")
                #     cv2.imwrite(filename, image)
                
                
        finally:
            self.pipeline.stop() # Stop streaming
            if RECORD_FRAMES:
                subprocess.call(["bash","./video.sh", self.log_path])
        

def get_median_depth(contour):
        # Convert camera coordinates to 3D coordinates
        depth_values = []
        for point in contour:
            x, y = point[0]
            depth = self.depth_frame.get_distance(x, y)
            if depth != 0 and not math.isnan(depth):
                depth_values.append(depth)
    
        return np.median(depth_values)

def detect_max_red_object(image):
    # Define lower and upper bounds for red color
    #lower_red = np.array([0, 0, 150])  # lower bound for red hue
    #upper_red = np.array([100, 100, 255])  # upper bound for red hue
    # purple red
    lower_red = np.array([168, 80, 80])  # lower bound for red hue
    upper_red = np.array([180, 255, 255])  # upper bound for red hue

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask_purple = cv2.inRange(hsv_image, lower_red, upper_red)

    # orange red
    lower_red = np.array([0, 80, 80])  # lower bound for red hue
    upper_red = np.array([12, 255, 255])

    # Convert BGR to HSV. Create a mask for red color
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask_orange = cv2.inRange(hsv_image, lower_red, upper_red)

    red_mask = red_mask_purple | red_mask_orange

    # Find contours of the red regions
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours: return
    if len(contours) <= 20: return
    max_contour = contours[0]
    
    center_x, center_y = get_center(max_contour, image)
    
    z = get_median_depth(max_contour)

    return center_x, center_y, z

class myDetector:
    def __init__(self) -> None:
        pass

    def track(self, color_image):
        return detect_max_red_object(color_image)



if __name__ == "__main__":
    log_path = None
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    rclpy.init()

    dp_node = CameraNode(log_path)
    dp_node.main_loop()
    # dp_node.destroy()
    
    rclpy.shutdown()


