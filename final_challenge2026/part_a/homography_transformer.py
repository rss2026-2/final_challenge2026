#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from visualization_msgs.msg import Marker
from vs_msgs.msg import Pixel, PixelArray
from geometry_msgs.msg import Point, Point32, Polygon

from viz_utils.visualization_tools import VisualizationTools

# The following collection of pixel locations and corresponding relative
# ground plane locations are used to compute our homography matrix

# PTS_IMAGE_PLANE units are in pixels
# see README.md for coordinate frame description

######################################################
# DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_IMAGE_PLANE = [[497, 268], # avg over 5 clicks (492 267, 492 267, 502 270, 501 267, 502 269 )
                   [395, 148], # avg over 5 clicks (397 148, 395 149, 394 148, 394 148, 395 147)
                   [160.0, 175], # avg over 5 clicks  (162 175, 158 175, 160 174, 158 175, 160 174)
                   [56, 159]]  # avg over 5 clicks (56 158, 56 159, 56 159, 57 158, 54 160)
######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
# DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_GROUND_PLANE = [[17.99, -10.51], # avg over 3 measurements  18 in x 10.5 in
                    [134.5,  -37.0], # avg over 3 measurements
                    [57.50, 24.25], # avg over 3 measurements
                    [99.0, 77.0]]  # avg over 3 measurements
######################################################

METERS_PER_INCH = 0.0254


class HomographyTransformer(Node):
    def __init__(self):
        super().__init__("homography_transformer_a")

        # -- Declared parameters -- 
        self.declare_parameter("goal_point_topic", "/goal_point")
        self.declare_parameter("goal_point_px_topic", "/goal_point_px")
        self.declare_parameter("goal_point_marker_topic", "/goal_point_marker")

        self.declare_parameter("line_topic", "/relative_line")
        self.declare_parameter("line_px_topic", "/line_px")
        self.declare_parameter("line_marker_topic", "/line_marker")

        self.goal_point_topic = self.get_parameter("goal_point_topic").get_parameter_value().string_value
        self.goal_point_px_topic = self.get_parameter("goal_point_px_topic").get_parameter_value().string_value
        self.goal_point_marker_topic = self.get_parameter("goal_point_marker_topic").get_parameter_value().string_value

        self.line_topic = self.get_parameter("line_topic").get_parameter_value().string_value
        self.line_px_topic = self.get_parameter("line_px_topic").get_parameter_value().string_value
        self.line_marker_topic = self.get_parameter("line_marker_topic").get_parameter_value().string_value

        # -- Publishers and subscribers --
        self.goal_point_px_sub = self.create_subscription(Pixel, self.goal_point_px_topic, self.goal_point_px_callback, 1)
        self.goal_point_pub = self.create_publisher(Point, self.goal_point_topic, 10)
        self.goal_point_marker_pub = self.create_publisher(Marker, self.goal_point_marker_topic, 1)
   
        self.line_px_sub = self.create_subscription(PixelArray, self.line_px_topic, self.line_px_callback, 1)
        self.line_pub = self.create_publisher(Polygon, self.line_topic, 10)
        self.line_marker_pub = self.create_publisher(Marker, self.line_marker_topic, 1)

        self.click_px_sub = self.create_subscription(Pixel, "/click_px", self.click_callback, 1)

        if not len(PTS_GROUND_PLANE) == len(PTS_IMAGE_PLANE):
            rclpy.logerr("ERROR: PTS_GROUND_PLANE and PTS_IMAGE_PLANE should be of same length")

        # Initialize data into a homography matrix

        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)

        self.get_logger().info("Homography Transformer Initialized")
    
    def goal_point_px_callback(self, msg):
        # Extract information from message
        u = msg.u
        v = msg.v

        # Call to main function
        x, y = self.transform_uv_to_xy(u, v)
        y += 0.06 # Zed camera to base_link offset

        # Publish relative xy position of object in real world
        relative_xy_msg = Point()
        relative_xy_msg.x = float(x)
        relative_xy_msg.y = float(y)

        self.goal_point_pub.publish(relative_xy_msg)
        VisualizationTools.draw_cylinder(x, y, self.goal_point_marker_pub, self.get_clock().now(), "/base_link")        

    def click_callback(self, msg):

        # Call to main function
        x, y = self.transform_uv_to_xy(msg.u, msg.v)
        
        self.get_logger().info(f'{msg.u=}, {msg.v=}')
        self.get_logger().info(f'{x=}, {y=}')
        
        # Publish relative xy position of object in real world
        relative_xy_msg = Point()
        relative_xy_msg.x = float(x)
        relative_xy_msg.y = float(y)
        y += 0.06 # Zed camera to base_link offset
        
        self.goal_point_pub.publish(relative_xy_msg)
        VisualizationTools.draw_cylinder(x, y, self.goal_point_marker_pub, self.get_clock().now(), "/base_link")        
    
    def line_px_callback(self, msg):
        
        world_points = Polygon()
        world_points_raw = []

        pixels = msg.pixels
        for i in range(0, len(pixels) - 1, 2):
            u1, v1 = pixels[i].u, pixels[i].v
            u2, v2 = pixels[i + 1].u, pixels[i + 1].v

            x1, y1 = self.transform_uv_to_xy(u1, v1)
            x2, y2 = self.transform_uv_to_xy(u2, v2)

            point1 = Point32()
            point1.x = float(x1)
            point1.y = float(y1)
            point1.z = 0.0

            point2 = Point32()
            point2.x = float(x2)
            point2.y = float(y2)
            point2.z = 1.0

            world_points.points.extend([point1, point2])
            world_points_raw.extend([[x1,y1],[x2,y2]]) # For drawing line in real world

        world_points_raw = np.array(world_points_raw)

        self.line_pub.publish(world_points)
        VisualizationTools.draw_line(world_points_raw[:, 0], world_points_raw[:, 1], self.line_marker_pub, self.get_clock().now())

    def transform_uv_to_xy(self, u, v):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.

        Returns a normal non-np 1x2 matrix of xy displacement vector from the
        camera to the point on the ground plane.
        Camera points along positive x axis and y axis increases to the left of
        the camera.

        Units are in meters.
        """
        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(self.h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        # self.get_logger().info(f"x-value: {x} \n y-value: {y}")
        return x, y

def main(args=None):
    rclpy.init(args=args)
    homography_transformer = HomographyTransformer()
    rclpy.spin(homography_transformer)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
