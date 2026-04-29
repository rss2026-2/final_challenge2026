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
        super().__init__("homography_transformer_final")

        # -- Declared parameters --
        self.declare_parameter('point_objects', ['tl', 'pm', 'person'])        
        self.point_objects = self.get_parameter('point_objects').get_parameter_value().string_array_value

        # Initialize dictionaries mapping objects to publishers and subscribers
        self.point_pubs = {}
        self.point_marker_pubs = {}
        self.point_px_subs = {}

        # Create publishers and subscribers by iterating through each object in point objects
        for obj_name in self.point_objects:

            # Create topic names
            point_topic = f'/{obj_name}_relative_point'
            point_px_topic = f'/{obj_name}_point_px'
            point_marker_topic = f'/{obj_name}_point_marker'

            # Create publishers
            point_pub = self.create_publisher(Point, point_topic, 10)
            point_marker_pub = self.create_publisher(Marker, point_marker_topic, 10)

            # Create subscriber, which uses a generic callback function with publishers as args
            point_px_sub = self.create_subscription(
            Pixel, 
            point_px_topic, 
            lambda msg, p_pub=point_pub, m_pub=marker_pub: self.point_px_callback(msg, p_pub, m_pub), 
            1)

            self.point_pubs[obj_name] = point_pub
            self.point_marker_pubs[obj_name] = point_marker_pub
            self.point_px_subs[obj_name] = point_px_sub
        
        # Click publishers and subscribers
        self.click_pub = self.create_publisher(Point, '/relative_click', 10)
        self.click_marker_pub = self.create_publisher(Marker, '/click_marker', 10)
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
    
    def point_px_callback(self, msg, point_pub, point_marker_pub):
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

        point_pub.publish(relative_xy_msg)
        VisualizationTools.draw_cylinder(x, y, point_marker_pub, self.get_clock().now(), "/base_link")        

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
        
        self.click_pub.publish(relative_xy_msg)
        VisualizationTools.draw_cylinder(x, y, self.click_marker_pub, self.get_clock().now(), "/base_link")        
    
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
