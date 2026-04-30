#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge

from vs_msgs.msg import ConeLocation
from geometry_msgs.msg import Point

# The following collection of pixel locations and corresponding relative
# ground plane locations are used to compute our homography matrix

# PTS_IMAGE_PLANE units are in pixels
# see README.md for coordinate frame description

######################################################
# DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_IMAGE_PLANE = [[199, 196], # avg over 5 clicks (199 196, 199 196, 199 195, 200 197, 200 197, 198 195, 199 196 )
                   [373, 209], # avg over 5 clicks (373 209, 373 209, 374 209, 374 208, 373 209)
                   [171, 249], # avg over 5 clicks  (171 249, 171 249, 170 250, 170 249, 171 250)
                   [409, 315]]  # avg over 5 clicks (409 315, 409 315, 407 314, 409 315, 409 315)
######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
# DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_GROUND_PLANE = [[46.40, 17.25], # avg over 3 measurements
                    [37.525, -7.75], # avg over 3 measurements
                    [24.025, 12.0], # avg over 3 measurements
                    [13.275, -1.625]]  # avg over 3 measurements
######################################################

METERS_PER_INCH = 0.0254


class HomographyTransformer(Node):
    def __init__(self):
        super().__init__("homography_transformer")

        self.goal_pub = self.create_publisher(ConeLocation, "/real_point", 10)
        self.click_px_sub = self.create_subscription(Point, "/goal_point", self.click_callback, 1)

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

    def click_callback(self, msg):
        u = msg.x
        v = msg.y

        x, y = self.transformUvToXy(u, v)

        relative_xy_msg = ConeLocation()
        relative_xy_msg.x_pos = float(x)
        relative_xy_msg.y_pos = float(y)

        self.goal_pub.publish(relative_xy_msg)


    def transformUvToXy(self, u, v):
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
        return x, y


def main(args=None):
    rclpy.init(args=args)
    homography_transformer = HomographyTransformer()
    rclpy.spin(homography_transformer)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
