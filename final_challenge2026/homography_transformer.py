#!/usr/bin/env python3

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker

# PTS_IMAGE_PLANE units are in pixels.
PTS_IMAGE_PLANE = [[199, 196],
                   [373, 209],
                   [171, 249],
                   [409, 315]]

# PTS_GROUND_PLANE units are in inches.
# car looks along positive x axis with positive y axis to left
PTS_GROUND_PLANE = [[46.40, 17.25],
                    [37.525, -7.75],
                    [24.025, 12.0],
                    [13.275, -1.625]]

METERS_PER_INCH = 0.0254


class HomographyTransformer(Node):
    def __init__(self):
        super().__init__("homography_transformer")

        self.lane_points_pub = self.create_publisher(PoseArray, "/relative_lane_points", 10)
        self.marker_pub = self.create_publisher(Marker, "/lane_marker", 1)
        self.lane_px_sub = self.create_subscription(PoseArray, "/lane_points_px", self.lane_points_callback, 1)

        if len(PTS_GROUND_PLANE) != len(PTS_IMAGE_PLANE):
            self.get_logger().error("PTS_GROUND_PLANE and PTS_IMAGE_PLANE should be the same length")

        np_pts_ground = np.float32(np.array(PTS_GROUND_PLANE) * METERS_PER_INCH)[:, np.newaxis, :]
        np_pts_image = np.float32(np.array(PTS_IMAGE_PLANE))[:, np.newaxis, :]

        self.h, _ = cv2.findHomography(np_pts_image, np_pts_ground)
        self.get_logger().info("Homography Transformer Initialized")

    def lane_points_callback(self, msg):
        world_points = PoseArray()
        world_points.header.stamp = msg.header.stamp
        world_points.header.frame_id = "base_link"

        marker = Marker()
        marker.header.stamp = msg.header.stamp
        marker.header.frame_id = "base_link"
        marker.ns = "lane_points"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        poses = msg.poses
        for i in range(0, len(poses) - 1, 2):
            x1, y1 = poses[i].position.x, poses[i].position.y
            x2, y2 = poses[i + 1].position.x, poses[i + 1].position.y

            wx1, wy1 = self.transform_uv_to_xy(x1, y1)
            wx2, wy2 = self.transform_uv_to_xy(x2, y2)

            pose1 = Pose()
            pose1.position.x = float(wx1)
            pose1.position.y = float(wy1)
            pose1.orientation.w = 1.0

            pose2 = Pose()
            pose2.position.x = float(wx2)
            pose2.position.y = float(wy2)
            pose2.orientation.w = 1.0

            world_points.poses.extend([pose1, pose2])

            p1 = Point()
            p1.x = float(wx1)
            p1.y = float(wy1)
            p1.z = 0.0

            p2 = Point()
            p2.x = float(wx2)
            p2.y = float(wy2)
            p2.z = 0.0

            marker.points.extend([p1, p2])

        self.lane_points_pub.publish(world_points)
        self.marker_pub.publish(marker)

    def transform_uv_to_xy(self, u, v):
        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(self.h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        return homogeneous_xy[0, 0], homogeneous_xy[1, 0]


def main(args=None):
    rclpy.init(args=args)
    node = HomographyTransformer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
