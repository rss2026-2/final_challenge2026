#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from vs_msgs.msg import ConeLocationPixel


class LaneFollower(Node):
    """
    A pure-pursuit controller for lane following.
    Subscribes to the transformed goal point and publishes drive commands.
    """

    def __init__(self):
        super().__init__("lane_follower")

        self.declare_parameter("drive_topic", "/vesc/low_level/input/navigation")
        self.DRIVE_TOPIC = self.get_parameter("drive_topic").value
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)

        self.declare_parameter("goal_topic", "/real_point")
        self.goal_topic = self.get_parameter("goal_topic").value
        self.create_subscription(ConeLocationPixel, self.goal_topic, self.goal_point_callback, 10)

        self.declare_parameter("target_point_topic", "/target_point")
        self.TARGET_POINT_TOPIC = self.get_parameter("target_point_topic").value
        self.target_pub = self.create_publisher(Point, self.TARGET_POINT_TOPIC, 10)

        self.declare_parameter("car_length", 0.325)
        self.declare_parameter("max_steering_angle", 0.34)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("lookahead", 0.8)
        self.CAR_LENGTH = self.get_parameter("car_length").get_parameter_value().double_value
        self.MAX_STEERING_ANGLE = self.get_parameter("max_steering_angle").get_parameter_value().double_value
        self.VELOCITY = self.get_parameter("velocity").get_parameter_value().double_value
        self.LOOKAHEAD = self.get_parameter("lookahead").get_parameter_value().double_value

        self.get_logger().info("Lane follower initialized")

    def goal_point_callback(self, msg):
        """
        Compute and publish a drive command immediately when a new goal point arrives.
        """
        goal_point = np.array([msg.u, msg.v], dtype=np.float32)
        target_point = self.get_point_on_line(goal_point, self.LOOKAHEAD)
        drive_cmd = AckermannDriveStamped()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"
        drive_cmd.header = header
        drive_cmd.drive = self.update_control(target_point)
        self.drive_pub.publish(drive_cmd)

    def update_control(self, target_point):
        """
        Convert a target point into a steering and speed command.
        """
        drive = AckermannDrive()
        new_steering_angle = self.compute_feedback_angle(target_point)
        drive.steering_angle = float(
            np.clip(new_steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        )
        drive.speed = self.VELOCITY
        return drive

    def compute_feedback_angle(self, target_point):
        """
        Pure pursuit steering law.
        """
        lookahead_dist = np.linalg.norm(target_point)
        if lookahead_dist == 0:
            return 0.0
        return np.arctan2(2 * self.CAR_LENGTH * target_point[1], lookahead_dist**2)

    def get_point_on_line(self, p2, lookahead_dist: float, p1=(0, 0)):
        """
        Finds a point on the line segment (p1->p2) at a specific lookahead distance from p1.
        """
        p1_vec = np.array([p1[0], p1[1]], dtype=np.float32)
        p2_vec = np.array([p2[0], p2[1]], dtype=np.float32)
        line_vec = p2_vec - p1_vec

        length = np.linalg.norm(line_vec)
        if length == 0:
            new_point_vec = p1_vec
        else:
            unit_vec = line_vec / length
            new_point_vec = p1_vec + (unit_vec * lookahead_dist)

        new_point_msg = Point(x=float(new_point_vec[0]), y=float(new_point_vec[1]), z=0.0)
        self.target_pub.publish(new_point_msg)
        return new_point_vec


def main(args=None):
    rclpy.init(args=args)
    lf = LaneFollower()
    rclpy.spin(lf)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
