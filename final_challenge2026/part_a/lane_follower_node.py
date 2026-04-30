#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from vs_msgs.msg import ParkingError
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Header
from geometry_msgs.msg import Point

class LaneFollower(Node):
    """
    A pure pursuit controller for lane following.
    Listens for goal points and publishes control commands immediately.
    Can be used in the simulator and on the real robot.
    """

    def __init__(self):
        super().__init__("lane_follower")
        # drive topic to publish to
        self.declare_parameter("drive_topic", "/vesc/low_level/input/navigation")
        self.DRIVE_TOPIC = self.get_parameter("drive_topic").value  # set in launch file; different for simulator vs racecar
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)

        # get the point from the homography transformer
        self.declare_parameter('goal_topic', '/real_point')
        self.goal_topic = self.get_parameter('goal_topic').value
        self.create_subscription(Point, self.goal_topic, self.goal_point_callback, 1)

        # visualize the target point
        self.declare_parameter("target_point_topic", '/target_point')
        self.TARGET_POINT_TOPIC = self.get_parameter('target_point_topic').value
        self.target_pub = self.create_publisher(Point, self.TARGET_POINT_TOPIC, 10)

        # added
        self.declare_parameter("car_length", 0.325)
        self.declare_parameter("max_steering_angle", 0.34)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("lookahead", 0.0)
        self.CAR_LENGTH = self.get_parameter('car_length').get_parameter_value().double_value
        self.MAX_STEERING_ANGLE = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.LOOKAHEAD = self.get_parameter('lookahead').get_parameter_value().double_value

        self.get_logger().info("Lane Follower Initialized")

    def goal_point_callback(self, msg):
        """Computes and publishes the drive command whenever a new goal point arrives."""
        goal_point = (msg.x, msg.y)
        target_point = self.get_point_on_line(goal_point, self.LOOKAHEAD)
        pure_persuit_drive_cmd = self.update_control(target_point)

        drive_cmd = AckermannDriveStamped()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'
        drive_cmd.header = header
        drive_cmd.drive = pure_persuit_drive_cmd
        self.drive_pub.publish(drive_cmd)


    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################

        # YOUR CODE HERE
        # Populate error_msg with the current goal point error if you wire this up later.
        error_msg.x_error = 0.0
        error_msg.y_error = 0.0
        error_msg.distance_error = 0.0

        #################################
        self.error_pub.publish(error_msg)


    def update_control(self, target_point):
        """
        Returns the ackerman drive command
        """
        drive = AckermannDrive()
        new_steering_angle = self.compute_feedback_angle(target_point)

        # it is in front of us reasonable angle, give it that angle
        drive.steering_angle = float(np.clip(new_steering_angle,
                                -self.MAX_STEERING_ANGLE,
                                self.MAX_STEERING_ANGLE))

        drive.speed = self.VELOCITY
        return drive

    def compute_feedback_angle(self, target_point):
        """
        Compute the steering angle by the pure persuit steering law.
        """
        lookahead_dist = np.linalg.norm(target_point)

        delta = np.arctan2(
            2 * self.CAR_LENGTH * target_point[1],
            lookahead_dist**2
        )
        return delta



    def get_point_on_line(self, p2, lookahead_dist: float, p1 = (0,0)):
        """
        Finds a point on a line segment (p1->p2) at a specific lookahead distance from p1.
        """
        #  create vector
        p1_vec = np.array([p1[0], p1[1]])
        p2_vec = np.array([p2[0], p2[1]])
        line_vec = p2_vec - p1_vec

        # normalize the direction vector
        length = np.linalg.norm(line_vec)
        if length == 0: return p1 # Avoid division by zero
        unit_vec = line_vec / length

        # find the point
        new_point_vec = p1_vec + (unit_vec * lookahead_dist)
        new_point_msg = Point(x=float(new_point_vec[0]), y=float(new_point_vec[1]), z=0.0)

        # publish and return
        self.target_pub.publish(new_point_msg)
        return new_point_vec

def main(args=None):
    rclpy.init(args=args)
    lf = LaneFollower()
    rclpy.spin(lf)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
