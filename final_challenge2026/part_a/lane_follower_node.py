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
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """

    def __init__(self):
        super().__init__("lane_follower")
        # drive topic to publish to
        self.declare_parameter("drive_topic")
        self.DRIVE_TOPIC = self.get_parameter("drive_topic").value  # set in launch file; different for simulator vs racecar
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)

        # get the point from the lane_detector_node
        self.declare_parameter('goal_topic', '/goal_point')
        self.goal_topic = self.get_parameter('goal_topic').value
        self.create_subscription(Point, self.goal_topic, self.relative_cone_callback, 1)

        # visualize the target point
        self.declare_parameter("target_point_topic", '/target_point')
        self.TARGET_POINT_TOPIC = self.get_parameter('target_point_topic').value
        self.target_pub = self.create_publisher(Point, self.TARGET_POINT_TOPIC, 10)

        # probably won't need these because we aren't stopping.
        self.parking_distance_min = 0.45 # meters; try playing with this number! it should be 1.5 - 2 feet away (0.45 - 0.6 m)
        self.parking_distance_max = 0.55
        self.relative_x = 0.0
        self.relative_y = 0.0

        # added
        self.declare_parameter("car_length", 0.325)
        self.declare_parameter("max_steering_angle", 0.34)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("lookahead", 0.8)
        self.CAR_LENGTH = self.get_parameter('car_length').get_parameter_value().double_value
        self.MAX_STEERING_ANGLE = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.LOOKAHEAD = self.get_parameter('lookahead').get_parameter_value().double_value

        timer_rate = 20 # rate at which we publish the drive command
        self.create_timer(1/timer_rate, self.timer_drive_pub_callback)

        self.get_logger().info("Parking Controller Initialized")

    def timer_drive_pub_callback(self):
        """Calculates and publishes the drive command at a specific frequency. """
        if self.relative_x is not None and self.relative_y is not None:
            # only calculate with the relevant pose
            drive_cmd = AckermannDriveStamped()
            # self.get_logger().info(f'New Drive Command')

            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'base_link'
            drive_cmd.header = header

            # choose target at the lookahead distance
            target_point = self.get_point_on_line((self.relative_x, self.relative_y), self.LOOKAHEAD)

            pure_persuit_drive_cmd = self.update_control(target_point) # get the drive command w speed and steer

            drive_cmd.drive = pure_persuit_drive_cmd

            self.drive_pub.publish(drive_cmd)

    def relative_cone_callback(self, msg):
        """Caches the pose of the intersection and calculates new drive command"""
        self.relative_x = msg.x
        self.relative_y = msg.y


    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################

        # YOUR CODE HERE
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)
        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = np.sqrt(self.relative_x ** 2 + self.relative_y ** 2)

        #################################
        self.error_pub.publish(error_msg)


    def update_control(self, target_point):
        """
        Returns the ackerman drive command
        """
        drive = AckermannDrive()
        # TODO: if there's a way to know if we lost the lines to give a stop command?
        # in the case that the cone is behind the car, can also be modified for when we don't see the car
        # if self.relative_x < 0:
        #     drive.speed = -0.5
        #     # steer toward the cone while reversing
        #     drive.steering_angle = float(np.clip(
        #         -np.sign(self.relative_y) * self.MAX_STEERING_ANGLE * 0.6,
        #         -self.MAX_STEERING_ANGLE,
        #         self.MAX_STEERING_ANGLE
        #     ))
        #     return drive


        # Check to see if we are too close
        # goal_dist = np.sqrt(self.relative_x**2 + self.relative_y**2)

        # # if we are in the stopping range and pointed at the cone, it's okay
        # if goal_dist < self.parking_distance_max and goal_dist > self.parking_distance_min:
        #     drive.speed = 0.0
        #     drive.steering_angle = 0.0
        #     return drive

        # calculate with the pure persuit
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



    def get_point_on_line(self, p2: tuple(float, float), lookahead_dist: float, p1: tuple(float, float) = (0,0)):
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
        new_point_msg = Point(x=new_point_vec[0], y=new_point_vec[1])

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
