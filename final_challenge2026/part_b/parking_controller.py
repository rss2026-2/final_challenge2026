#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Header, Bool

from visualization_msgs.msg import Marker

from viz_utils.visualization_tools import VisualizationTools


class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """

    def __init__(self):
        super().__init__("parking_controller")

        # -- Declared parameters --
        self.declare_parameter("drive_topic", "/pc_drive")
        self.declare_parameter("point_topic", "/pc_relative_point")
        self.declare_parameter("path_marker_topic", "/pc_path")
        self.declare_parameter("parking_error_topic", "/parking_error")
        
        self.declare_parameter("car_length", 0.325)
        self.declare_parameter("max_steering_angle", 0.34)
        self.declare_parameter("velocity", 0.5)
        self.declare_parameter("lookahead", 0.8)
        self.declare_parameter("error_epsilon", 0.05)
        self.declare_parameter("parking_distance_min", 0.75)
        self.declare_parameter("parking_distance_max", 1.10)
        self.declare_parameter("steering_angle_thresh", 1.2)
        
        self.drive_topic = self.get_parameter('drive_topic').value  # set in launch file; different for simulator vs racecar        
        self.point_topic = self.get_parameter('point_topic').value
        self.path_marker_topic = self.get_parameter('path_marker_topic').value
        self.parking_error_topic = self.get_parameter('parking_error_topic').value
        
        self.car_length = self.get_parameter('car_length').get_parameter_value().double_value
        self.max_steering_angle = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.velocity = self.get_parameter('velocity').get_parameter_value().double_value
        self.lookahead = self.get_parameter('lookahead').get_parameter_value().double_value
        self.epsilon = self.get_parameter('error_epsilon').get_parameter_value().double_value
        self.steering_angle_thresh = self.get_parameter('steering_angle_thresh').get_parameter_value().double_value

        # -- Publishers and subscribers --
        self.point_sub = self.create_subscription(ConeLocation, self.point_topic, self.relative_cone_callback, 1)
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.parking_error_pub = self.create_publisher(ParkingError, self.parking_error_topic, 10)
        self.marker_pub = self.create_publisher(Marker, '/drive_to_point', 10)
        self.path_marker_pub = self.create_publisher(Marker, self.path_marker_topic, 10)
        
        # -- Initialized variables --
        self.drive_cmd = None
        self.proximity_check = False
        self.relative_x, self.relative_y = 0.0, 0.0

        timer_rate = 20
        self.create_timer(1/timer_rate, self.timer_callback)

        self.get_logger().info("Parking Controller Initialized")

    def timer_callback(self):
        if self.drive_cmd is not None:
            self.drive_pub.publish(self.drive_cmd)
            self.error_publisher()

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()

        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd


        """We aren't aiming to give you a specific algorithm to run your controller, and we encourage you to play around. Try answering these questions:

        1. What should the robot do if the cone is far in front?
            Pure Persuit
        2. What should the robot do if it is too close?
            Reverse
        3. What if the robot isn't too close or far, but the cone isn't directly in front of the robot?
            reverse and steer towards, then go straight towards
        4. How can we keep the cone in frame when we are using our real camera?
            dont give super exageerated angles, don't go past"""

        # self.get_logger().info(f'New Drive Command')

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'
        drive_cmd.header = header

        # Generate spline path
        path = self.generate_hermite_path(self.relative_x, self.relative_y)
        # self.get_logger().info(f'Path: {path[0]}')

        # visualize path
        x, y = zip(*path)
        # VisualizationTools.plot_line(list(x), list(y), self.path_marker_pub)

        goal_dist = np.sqrt(self.relative_x**2 + self.relative_y**2)

        self.lookahead = max(0.3, min(1.2, 0.5 * goal_dist))
        # self.get_logger().info(f'{self.lookahead=}')


        # choose lookahead target
        target_point = self.get_lookahead_point(path)

        pure_persuit_drive_cmd = self.update_control(target_point)

        drive_cmd.drive = pure_persuit_drive_cmd
        #################################

        self.drive_cmd = drive_cmd

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

        self.parking_error_pub.publish(error_msg)

    def generate_hermite_path(self, cone_x, cone_y, tangent_strength=None, num_points=50):
        """
        Calculates a smooth path to a cone in the car's local frame.
        - cone_x, cone_y: Position of the cone relative to the car.
        - tangent_strength: Controls 'curviness'. Default is 1.5x the distance.
        """

        # 1. Start: Car is at (0,0) facing +X
        p0 = np.array([0.0, 0.0])

        # 2. Goal: Find the point parking_distance in front of the cone
        phi = np.arctan2(cone_y, cone_x) # Angle from car to cone
        p1 = np.array([
            cone_x, #- self.parking_distance_min * np.cos(phi),
            cone_y # - self.parking_distance_min * np.sin(phi)
        ])

        # 3. Tangents: Direction car should be moving at start and end
        # Strength (L) affects how wide the turn is
        dist_to_goal = np.linalg.norm(p1 - p0)
        L = tangent_strength if tangent_strength else dist_to_goal * 1.5

        m0 = np.array([L, 0.0])              # Start tangent: straight forward
        # m1 = np.array([L * np.cos(phi), L * np.sin(phi)]) # End tangent: facing cone
        m1 = np.array([cone_x, cone_y])

        # 4. Generate the Spline points
        t = np.linspace(0, 1, num_points)
        path = []
        for val in t:
            # Cubic Hermite Basis Functions
            h00 = 2*val**3 - 3*val**2 + 1
            h10 = val**3 - 2*val**2 + val
            h01 = -2*val**3 + 3*val**2
            h11 = val**3 - val**2

            # Calculate point on the curve
            point = h00*p0 + h10*m0 + h01*p1 + h11*m1
            path.append(point)


        return np.array(path)

    def get_lookahead_point(self, path):
        """
        Returns the first point on the path at least lookahead distance away.
        """
        # for p in path:
        #     if np.linalg.norm(p) > self.lookahead:
        #         return p

        # return path[-1]
        dists = np.linalg.norm(path, axis=1)
        closest_idx = np.argmin(dists)

        for i in range(closest_idx, len(path)):
            if np.linalg.norm(path[i]) > self.lookahead:
                return path[i]

        point = path[-1]

        self.draw_marker(point[0], point[1], 'base_link')

        return point

    def update_control(self, target_point):
        """
        Returns the ackerman drive command
        """
        drive = AckermannDrive()
        # in the case that the cone is behind the car, can also be modified for when we don't see the car
        if self.relative_x < 0:
            self.get_logger().info("it's behind us, reverse")
            drive.speed = -0.5
            # steer toward the cone while reversing
            drive.steering_angle = float(np.clip(
                -np.sign(self.relative_y) * self.max_steering_angle * 0.6,
                -self.max_steering_angle,
                self.max_steering_angle
            ))
            return drive


        # Check to see if we are too close
        goal_dist = np.sqrt(self.relative_x**2 + self.relative_y**2)

        # if we are in the stopping range and pointed at the cone, it's okay
        if goal_dist < self.parking_distance_max and goal_dist > self.parking_distance_min:
            # TODO add something here if we are too close but not pointed well...
            self.get_logger().info("we are in the stopping range and pointed at the cone")
            drive.speed = 0.0
            drive.steering_angle = 0.0
            return drive

        # calculate with the pure persuit
        new_steering_angle = self.compute_feedback_angle(target_point)

        # If the turn we have to make is too tight or the cone is cut off, or the cone is just plainly too close, reverse first
        reason_for_reversing = ""
        
        turning_angle_too_tight = abs(new_steering_angle) > self.max_steering_angle * self.STEERING_ANGLE_THRESH
        if turning_angle_too_tight: 
            reason_for_reversing += "TURNING ANGLE TOO TIGHT"
        
        cone_too_close = goal_dist < self.parking_distance_min
        if cone_too_close: 
            reason_for_reversing += ", TOO CLOSE"
            
        if turning_angle_too_tight or cone_too_close:
            self.get_logger().info(f'{reason_for_reversing}; steering_angle: {new_steering_angle}' )
            drive.speed = -0.5
            reverse_angle = -0.5 * new_steering_angle
            drive.steering_angle = float(np.clip(reverse_angle,
                                     -self.max_steering_angle,
                                     self.max_steering_angle))

        else: # if it is in front of us reasonable angle, give it that angle
            # self.get_logger().info("just drivin")
            drive.steering_angle = float(np.clip(new_steering_angle,
                                    -self.max_steering_angle,
                                    self.max_steering_angle))

            drive.speed = self.velocity # no longer by mode and proximity 

        return drive

    def compute_feedback_angle(self, target_point):
        """

        """
        lookahead_dist = np.linalg.norm(target_point)

        # pure pursuit steering law
        delta = np.arctan2(
            2 * self.car_length * target_point[1],
            lookahead_dist**2
        )

        # delta = np.clip(delta, -self.max_steering_angle, self.max_steering_angle)
        # not clipping because we want to check potential reverse behavior
        return delta

    # def get_speed_by_mode_and_proximity(self, distance_to_obj):
    #     # for the cone, we want to slow down as we appraoch the cone
    #     if self.DETECTION_MODE == "cone":
    #         # slow down near goal
    #         if distance_to_obj > 1.5:
    #             return self.velocity
    #         else:
    #             return 0.5
    #     # for the line detection mode, don't want to scale speed because we want to go full speed always
    #     else:
    #         return self.velocity

    def draw_marker(self, cone_x, cone_y, message_frame):
        """
        Publish a marker to represent the cone in rviz.
        (Call this function if you want)
        """
        marker = Marker()
        marker.header.frame_id = message_frame
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = .2
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = cone_x
        marker.pose.position.y = cone_y
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
