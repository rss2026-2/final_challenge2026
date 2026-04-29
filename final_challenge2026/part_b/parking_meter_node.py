import rclpy
from rclpy.node import Node
import numpy as np

class ParkingMeter(Node):
    """
    A node that handles parking meter behavior and updates the
    drive command accordingly.
    """

    def __init__(self):
        super().__init__("parking_meter")

        # -- Declared parameters --
        self.declare_parameter('pm_drive_topic', '/vesc/high_level/input/nav_1')
        self.declare_parameter('pm_point_topic', '/pm_relative_point')
        self.declare_parameter('annotated_img_topic', '/yolo/annotated_image')
        # self.declare_parameter('parking_topic', '/is_parking')

        self.pm_drive_topic = self.get_parameter('pm_drive_topic').value
        self.pm_point_topic = self.get_parameter('pm_point_topic').value
        self.annotated_img_topic = self.get_parameter('annotated_img_topic').value
        # self.is_parking_topic = self.get_parameter('is_parking').value
        
        # -- Publishers and subscribers --
        self.pm_point_sub = self.create_subscription(Point, self.pm_point_topic, self.pm_point_callback, 1)
        self.annotated_img_sub = self.create_subscription(Image, self.annotated_img_topic, self.img_callback, 1)

        self.pm_drive_pub = self.create_publisher(AckermannDriveStamped, self.pm_drive_topic, 10)

        # -- Initialized variables --

        # Variable for cached parking meters to ignore
        # Variable for if we've already finished parking or not (when set to true, do the save img/wait/leave behavior)

        self.get_logger().info("=== Parking Meter Initialized ===")

    def pm_point_callback(self, msg):
        """
        Handles parking meter behavior when we see one
        """
        # TODO: check cache to ensure we haven't already parked here

        # TODO: perform homography on parking meter

        # TODO: park at the meter

        # TODO: take a picture

        # TODO: wait 5sec

        # TODO: stop sending zero drive command and cache parking meter to ignore it

    def img_callback(self, msg):
        """
        Image callback that handles saving the parking meter zed img 
        at the right time
        """
        pass