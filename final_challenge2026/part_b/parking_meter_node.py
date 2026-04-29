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
        self.declare_parameter('parking_meter_topic', '/parking_meter')
        # self.declare_parameter('parking_topic', '/is_parking')

        self.pm_drive_topic = self.get_parameter('pm_drive_topic').value
        self.parking_meter_topic = self.get_parameter('parking_meter_topic').value
        # self.is_parking_topic = self.get_parameter('is_parking').value
        
        # -- Publishers and subscribers --
        self.parking_meter_sub = self.create_subscription(Image, self.parking_meter_topic, self.parking_meter_callback, 1)

        self.pm_drive_pub = self.create_publisher(AckermannDriveStamped, self.pm_drive_topic, 10)

        # -- Initialized variables --

        # Variable for cached parking meters to ignore
        # Variable for if we've already finished parking or not (when set to true, do the save img/wait/leave behavior)

        self.get_logger().info("=== Parking Meter Initialized ===")

    def parking_meter_callback(self, msg):
        """
        Handles parking meter behavior when we see one
        """
        # TODO: check cache to ensure we haven't already parked here

        # TODO: perform homography on parking meter

        # TODO: park at the meter

        # TODO: take a picture

        # TODO: wait 5sec

        # TODO: stop sending zero drive command and cache parking meter to ignore it