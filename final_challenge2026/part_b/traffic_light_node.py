import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Header, Bool
from ackermann_msgs.msg import AckermannDriveStamped

class TrafficLight(Node):
    """
    A node that handles traffic light behavior and updates the
    drive command accordingly.
    """

    def __init__(self):
        super().__init__("traffic_light")

        # -- Declared parameters --
        self.declare_parameter('tl_drive_topic', '/vesc/high_level/input/nav_0')
        self.declare_parameter('traffic_light_topic', '/traffic_light')
        self.declare_parameter('traffic_light_only_topic', '/traffic_light_only')

        self.tl_drive_topic = self.get_parameter('tl_drive_topic').value
        self.traffic_light_topic = self.get_parameter('traffic_light_topic').value
        self.traffic_light_only_topic = self.get_parameter('traffic_light_only_topic').value

        # -- Publishers and subscribers --
        self.traffic_light_sub = self.create_subscription(Image, self.traffic_light_topic, self.traffic_light_callback, 1)
        self.traffic_light_only_sub = self.create_subscription(Image, self.traffic_light_only_topic, self.traffic_light_only_callback, 1)
        
        self.tl_drive_pub = self.create_publisher(AckermannDriveStamped, self.tl_drive_topic, 10)

        # -- Initialized variables --
        self.traffic_light_close = False

        self.get_logger().info("=== Traffic Light Node Initialized ===")
    
    def traffic_light_callback(self, msg):
        """
        Looks for how far away the traffic light is before handling red light signal behavior
        """
        # TODO: Perform homography on traffic light
        
        # If a traffic light is close enough,
            # Set traffic_light_close to True
            # self.traffic_light_close = True
    
    def traffic_light_only_callback(self, msg):
        """
        Looks for whether the signal is a red light or not
        """
        # If we're not even close to the traffic light avoid this callback
        if not self.traffic_light_close:
            return
        
        # TODO: Call color segmentation function to determine traffic light signal status
        # If red light is detected, publish stop
        # self.publish_stop()
        
        # TODO: Maybe handle slowing down for a yellow light? self.publish_slow()???
        pass

    def color_segmentation_CHANGE_MY_NAME(self):
        """
        Color segmentation (change me)

        Returns integer determining whether a red, (yellow???), or no light was detected
        """
        # TODO: Write color segmentation behavior
        pass

    def publish_stop(self, angle=None, frame='base_link'):
        """
        Publishes a command for the car to stop.
        """
        new_msg = AckermannDriveStamped()
        
        header = Header()
        header.stamp = self.get_clock().now()
        header.frame_id = 'base_link'
        new_msg.header = header

        drive_command = new_msg.drive
        drive_command.speed = 0.0
        drive_command.acceleration = 0.0
        drive_command.jerk = 0.0
        if angle is not None:
            drive_command.steering_angle = angle

        self.tl_drive_pub.publish(new_msg)

def main(args=None):
    rclpy.init(args=args)
    light_detector = TrafficLight()
    rclpy.spin(light_detector)
    rclpy.shutdown()


if __name__ == '__main__':
    main()