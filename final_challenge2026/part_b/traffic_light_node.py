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
        self.declare_parameter('red_light_topic', '/red_light')
        self.declare_parameter('timer_hz', 20)

        self.tl_drive_topic = self.get_parameter('tl_drive_topic').value
        self.red_light_topic = self.get_parameter('red_light_topic').value
        timer_hz = self.get_parameter('timer_hz').get_parameter_value().double_value

        # -- Publishers and subscribers --
        self.red_light_sub = self.create_subscription(Bool, self.red_light_topic, self.red_light_callback, 1)
        
        self.tl_drive_pub = self.create_publisher(AckermannDriveStamped, self.tl_drive_topic, 10)

        self.timer = self.create_timer(1/timer_hz, self.tl_detection_timer_callback)

        self.get_logger().info("=== Traffic Light Node Initialized ===")

    # TODO: Write behavior to detect a red light
    # probably don't need red_light_callback after that unless we want to use /red_light in other nodes
    
    def tl_detection_timer_callback(self):
        """
        Timer callback that scans for a red light on a traffic light
        """
        # If we see a traffic light with X% confidence,
        confidence = self.yolo_tf_detection_CHANTE_MY_NAME()
            # Perform color segmentation on the spot where the red light would be
            # If we see a red light, 
                # Publish true to /red_light (or just call publish_stop())
            
            # Maybe handle slowing down for a yellow light?

    def red_light_found_callback(self, msg):
        """
        Handles behavior for when our robot sees a red light.
        """
        # Don't do anything if we received a false boolean
        if msg.data == False:
            return

        # TODO: Give the angle to publish_stop through a drive callback or 
        # topic that publishes only the angle
        self.publish_stop()
        self.get_logger().info("Published TL stop command")

    def yolo_tf_detection_CHANTE_MY_NAME(self):
        """
        YOLO traffic light detection (change me)
        """
        # TODO: Write YOLO traffic light detection behavior
        # Probably always return a 0.0-1.0 number that is TL confidence %
        pass

    def color_segmentation_CHANGE_MY_NAME(self):
        """
        Color segmentation (change me)
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