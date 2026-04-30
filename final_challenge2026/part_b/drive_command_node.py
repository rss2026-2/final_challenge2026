import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Header, String


class SimpleDrivePublisher(Node):
    """
    A node that publishes simple drive commands to a given topic for testing purposes
    """

    def __init__(self):
        super().__init__("image_publisher")

        ### -- Declared parameters (Start) -- ###
        # -- ROS2 Topics
        self.declare_parameter("publish_topic", "/drive")
        self.declare_parameter("subscribe_topic", "/publish_drive")

        self.publish_topic = self.get_parameter("publish_topic").get_parameter_value().string_value
        self.subscribe_topic = self.get_parameter("subscribe_topic").get_parameter_value().string_value

        # -- Dynamic Params
        self.declare_parameter("speed", 1.0)

        self.speed = self.get_parameter("speed").get_parameter_value().double_value
        ### -- Declared parameters (End) -- ###

        ### -- Publishers and Subscribers (Start) -- ###
        # -- Pubs
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.publish_topic, 10)

        # -- Subs
        self.command_sub = self.create_subscription(String, self.subscribe_topic, self.publish_drive_cb, 1)
        ### -- Publishers and Subscribers (End) -- ###

        self.get_logger().info("=== Simple Drive Publisher Node Initialized ===")

    def publish_drive_cb(self, str_msg):
        """
        Receives drive commands in the form of string messages
        and publishes them to the car.

        How to send a single (message): #/publish_drive is the default for subscribe_topic, replace if changed
            ros2 topic pub -1 /publish_drive std_msgs/msg/String "{data: (message)}"
        
        Accepted Messages:
            'forward'
            'reverse'
            'stop'

        Example:
            ros2 topic pub -1 /publish_drive std_msgs/msg/String "{data: 'forward'}"

        Args:
            str_msg (ROS2 String): contains the command in its data

        """
        drive_cmd = AckermannDriveStamped()

        header = Header()
        stamp = self.get_clock().now().to_msg()
        header.stamp = stamp
        header.frame_id = 'base_link'
        drive_cmd.header = header

        command = str_msg.data
        self.get_logger().info(f"Received drive command: {command}")
        if command == "forward":
            drive_cmd.drive.speed = self.speed
        elif command == "reverse":
            drive_cmd.drive.speed = -self.speed
        elif command == "stop":
            drive_cmd.drive.speed = 0.0
        else:
            self.get_logger().warn(f"### Drive command not recognized ###")
            return
        
        self.drive_pub.publish(drive_cmd)

    

def main(args=None):
    rclpy.init(args=args)
    drive_publisher = SimpleDrivePublisher()
    rclpy.spin(drive_publisher)
    rclpy.shutdown()

if __name__ == "__main__":
    main()