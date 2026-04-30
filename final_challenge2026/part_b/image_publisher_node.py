import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Header, Bool

import cv2
from cv_bridge import CvBridge

class ImagePublisher(Node):
    """
    A node that publishes images to a given topic for testing purposes
    """

    def __init__(self):
        super().__init__("image_publisher")

        ### -- Declared parameters (Start) -- ###
        # -- ROS2 Topics
        self.declare_parameter("publish_topic", "/published_image")
        self.declare_parameter("subscribe_topic", "/publish_img")

        self.publish_topic = self.get_parameter("publish_topic").get_parameter_value().string_value
        self.subscribe_topic = self.get_parameter("subscribe_topic").get_parameter_value().string_value

        # -- Static Params
        self.declare_parameter("publish_rate", 20)
        self.declare_parameter("image_path", "src/final_challenge2026/final_challenge2026/part_b/computer_vision/red_light_1.jpg")

        self.publish_rate = self.get_parameter("publish_rate").get_parameter_value().integer_value
        self.image_path = self.get_parameter("image_path").get_parameter_value().string_value
        ### -- Declared parameters (End) -- ###

        ### -- Publishers and Subscribers (Start) -- ###
        # -- Pubs
        self.image_pub = self.create_publisher(Image, self.publish_topic, 10)

        # -- Subs
        self.publish_image_sub = self.create_subscription(Bool, self.subscribe_topic, self.publish_img_cb, 1)
        self.timer_sub = self.create_timer(1/self.publish_rate, self.publish_timer_cb)
        ### -- Publishers and Subscribers (End) -- ###

        self.publish_img = False
        self.load_image(self.image_path)

        self.bridge = CvBridge()
        self.get_logger().info("=== Image Publisher Node Initialized ===")

    def publish_img_cb(self, bool_msg):
        self.get_logger().info(f"Publish Image: {bool_msg.data}")
        self.publish_img = bool_msg.data

    def publish_timer_cb(self):
        if self.publish_img == True:

            if self.image is None:
                self.get_logger().info("No Image to publish. Setting image publisher to False.")
                self.publish_img = False
                return

            image_msg = self.bridge.cv2_to_imgmsg(self.image, "bgr8")
            self.image_pub.publish(image_msg)

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)

        if self.image is None:
            self.get_logger().warning(f"Failed to load image from file: {image_path}")

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    rclpy.shutdown()

if __name__ == "__main__":
    main()