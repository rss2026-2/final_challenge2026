import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge
from final_challenge2026.part_b.computer_vision.color_segmentation import find_most_prominent_color

from sensor_msgs.msg import Image
from std_msgs.msg import Header, Bool
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point

class TrafficLight(Node):
    """
    A node that handles traffic light behavior and updates the
    drive command accordingly.
    """

    def __init__(self):
        super().__init__("traffic_light")

        ### -- Declared parameters (Start) -- ###
        # -- ROS2 Topics
        self.declare_parameter('tl_drive_topic', '/vesc/high_level/input/nav_0')
        self.declare_parameter('tl_point_topic', '/tl_relative_point')
        # self.declare_parameter('traffic_light_topic', '/traffic_light')
        self.declare_parameter('traffic_light_topic', '/zed/zed_node/rgb/image_rect_color')
        # self.declare_parameter('red_light_topic', '/red_light')

        self.tl_drive_topic = self.get_parameter('tl_drive_topic').get_parameter_value().string_value
        self.tl_point_topic = self.get_parameter('tl_point_topic').get_parameter_value().string_value
        self.traffic_light_topic = self.get_parameter('traffic_light_topic').get_parameter_value().string_value
        # self.red_light_topic = self.get_parameter('red_light_topic').get_parameter_value().string_value
        ### -- Declared parameters (End) -- ###

        ### -- Publishers and Subscribers (Start) -- ###
        # -- Pubs
        self.tl_drive_pub = self.create_publisher(AckermannDriveStamped, self.tl_drive_topic, 10)
        # self.red_light_pub = self.create_publisher(Bool, self.red_light_topic, 10)
        self.image_debug_pub = self.create_publisher(Image, "/debug_img", 10)

        # -- Subs
        self.tl_point_sub = self.create_subscription(Point, self.tl_point_topic, self.tl_point_callback, 1)
        self.traffic_light_sub = self.create_subscription(Image, self.traffic_light_topic, self.traffic_light_callback, 1)
        # self.red_light_sub = self.create_subscription(Bool, self.red_light_topic, self.red_light_callback, 1)
        ### -- Publishers and Subscribers (End) -- ###

        self.CLOSE_DIST = 3.0 # change me
        self.traffic_light_close = False

        
        self.bridge = CvBridge()
        self.get_logger().info("=== Traffic Light Node Initialized ===")

    def tl_point_callback(self, msg):
        """
        Point callback that checks to see if the traffic light is close enough to consider
        performing color segmentation on
        """
        if msg.x < self.CLOSE_DIST:
            self.traffic_light_close = True
        else:
            self.traffic_light_close = False

    def traffic_light_callback(self, image_msg):
        """
        Image callback that scans for a red light on a traffic light
        """
        self.get_logger().info("Image received!")
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # Perform color segmentation on the spot where the red light would be
        tf_color = self.tf_color_detection(image)

        if tf_color is not None:
            self.get_logger().info(f"Color Found: {tf_color}")
        else:
            self.get_logger().info("No colors found")
            
        if tf_color == "red":
            # If we see a red light, 
            # Publish true to /red_light (or just call publish_stop())
            bool_msg = Bool()
            bool_msg.data = True

            # self.red_light_pub.publish(bool_msg)
            if self.traffic_light_close:
                self.publish_stop()

        # Probably no yellow light
        # elif tf_color == "yellow":
        #     # TODO: Maybe handle slowing down for a yellow light?
        #     # for now, just don't stop
        #     bool_msg = Bool()
        #     bool_msg.data = False
        #     # self.red_light_pub.publish(bool_msg)
        #     pass

        elif tf_color == "green":
            # If we see a green light, 
            # Publish false to /red_light
            bool_msg = Bool()
            bool_msg.data = False
            # self.red_light_pub.publish(bool_msg)
            pass
            
        else:
            # TODO: What to do if we couldn't detect any of the colors,
            # which prob won't happen
            # for now just don't stop
            bool_msg = Bool()
            bool_msg.data = False
            # self.red_light_pub.publish(bool_msg) 
            pass

    # def red_light_callback(self, msg):
    #     """
    #     Looks for whether the signal is a red light or not
    #     """
    #     # If we're not even close to the traffic light avoid this callback
    #     if not self.traffic_light_close:
    #         return

    #     # TODO: Give the angle to publish_stop through a drive callback or 
    #     # topic that publishes only the angle
    #     self.publish_stop()
    #     self.get_logger().info("Published TL stop command")

    def tf_color_detection(self, image):
        """
        Color segmentation (change me)

        Returns integer determining whether a red, (yellow???), or no light was detected
        """

        red_hsv_range_1 = [[0, 70, 75], [10, 90, 100]]
        red_hsv_range_2 = [[350, 70, 75], [360, 90, 100]]

        green_hsv_range = [[140, 40, 75],[180, 100, 100]]

        # yellow_hsv_range = [[48, 40, 70],[60, 90, 100]]

        color_dict = {
            "red": [red_hsv_range_1, red_hsv_range_2],
            "green": [green_hsv_range],
            # "yellow": [yellow_hsv_range]
        }

        colors_to_draw = {
            "red": (0,0,255),
            "green": (0,255,0),
            # "yellow": (0,255,255)
        }
        img_copy = image.copy()
        ret_color = find_most_prominent_color(image, color_dict, img_copy, colors_to_draw)

        image_msg = self.bridge.cv2_to_imgmsg(img_copy, "bgr8")
        self.image_debug_pub.publish(image_msg)
        
        return ret_color

    def publish_stop(self, angle=None, frame='base_link'):
        """
        Publishes a command for the car to stop.
        """
        new_msg = AckermannDriveStamped()
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
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