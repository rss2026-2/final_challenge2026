#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseArray, Point
from sensor_msgs.msg import Image

class LineDetector(Node):
    """
    Uses hough line detector to outline the track lines. Estimates the left and right lines
    of the track. Publishes a goal point that car can follow with pure pursuit.
    """

    def __init__(self):
        super().__init__("line_detector")
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('debug_topic', '/track_lines')
        self.declare_parameter('low_threshold', 50)
        self.declare_parameter('high_threshold', 150)
        self.declare_parameter('direction', 'left')
        self.declare_parameter('goal_y_offset', 70)
        self.declare_parameter('goal_topic', '/goal_point')

        self.debug_topic = self.get_parameter('debug_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.goal_topic = self.get_parameter('goal_topic').value
        self.low_threshold = self.get_parameter('low_threshold').value
        self.high_threshold = self.get_parameter('high_threshold').value
        self.direction = self.get_parameter('direction').value
        self.goal_y_offset = self.get_parameter('goal_y_offset').value

        self.image_sub = self.create_subscription(Image, self.image_topic, self.hough_fallback, 5)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)
        self.goal_pub = self.create_publisher(Point, self.goal_topic, 10)
        self.bridge = CvBridge()

        # cache lanes in case of frames dropping
        self.last_left_line = None
        self.last_right_line = None
        self.last_lane_width = None
        self.last_goal_msg = None
        self.last_goal_dbg = None
        self.smoothed_left_line = None
        self.goal_y_ref = None
        self.goal_y_tolerance = 80.0
        self.line_alpha = 0.25
        self.goal_alpha = 0.18
        self.goal_lane_offset_ratio = 0.23

        # use this line to debug with static images:
        # self.load_and_publish_image('src/final_challenge2026/racetrack_images/lane_3/image45.png')

        self.get_logger().info(f"Line Detector Node Initialized - Publishing Debug Image to '{self.debug_topic}'")

### ----------------- LINE DETECTOR  ----------------- ####

    def hough_fallback(self, msg):
        """
        Inputs image from ZED camera, highlights the track lines in the image.
        1) segments by HSV values for white
        2) uses Canny to find edges
        3) filters the top 20% of the image out
        4) uses hough lines to generate probabilistic distribution of all lines in the image
        5) draws lines over the image

        :param msg: ROS2 Image message
        :returns None
        """
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        lane_image = np.copy(image)
        hsv = cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        edges = cv2.Canny(mask, 75, 150)

        height, width = edges.shape
        # polygon = np.array([[(0, height), (width, height), (width, 0), (0,0)]])
        if self.direction == "left":
            polygon = np.array([[(int(width * 0.1), int(height * 0.8)), (int(width * 0.95), int(height * 0.8)), (int(width * 0.90), int(height * 0.4)), (int(width * 0.2), int(height * 0.4))]])
        else:
            polygon = np.array([[(int(width * 0.05), int(height * 0.8)), (int(width * 0.9), int(height * 0.8)), (int(width * 0.8), int(height * 0.4)), (int(width * 0.1), int(height * 0.4))]])

        black_mask = np.zeros_like(edges)
        cv2.fillPoly(black_mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, black_mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=100, minLineLength=40, maxLineGap=10)

        # goal point
        self.find_goal(lines, lane_image)

### ----------------- GOAL FINDER  ----------------- ####

    def find_goal(self, lines, image):
        """
        Stable left-turn lane tracking.

        We fit only the left boundary, smooth that fit across frames, and
        derive a goal point by shifting a fixed distance to the right of the
        smoothed boundary. This avoids the instability from trying to solve
        both lane boundaries every frame.
        """
        if lines is None:
            if self.last_goal_msg is not None and self.last_goal_dbg is not None:
                self.goal_pub.publish(self.last_goal_msg)
                self.publish_debug_image(self.last_goal_dbg)
                return self.last_goal_msg
            self.get_logger().info("ERROR: Detected no lines.")
            return

        h, w = image.shape[:2]
        y_bot = h - 1
        left_segments = []

        mid_x = w / 2.0
        for x1, y1, x2, y2 in [s[0] for s in lines]:
            if abs(np.arctan2(y2 - y1, x2 - x1)) >= np.deg2rad(15) and y2 != y1:
                x_bot = x1 + (y_bot - y1) * (x2 - x1) / (y2 - y1)
                if x_bot < mid_x:
                    left_segments.append((x1, y1, x2, y2))

        dbg = image.copy()
        for s in left_segments:
            cv2.line(dbg, s[:2], s[2:], (255, 0, 0), 2)

        if not left_segments and self.smoothed_left_line is None:
            if self.last_goal_msg is not None and self.last_goal_dbg is not None:
                self.goal_pub.publish(self.last_goal_msg)
                self.publish_debug_image(self.last_goal_dbg)
                return self.last_goal_msg
            self.get_logger().info("ERROR: No stable left lane yet.")
            return

        # Fit a single stable left boundary from all candidate segments.
        if left_segments:
            pts = []
            for s in left_segments:
                pts.append([s[0], s[1]])
                pts.append([s[2], s[3]])
            pts = np.array(pts, dtype=np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            if abs(vy) < 1e-6:
                if self.last_goal_msg is not None and self.last_goal_dbg is not None:
                    self.goal_pub.publish(self.last_goal_msg)
                    self.publish_debug_image(self.last_goal_dbg)
                    return self.last_goal_msg
                return

            goal_y = int(np.clip(h - 1 - self.goal_y_offset, int(h * 0.35), h - 1))
            y_top = int(np.clip(goal_y - 80, 0, h - 1))

            x_bot = float(x0 + (y_bot - y0) * vx / vy)
            x_goal_line = float(x0 + (goal_y - y0) * vx / vy)
            x_top = float(x0 + (y_top - y0) * vx / vy)

            new_left_line = np.array([[x_bot, y_bot], [x_top, y_top]], dtype=np.float32)
            if self.smoothed_left_line is None:
                self.smoothed_left_line = new_left_line
            else:
                self.smoothed_left_line = (
                    self.line_alpha * new_left_line +
                    (1.0 - self.line_alpha) * self.smoothed_left_line
                )
        else:
            # No new detection this frame, reuse the last stable line.
            new_left_line = self.smoothed_left_line
            goal_y = int(np.clip(h - 1 - self.goal_y_offset, int(h * 0.35), h - 1))
            y_top = int(np.clip(goal_y - 80, 0, h - 1))

        left_line = self.smoothed_left_line if self.smoothed_left_line is not None else new_left_line
        (x_bot, y_bot_i), (x_top, y_top_i) = left_line
        if abs(y_top_i - y_bot_i) < 1e-6:
            if self.last_goal_msg is not None and self.last_goal_dbg is not None:
                self.goal_pub.publish(self.last_goal_msg)
                self.publish_debug_image(self.last_goal_dbg)
                return self.last_goal_msg
            return None, None

        # Interpolate the left boundary at the goal row, then shift right into the lane.
        t = (goal_y - y_bot_i) / (y_top_i - y_bot_i)
        x_on_left = x_bot + t * (x_top - x_bot)
        lane_offset_px = max(50.0, w * self.goal_lane_offset_ratio)
        goal_x = np.clip(x_on_left + lane_offset_px, 0, w - 1)

        goal_msg = Point(x=float(goal_x), y=float(goal_y), z=0.0)
        if self.smoothed_left_line is not None:
            left_debug = self.smoothed_left_line.astype(np.int32)
            cv2.line(dbg, tuple(left_debug[0]), tuple(left_debug[1]), (255, 0, 255), 4)
        cv2.circle(dbg, (int(goal_x), int(goal_y)), 7, (255, 255, 0), -1)

        if self.last_goal_msg is None:
            self.last_goal_msg = goal_msg
            self.last_goal_dbg = dbg
        else:
            self.last_goal_msg = Point(
                x=float(self.goal_alpha * goal_msg.x + (1.0 - self.goal_alpha) * self.last_goal_msg.x),
                y=float(self.goal_alpha * goal_msg.y + (1.0 - self.goal_alpha) * self.last_goal_msg.y),
                z=0.0,
            )
            self.last_goal_dbg = dbg

        self.last_left_line = tuple(map(int, [int(x_bot), int(y_bot_i), int(x_top), int(y_top_i)]))
        self.goal_y_ref = goal_y
        self.goal_pub.publish(self.last_goal_msg)
        self.publish_debug_image(dbg)
        return self.last_goal_msg

### ----------------- PUBLISHERS  ----------------- ####

    def publish_lines(self, image, lines):
        """
        Helper to publish an image overlaying all lines detected on an image.

        :params image: Numpy Array stores image pixel data
        :params lines: Numpy Array stores endpoints for each line

        :returns None
        """
        if lines is None:
            return

        line_image = np.zeros_like(image)
        try:
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        except:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
        self.publish_debug_image(combo_image)

    def publish_debug_image(self, image):
        """
        Helper that publishes an image to the debug topic.

        :params image: Numpy Array that stores pixel information of a picture
        :returns None
        """
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        except:
            debug_msg = self.bridge.cv2_to_imgmsg(image, "mono8")
        self.debug_pub.publish(debug_msg)

    def load_and_publish_image(self, path_name):
        """
        Helper for debugging. Publishes an image from a path_name

        :param path_name: Str for the file_path
        :returns None
        """
        image = cv2.imread(path_name)
        lane_image = np.copy(image)
        self.get_logger().info("Publishing source image...")
        self.publish_debug_image(lane_image)


def main(args=None):
    rclpy.init(args=args)
    line_detector = LineDetector()
    rclpy.spin(line_detector)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
