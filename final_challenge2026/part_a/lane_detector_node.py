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
        self.last_goal_msg = None
        self.last_goal_dbg = None
        self.left_fit = None
        self.right_fit = None
        self.line_alpha = 0.22
        self.goal_alpha = 0.2
        self.goal_row_ratio = 0.62

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

        We fit the left and right lane boundaries independently as x(y)
        lines, smooth each fit across frames, and then publish a goal point
        on a fixed image row at the center of the lane.
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
        goal_y = int(np.clip(h * self.goal_row_ratio, int(h * 0.35), h - 1))
        left_segments = []
        right_segments = []

        mid_x = w / 2.0
        for x1, y1, x2, y2 in [s[0] for s in lines]:
            if abs(np.arctan2(y2 - y1, x2 - x1)) >= np.deg2rad(15) and y2 != y1:
                x_bot = x1 + (y_bot - y1) * (x2 - x1) / (y2 - y1)
                if x_bot < mid_x:
                    left_segments.append((x1, y1, x2, y2))
                else:
                    right_segments.append((x1, y1, x2, y2))

        dbg = image.copy()
        for s in left_segments:
            cv2.line(dbg, s[:2], s[2:], (255, 0, 0), 2)
        for s in right_segments:
            cv2.line(dbg, s[:2], s[2:], (0, 255, 0), 2)

        left_fit = self._fit_and_smooth_boundary(left_segments, self.left_fit)
        right_fit = self._fit_and_smooth_boundary(right_segments, self.right_fit)

        if left_fit is not None:
            self.left_fit = left_fit
        if right_fit is not None:
            self.right_fit = right_fit

        if self.left_fit is None or self.right_fit is None:
            if self.last_goal_msg is not None and self.last_goal_dbg is not None:
                self.goal_pub.publish(self.last_goal_msg)
                self.publish_debug_image(self.last_goal_dbg)
                return self.last_goal_msg
            self.get_logger().info("ERROR: No stable lane pair yet.")
            return

        goal_x = self._lane_center_x_at_y(goal_y)
        if goal_x is None:
            if self.last_goal_msg is not None and self.last_goal_dbg is not None:
                self.goal_pub.publish(self.last_goal_msg)
                self.publish_debug_image(self.last_goal_dbg)
                return self.last_goal_msg
            return None, None

        goal_msg = Point(x=float(goal_x), y=float(goal_y), z=0.0)

        self._draw_boundary(dbg, self.left_fit, (255, 0, 255))
        self._draw_boundary(dbg, self.right_fit, (0, 255, 255))
        cv2.circle(dbg, (int(goal_x), int(goal_y)), 7, (255, 255, 0), -1)

        if self.last_goal_msg is None:
            self.last_goal_msg = goal_msg
        else:
            self.last_goal_msg = Point(
                x=float(self.line_alpha * goal_msg.x + (1.0 - self.line_alpha) * self.last_goal_msg.x),
                y=float(goal_msg.y),
                z=0.0,
            )
        self.last_goal_dbg = dbg

        self.goal_pub.publish(self.last_goal_msg)
        self.publish_debug_image(dbg)
        return self.last_goal_msg

    def _fit_and_smooth_boundary(self, segments, prev_fit):
        if not segments:
            return prev_fit

        pts = []
        for s in segments:
            pts.append([s[0], s[1]])
            pts.append([s[2], s[3]])
        pts = np.array(pts, dtype=np.float32)
        ys = pts[:, 1]
        xs = pts[:, 0]
        if np.ptp(ys) < 1e-6:
            return prev_fit

        m, b = np.polyfit(ys, xs, 1)
        fit = np.array([float(m), float(b)], dtype=np.float32)

        if prev_fit is None:
            return fit
        return self.line_alpha * fit + (1.0 - self.line_alpha) * prev_fit

    def _x_at_y(self, fit, y):
        if fit is None:
            return None
        return float(fit[0] * y + fit[1])

    def _lane_center_x_at_y(self, y):
        left_x = self._x_at_y(self.left_fit, y)
        right_x = self._x_at_y(self.right_fit, y)
        if left_x is None or right_x is None:
            return None
        if left_x > right_x:
            left_x, right_x = right_x, left_x
        return float(0.5 * (left_x + right_x))

    def _draw_boundary(self, image, fit, color):
        if fit is None:
            return
        h, w = image.shape[:2]
        y1 = h - 1
        y2 = int(h * 0.35)
        x1 = int(np.clip(self._x_at_y(fit, y1), 0, w - 1))
        x2 = int(np.clip(self._x_at_y(fit, y2), 0, w - 1))
        cv2.line(image, (x1, y1), (x2, y2), color, 4)

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
