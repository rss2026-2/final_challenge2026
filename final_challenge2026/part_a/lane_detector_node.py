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
        self.declare_parameter("image_topic", "/zed/zed_node/rgb/image_rect_color")
        self.declare_parameter("debug_topic", "/track_lines")
        self.declare_parameter("low_threshold", 50)
        self.declare_parameter("high_threshold", 150)
        self.declare_parameter("direction", "left")
        self.declare_parameter("horizon_y_ratio", 0.54)
        self.declare_parameter("goal_topic", "/goal_point")

        self.debug_topic = self.get_parameter("debug_topic").value
        self.image_topic = self.get_parameter("image_topic").value
        self.goal_topic = self.get_parameter("goal_topic").value
        self.low_threshold = self.get_parameter("low_threshold").value
        self.high_threshold = self.get_parameter("high_threshold").value
        self.direction = self.get_parameter("direction").value
        self.horizon_y_ratio = self.get_parameter("horizon_y_ratio").value

        self.image_sub = self.create_subscription(Image, self.image_topic, self.hough_callback, 5)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)
        self.goal_pub = self.create_publisher(Point, self.goal_topic, 10)
        self.bridge = CvBridge()

        # cache lanes in case of frames dropping
        self.last_left_line = None
        self.last_right_line = None
        self.lane_width_sum_px = 0.0
        self.lane_width_count = 0
        self.avg_lane_width_px = 146.7

        # use this line to debug with static images:
        # self.load_and_publish_image('src/final_challenge2026/racetrack_images/lane_3/image45.png')

        self.get_logger().info(
            f"Line Detector Node Initialized - Publishing Debug Image to '{self.debug_topic}'"
        )

    ### ----------------- LINE DETECTOR  ----------------- ####

    def hough_callback(self, msg):
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

        edges = cv2.Canny(mask, self.low_threshold, self.high_threshold)

        height, width = edges.shape
        if self.direction == "left":
            polygon = np.array(
                [[
                    (int(width * 0.1), int(height * 0.8)),
                    (int(width * 0.95), int(height * 0.8)),
                    (int(width * 0.90), int(height * 0.4)),
                    (int(width * 0.2), int(height * 0.4)),
                ]]
            )
        else:
            polygon = np.array(
                [[
                    (int(width * 0.05), int(height * 0.8)),
                    (int(width * 0.9), int(height * 0.8)),
                    (int(width * 0.8), int(height * 0.4)),
                    (int(width * 0.1), int(height * 0.4)),
                ]]
            )

        black_mask = np.zeros_like(edges)
        cv2.fillPoly(black_mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, black_mask)

        lines = cv2.HoughLinesP(
            masked_edges,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=40,
            maxLineGap=10,
        )

        # goal point
        self.find_goal(lines, lane_image)

    ### ----------------- GOAL FINDER  ----------------- ####

    def find_goal(self, lines, image):
        """
        Uses Hough lines inside a left-tilted trapezoid ROI and chooses the
        inner-most left/right lines that intersect the bottom of the image on
        their respective sides. The goal point is placed on a fixed horizon row
        halfway between those two lines.

        :param lines: Numpy Array storing the start and end points of each line
        :image: Numpy Array storing the pixel information of the original image

        :returns None or msg a ROS2 Point message
        """
        if lines is None:
            self.get_logger().info("ERROR: Detected no lines.")
            return

        h, w = image.shape[:2]
        horizon_y = int(np.clip(h * self.horizon_y_ratio, int(h * 0.2), h - 1))
        y_bot = h - 1
        ls, rs = [], []

        for x1, y1, x2, y2 in [s[0] for s in lines]:
            if abs(np.arctan2(y2 - y1, x2 - x1)) >= np.deg2rad(15) and y2 != y1:
                x_bot = x1 + (y_bot - y1) * (x2 - x1) / (y2 - y1)
                (ls if x_bot < (w / 2.0) else rs).append((x1, y1, x2, y2))

        get_x = lambda s: s[0] + (y_bot - s[1]) * (s[2] - s[0]) / (s[3] - s[1])
        curr_pair = (max(ls, key=get_x) if ls else None, min(rs, key=get_x) if rs else None)

        if curr_pair[0] is None and curr_pair[1] is None:
            self.get_logger().info("FUCK NO LANES DETECTED")
            curr_pair = (self.last_left_line, self.last_right_line)
        elif curr_pair[0] is None and curr_pair[1] is not None:
            self.get_logger().info("Inferring the left lane...")
            curr_pair = (self._shift_line(curr_pair[1], -self.avg_lane_width_px, w), curr_pair[1])
        elif curr_pair[0] is not None and curr_pair[1] is None:
            self.get_logger().info("Inferring the right lane...")
            curr_pair = (curr_pair[0], self._shift_line(curr_pair[0], self.avg_lane_width_px, w))

        # BELOW IS TO DEBUG DROPPING A LINE IN CURR_PAIR.
        # if None in curr_pair:
        #     self.get_logger().info("FUCK NO LANES DETECTED")
        #     curr_pair = (self.last_left_line, self.last_right_line)

        msg, dbg = self.goal_from_pair(curr_pair, image, ls, rs, h, w, horizon_y)

        if msg is None:
            fb_pair = (self.last_left_line, self.last_right_line)
            msg, dbg = self.goal_from_pair(fb_pair, image, ls, rs, h, w, horizon_y)
            if msg is None:
                return

        self.goal_pub.publish(msg)
        self.publish_debug_image(dbg)
        return msg

    def goal_from_pair(self, pair, image, ls, rs, h, w, horizon_y):
        """
        Takes in a pair of lines and outputs the goal point on the horizon row.

        :param pair: a tuple of line segments
        :param image: Numpy Array containing raw image pixel data
        :param ls: list of all left line segments
        :param rs: list of all right line segments
        :param h: height of the image
        :param w: width of the image
        :param horizon_y: the fixed y row for the goal point

        :returns Tuple(ROS2 Point message, Image matrix with the goal and line segments outlined)
        """
        left_seg, right_seg = pair
        if left_seg is None or right_seg is None:
            return None, None

        models = []
        for s in (left_seg, right_seg):
            vx, vy, x0, y0 = cv2.fitLine(
                np.array([[s[0], s[1]], [s[2], s[3]]], dtype=np.float32),
                cv2.DIST_L2,
                0,
                0.01,
                0.01,
            ).flatten()
            models.append((np.array([vx, vy]) * np.sign(vy + 1e-6), np.array([x0, y0])))

        (lv, lp), (rv, rp) = models

        def x_at_y(v, p, y):
            if abs(v[1]) < 1e-6:
                return None
            return float(p[0] + (y - p[1]) * v[0] / v[1])

        left_x_h = x_at_y(lv, lp, horizon_y)
        right_x_h = x_at_y(rv, rp, horizon_y)
        if left_x_h is None or right_x_h is None:
            return None, None

        lane_width_px = abs(right_x_h - left_x_h)
        self.lane_width_sum_px += lane_width_px
        self.lane_width_count += 1
        avg_lane_width_px = self.lane_width_sum_px / self.lane_width_count
        print(f"average lane width over node lifetime: {avg_lane_width_px:.1f} px")

        goal_x = float(np.clip(0.5 * (left_x_h + right_x_h), 0, w - 1))
        goal_y = float(horizon_y)

        self.last_left_line = left_seg
        self.last_right_line = right_seg

        dbg = image.copy()
        roi = np.array(self._roi_polygon(image), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(dbg, [roi], True, (0, 255, 255), 2)
        cv2.line(dbg, (w // 2, 0), (w // 2, h - 1), (0, 255, 0), 2)
        for s in ls:
            cv2.line(dbg, s[:2], s[2:], (255, 0, 0), 1)
        for s in rs:
            cv2.line(dbg, s[:2], s[2:], (0, 255, 0), 1)
        cv2.line(dbg, (0, horizon_y), (w - 1, horizon_y), (255, 255, 0), 2)
        for s, c in zip((left_seg, right_seg), [(255, 0, 255), (0, 255, 255)]):
            cv2.line(dbg, s[:2], s[2:], c, 4)
        for (v, p), c in zip(models, [(0, 0, 255), (0, 255, 0)]):
            if abs(v[1]) > 1e-6:
                cv2.line(
                    dbg,
                    (int(p[0] + (h - 1 - p[1]) * v[0] / v[1]), h - 1),
                    (int(p[0] + (horizon_y - p[1]) * v[0] / v[1]), horizon_y),
                    c,
                    3,
                )
        cv2.circle(dbg, (int(goal_x), int(goal_y)), 7, (255, 255, 0), -1)
        cv2.circle(dbg, (int(left_x_h), horizon_y), 5, (255, 0, 255), -1)
        cv2.circle(dbg, (int(right_x_h), horizon_y), 5, (0, 255, 255), -1)

        return Point(x=float(goal_x), y=float(goal_y), z=0.0), dbg

    def _shift_line(self, line, x_offset, width):
        if line is None:
            return None

        x1, y1, x2, y2 = line
        x1_new = int(np.clip(x1 + x_offset, 0, width - 1))
        x2_new = int(np.clip(x2 + x_offset, 0, width - 1))
        return (x1_new, y1, x2_new, y2)

    def _roi_polygon(self, image):
        h, w = image.shape[:2]
        if self.direction == "left":
            return [
                (int(w * 0.1), int(h * 0.8)),
                (int(w * 0.95), int(h * 0.8)),
                (int(w * 0.90), int(h * 0.4)),
                (int(w * 0.2), int(h * 0.4)),
            ]
        return [
            (int(w * 0.05), int(h * 0.8)),
            (int(w * 0.9), int(h * 0.8)),
            (int(w * 0.8), int(h * 0.4)),
            (int(w * 0.1), int(h * 0.4)),
        ]

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
        except Exception:
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
        except Exception:
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


if __name__ == "__main__":
    main()
