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

        self.image_sub = self.create_subscription(Image, self.image_topic, self.hough_callback, 5)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)
        self.goal_pub = self.create_publisher(Point, self.goal_topic, 10)
        self.bridge = CvBridge()

        # cache lanes in case of frames dropping
        self.last_left_line = None
        self.last_right_line = None
        self.last_lane_width = None
        self.goal_y_ref = None
        self.goal_y_tolerance = 25.0

        # use this line to debug with static images:
        # self.load_and_publish_image('src/final_challenge2026/racetrack_images/lane_3/image45.png')

        self.get_logger().info(f"Line Detector Node Initialized - Publishing Debug Image to '{self.debug_topic}'")

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
        Inputs lines found from hough transform. Outputs a goal point for the car to follow.

        1) Filter out lines that are too flat (these are noise)
        2) Detect the left and right lane segments. Extend each line to the bottom of the
        image. Lines that intersect the image to the left of the midpoint are left while those
        that intersect to the right are right.
        3) Choose the left segment that is furthest to the right and the right segment that is
        furthest to the right (smallest width lane formed by the lines).
        2) Fit LOBF that extends past the lines. Find the intersection of LOBF.
        3) Bisect the area between the two lines.
        4) Follow the bisection for distance of self.goal_y_offset.
        5) Draw the goal point.

        :param lines: Numpy Array storing the start and end points of each line
        :image: Numpy Array storing the pixel information of the original image

        :returns None or msg a ROS2 Point message
        """
        if lines is None:
            self.get_logger().info("ERROR: Detected no lines.")
            return

        h, w = image.shape[:2]
        y_bot = h - 1
        ls, rs = [], []

        for x1, y1, x2, y2 in [s[0] for s in lines]:
            if abs(np.arctan2(y2 - y1, x2 - x1)) >= np.deg2rad(15) and y2 != y1:
                x_bot = x1 + (y_bot - y1) * (x2 - x1) / (y2 - y1)
                (ls if x_bot < (w / 2.0) else rs).append((x1, y1, x2, y2))

        get_x = lambda s: s[0] + (y_bot - s[1]) * (s[2] - s[0]) / (s[3] - s[1])
        curr_pair = (max(ls, key=get_x) if ls else None, min(rs, key=get_x) if rs else None)

        msg, dbg = self.goal_from_pair(curr_pair, image, ls, rs, h, w)

        if msg is None:
            self.get_logger().info("WARNING: Using fallback lanes.")
            fb_pair = (getattr(self, 'last_left_line', None), getattr(self, 'last_right_line', None))
            msg, dbg = self.goal_from_pair(fb_pair, image, ls, rs, h, w)
            if msg is None:
                return

        self.goal_pub.publish(msg)
        self.publish_debug_image(dbg)
        return msg

    def infer_lines(self, l, side):
        """
        Infer the missing lane line by translating the detected line sideways.

        The translation distance is learned from the most recent valid lane pair.
        """
        if l is None:
            return None, None

        lane_width = getattr(self, "last_lane_width", None)
        if lane_width is None or not np.isfinite(lane_width) or lane_width <= 0:
            return None, None

        shift = lane_width if side == "right" else -lane_width
        inferred = tuple(int(round(v)) for v in (l[0] + shift, l[1], l[2] + shift, l[3]))
        if side == "left":
            return inferred, l
        if side == "right":
            return l, inferred
        return None, None

    def goal_from_pair(self, pair, image, ls, rs, h, w):
        """
        Takes in a pair of lines and outputs the goal point. Does steps 2-4 detailed in
        self.find_goal.

        :param pair: a tuple of line segments
        :param image: Numpy Array containing raw image pixel data
        :param ls: list of all left line segments
        :param rs: list of all right line segments
        :param h: height of the image
        :param w: width of the image

        :returns Tuple(ROS2 Point message, Image matrix with the goal and line segments outlined)
        """

        if pair == (None, None):
            return None, None
        if pair[0] is None:
            pair = self.infer_lines(pair[1], "left")
            if pair == (None, None):
                return None, None
        elif pair[1] is None:
            pair = self.infer_lines(pair[0], "right")
            if pair == (None, None):
                return None, None

        models = []
        for s in pair:
            vx, vy, x0, y0 = cv2.fitLine(np.array([[s[0], s[1]], [s[2], s[3]]], dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            models.append((np.array([vx, vy]) * np.sign(vy + 1e-6), np.array([x0, y0])))

        (lv, lp), (rv, rp) = models

        A = np.array([[-lv[1], lv[0]], [-rv[1], rv[0]]])
        b = np.array([-lv[1]*lp[0] + lv[0]*lp[1], -rv[1]*rp[0] + rv[0]*rp[1]])
        if abs(np.linalg.det(A)) < 1e-6: return None, None
        inter = np.linalg.solve(A, b)

        bis = (lv / np.linalg.norm(lv)) + (rv / np.linalg.norm(rv))
        if np.linalg.norm(bis) < 1e-6: return None, None
        bis = (bis / np.linalg.norm(bis)) * np.sign(bis[1] + 1e-6)

        gy = min(inter[1] + self.goal_y_offset, h - 1)
        gx = np.clip(inter[0] + (gy - inter[1]) * bis[0] / bis[1], 0, w - 1)

        if getattr(self, 'goal_y_ref', None) is not None and abs(gy - self.goal_y_ref) > getattr(self, 'goal_y_tolerance', float('inf')):
            return None, None
        self.goal_y_ref = gy
        self.last_left_line, self.last_right_line = pair
        if abs(models[0][0][1]) > 1e-6 and abs(models[1][0][1]) > 1e-6:
            left_x_bot = models[0][1][0] + (h - 1 - models[0][1][1]) * models[0][0][0] / models[0][0][1]
            right_x_bot = models[1][1][0] + (h - 1 - models[1][1][1]) * models[1][0][0] / models[1][0][1]
            self.last_lane_width = abs(float(right_x_bot - left_x_bot))

        dbg = image.copy()
        for s in ls: cv2.line(dbg, s[:2], s[2:], (255, 0, 0), 1)
        for s in rs: cv2.line(dbg, s[:2], s[2:], (0, 255, 0), 1)
        for s, c in zip(pair, [(255, 0, 255), (0, 255, 255)]): cv2.line(dbg, s[:2], s[2:], c, 4)
        for (v, p), c in zip(models, [(0, 0, 255), (0, 255, 0)]):
            if abs(v[1]) > 1e-6: cv2.line(dbg, (int(p[0] + (h-1-p[1])*v[0]/v[1]), h-1), (int(p[0] + (h*0.45-p[1])*v[0]/v[1]), int(h*0.45)), c, 3)
        ix, iy, gx_int, gy_int = int(inter[0]), int(inter[1]), int(gx), int(gy)
        cv2.circle(dbg, (ix, iy), 7, (0, 255, 255), -1)
        cv2.circle(dbg, (gx_int, gy_int), 7, (255, 255, 0), -1)
        cv2.line(dbg, (ix, iy), (gx_int, gy_int), (255, 255, 0), 2)

        return Point(x=float(gx), y=float(gy), z=0.0), dbg

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
