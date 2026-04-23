#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

class LineDetector(Node):
    """
    Uses hough line detector to outline the track lines.
    """

    def __init__(self):
        super().__init__("line_detector")
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('debug_topic', '/track_lines')
        self.declare_parameter('low_threshold', 50)
        self.declare_parameter('high_threshold', 150)

        self.debug_topic = self.get_parameter('debug_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.low_threshold = self.get_parameter('low_threshold').value
        self.high_threshold = self.get_parameter('high_threshold').value

        self.image_sub = self.create_subscription(Image, self.image_topic, self.hough_fallback, 5)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)
        self.debug_offline_pub = self.create_publisher(Image, self.image_topic, 10)
        self.bridge = CvBridge()

        # use this line to debug with static images:
        # self.load_and_publish_image('src/final_challenge2026/racetrack_images/lane_3/image45.png')
        self.get_logger().info(f"Line Detector Node Initialized - Publishing Debug Image to '{self.debug_topic}'")

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
        polygon = np.array([[(0, height), (width, height), (width, int(height * 0.2)), (0, int(height * 0.2))]])
        black_mask = np.zeros_like(edges)
        cv2.fillPoly(black_mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, black_mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=10)

        self.publish_lines(lane_image, lines)

    def image_callback(self, msg):
        """
        Another rendition of hough callback. Slightly more complicated. I tried doing some line merger
        that merges lines if they are close enough together and have the same slope. Performs roughly the
        same as the base line.

        :param msg: ROS2 Image message
        :returns None
        """
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        lane_image = np.copy(image)
        gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        canny = cv2.Canny(blur, self.low_threshold, self.high_threshold)
        self.publish_debug_image(canny)

        cropped_image = self.region_of_interest(canny)
        self.publish_debug_image(cropped_image)

        lines = cv2.HoughLinesP(cropped_image, rho=1, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)

        merged_lines = self.merge_lines(lines)
        self.publish_lines(merged_lines)

    def merge_lines(self, lines):
        """
        Merges similar lines by slope and end point distance.

        :param lines: Numpy array stores the endpoints of the lines
        :returns merged: Numpy array stores the endpoints of the merged lines
        """
        if lines is None:
            return []

        def angle_diff(a, b):
            d = abs(a - b)
            return min(d, np.pi - d)

        def point_to_segment_dist(px, py, x1, y1, x2, y2):
            p = np.array([px, py], dtype=np.float32)
            a = np.array([x1, y1], dtype=np.float32)
            b = np.array([x2, y2], dtype=np.float32)
            ab = b - a
            ab2 = np.dot(ab, ab)
            if ab2 == 0:
                return np.linalg.norm(p - a)
            t = np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0)
            proj = a + t * ab
            return np.linalg.norm(p - proj)

        raw_lines = [tuple(l[0]) for l in lines]
        clusters = []

        for x1, y1, x2, y2 in raw_lines:
            angle = np.arctan2(y2 - y1, x2 - x1) % np.pi
            placed = False

            for c in clusters:
                d1 = point_to_segment_dist(x1, y1, *c["line"])
                d2 = point_to_segment_dist(x2, y2, *c["line"])
                d3 = point_to_segment_dist(*c["line"][:2], x1, y1, x2, y2)
                d4 = point_to_segment_dist(*c["line"][2:], x1, y1, x2, y2)

                if angle_diff(angle, c["angle"]) <= 0.23 and min(d1, d2, d3, d4) <= 20.0:
                    c["points"].append((x1, y1, x2, y2))
                    placed = True
                    break

            if not placed:
                clusters.append({
                    "angle": angle,
                    "line": (x1, y1, x2, y2),
                    "points": [(x1, y1, x2, y2)],
                })

        merged = []
        for c in clusters:
            pts = []
            for x1, y1, x2, y2 in c["points"]:
                pts.extend([[x1, y1], [x2, y2]])

            pts = np.array(pts, dtype=np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            direction = np.array([vx, vy], dtype=np.float32)
            point = np.array([x0, y0], dtype=np.float32)

            t = (pts - point) @ direction
            p1 = point + direction * t.min()
            p2 = point + direction * t.max()
            merged.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        return merged

    def region_of_interest(self, image):
        """
        Filters out unecessary parts of the image. Draws a polygon around the necessary region.

        :param image: Numpy Array stores image pixel data
        :return masked_image: Numpy Array stores filtered image pixel data
        """
        height = image.shape[0]
        width = image.shape[1]
        polygons = np.array([
        [(0, height), (width, height), (width, int(height * 0.5)), (width//2, int(height*0.3)), (0,int(height*0.5))]
        ])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def publish_lines(self, image, lines):
        """
        Helper to publish an image overlaying lines on an image.

        :params image: Numpy Array stores image pixel data
        :params lines: Numpy Array stores endpoints for each line

        :returns None
        """
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
        self.get_logger().info("Publishing a debug image...")
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
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(lane_image, "bgr8")
        except:
            debug_msg = self.bridge.cv2_to_imgmsg(lane_image, "mono8")
        self.debug_offline_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    line_detector = LineDetector()
    rclpy.spin(line_detector)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
