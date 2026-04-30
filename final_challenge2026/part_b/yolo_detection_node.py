#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dataclasses import dataclass
from rclpy.node import Node
from typing import List
from ultralytics import YOLO

from vs_msgs.msg import Pixel


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # Bounding box coordinates in the original image:
    x1: int
    y1: int
    x2: int
    y2: int


class YoloDetection(Node):
    def __init__(self) -> None:
        super().__init__("yolo_detection")

        ################
        # Added code for final challenge

        # -- Declared parameters --
        self.declare_parameter('traffic_light_topic', '/traffic_light')
        self.declare_parameter('tl_point_px_topic', '/tl_point_px')
        self.declare_parameter('pm_point_px_topic', '/pm_point_px')
        self.declare_parameter('person_point_px_topic', '/person_point_px')
        self.declare_parameter('camera_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('publish_topic', '/yolo/annotated_image')

        self.traffic_light_topic = self.get_parameter('traffic_light_topic').value
        self.tl_point_px_topic = self.get_parameter('tl_point_px_topic').value
        self.pm_point_px_topic = self.get_parameter('pm_point_px_topic').value
        self.person_point_px_topic = self.get_parameter('person_point_px_topic').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.publish_topic = self.get_parameter('publish_topic').value

        # -- Publishers and subscribers --
        self.traffic_light_pub = self.create_publisher(Image, self.traffic_light_topic, 10)
        self.tl_point_px_pub = self.create_publisher(Pixel, self.tl_point_px_topic, 10)
        self.pm_point_px_pub = self.create_publisher(Pixel, self.pm_point_px_topic, 10)
        self.person_point_px_pub = self.create_publisher(Pixel, self.person_point_px_topic, 10)

        ################

        # Declare and get ROS parameters
        self.model_name = (
            self.declare_parameter("model", "yolo11n.pt")
            .get_parameter_value()
            .string_value
        )
        self.conf_threshold = (
            self.declare_parameter("conf_threshold", 0.5)
            .get_parameter_value()
            .double_value
        )
        self.iou_threshold = (
            self.declare_parameter("iou_threshold", 0.7)
            .get_parameter_value()
            .double_value
        )

        self.device = "cpu"
        self.model = YOLO(self.model_name)
        self.model.to(self.device)

        self.class_color_map = self.get_class_color_map()
        self.allowed_cls = [
            i for i, name in self.model.names.items()
            if name in self.class_color_map
        ]

        self.get_logger().info(f"Running {self.model_name} on device {self.device}")
        self.get_logger().info(f"Confidence threshold: {self.conf_threshold}")
        if self.allowed_cls:
            self.get_logger().info(f"You've chosen to keep these class IDs: {self.allowed_cls}")
        else:
            self.get_logger().warn("No allowed classes matched the model's class list.")

        # Create publisher and subscribers
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, self.camera_topic, self.on_image, 1)
        self.pub = self.create_publisher(
            Image, self.publish_topic, 10)
        
        self.get_logger().info(f"=== YOLO Annotator Node Initialized ===")

    def get_class_color_map(self) -> dict[str, tuple[int, int, int]]:
        """
        Return a dictionary mapping a list of COCO class names you want to keep
        to the detection BGR colors in the annotated image. COCO class names include
        "chair", "couch", "tv", "laptop", "dining table", and many more. The list
        of available classes can be found in `self.model.names`.
        """
        return {
            "person": (150 , 0, 150),
            "parking meter": (255, 0, 0),
            "traffic light": (255, 104, 31)
        }

    def on_image(self, msg: Image) -> None:
        # Convert ROS -> OpenCV (BGR)
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Run YOLO inference
        try:
            results = self.model(
                bgr,
                classes=self.allowed_cls,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        if not results:
            return

        # Convert results to Detection List
        dets = self.results_to_detections(results[0])

        # Draw detections on BGR image
        annotated = self.draw_detections(bgr, dets)

        #######
        # Added code for final challenge
        # We want to publish each detection individually to handle different behavior for each detection
        self.publish_detections(bgr, dets, msg.header)
        #######

        # Publish annotated BGR image
        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self.pub.publish(out_msg)

    def results_to_detections(self, result) -> List[Detection]:
        """
        Convert an Ultralytics result into a Detection list.

        YOLOv11 outputs:
          result.boxes.xyxy: (N, 4) tensor
          result.boxes.conf: (N,) tensor
          result.boxes.cls:  (N,) tensor
        """
        detections = []

        if result.boxes is None:
            return detections

        xyxy = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls

        # Convert Torch tensors -> CPU numpy
        xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
        conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
        cls_np = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)
        """
        class Detection:
        class_id: int
        class_name: str
        confidence: float
        # Bounding box coordinates in the original image:
        x1: int
        y1: int
        x2: int
        y2: int
        """
        for xyxy, conf, cls in zip(xyxy_np, conf_np, cls_np):
            if cls not in self.allowed_cls:
                continue
            det = Detection(class_id=cls,
                            class_name=self.model.names[cls],
                            confidence=conf,
                            x1=int(xyxy[0]),
                            x2=int(xyxy[2]),
                            y1=int(xyxy[1]),
                            y2=int(xyxy[3]))
            detections.append(det)
        return detections

    
    # -- Added code to this function for final challenge --
    def draw_detections(
        self,
        bgr_image: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:

        out_image = bgr_image.copy()

        for det in detections:
            boundingbox = ((det.x1,det.y1),(det.x2,det.y2))

            cv2.rectangle(out_image,
                          boundingbox[0],
                          boundingbox[1],
                          self.class_color_map[det.class_name],
                          2)

            cv2.putText(out_image,
                        f"{det.class_name} {det.confidence}", # label
                        (int(det.x1), int(det.y1) - 10), # org (where to place the text)
                        cv2.FONT_HERSHEY_SIMPLEX, # font
                        0.5, # font scale
                        self.class_color_map[det.class_name], # text color
                        2) # thickness
            
        return out_image

    ###############    
    # Added function for final challenge
    def publish_detections(self, bgr_img, detections, header):
        """
        Publish detection to the correct topic for later behavior.
        
        Args:
            bgr_img: The input image of the zed camera
            detections: The object containing detection information
            header: The header of the original img
        """

        for det in detections:
            
            detection_name = det.class_name
            
            # Define the location of the detection using the bottom-midpoint value
            x_mid = (det.x1 + det.x2)/2.0
            pixel_msg = Pixel()
            pixel_msg.u = float(x_mid)
            pixel_msg.v = float(det.y2)

            if detection_name is not None:
                self.get_logger().info(f"YOLO has detected: {detection_name}")
            else:
                self.get_logger().warn(f"YOLO could not detect anything")
                
            publisher = None
            # Use the correct publisher depending on what the detection is
            if detection_name == 'parking meter':
                publisher = self.pm_point_px_pub
            
            elif detection_name == 'person':
                publisher = self.person_point_px_pub
            
            elif detection_name == 'traffic light':
                publisher = self.tl_point_px_pub
                
                # -- Special behavior to publish the cropped traffic light --
                out_img = bgr_img.copy()  

                # Crop the image to the bbox
                assert det.x1 < det.x2 and det.y1 < det.y2, f"why is {det.x1=} < {det.x2=} or {det.y1=} < {det.y2=}?"
                cropped_out_img = out_img[det.y1:det.y2, det.x1:det.x2]

                # Create the msg
                cropped_img_msg = self.bridge.cv2_to_imgmsg(cropped_out_img, encoding="bgr8")
                cropped_img_msg.header = header

                # Publish the msg
                self.traffic_light_pub.publish(cropped_img_msg)

            else:
                self.get_logger().info(f'detection "{detection_name}" does not match any topic we have set up to publish to')
                return
        
            publisher.publish(pixel_msg)
            self.get_logger().info(f'published {detection_name} detection to its respective topic')
    ###############    


def main() -> None:
    rclpy.init()
    node = YoloDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()