#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
import torch

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dataclasses import dataclass
from rclpy.node import Node
from typing import List
from ultralytics import YOLO


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
        self.declare_parameter('traffic_light_only_topic', '/traffic_light_only')
        self.declare_parameter('parking_meter_topic', '/parking_meter')
        self.declare_parameter('person_topic', '/person')

        self.traffic_light_topic = self.get_parameter('traffic_light_topic').value
        self.traffic_light_only_topic = self.get_parameter('traffic_light_only_topic').value
        self.parking_meter_topic = self.get_parameter('parking_meter_topic').value
        self.person_topic = self.get_parameter('person_topic').value

        # -- Publishers and subscribers --
        self.traffic_light_pub = self.create_publisher(Image, self.traffic_light_topic, 10)
        self.traffic_light_only_pub = self.create_publisher(Image, self.traffic_light_only_topic, 10)
        self.parking_meter_pub = self.create_publisher(Image, self.parking_meter_topic, 10)
        self.person_pub = self.create_publisher(Image, self.person_topic, 10)

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

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
            Image, "/zed/zed_node/rgb/image_rect_color", self.on_image, 10)
        self.pub = self.create_publisher(
            Image, "/yolo/annotated_image", 10)

    ###############    
    # Added function for final challenge
    def publish_detection(self, det, out_image):
        """
        Publish detection to the correct topic for later behavior.
        
        Args:
            detection: The object containing detection information
            out_image: The output image of the zed camera with the bbox around the detection
        """
        detection_name = det.class_name

        # If we see a parking meter, we need: 
            # the full zed image including the bbox around the parking meter to know how far away it is.
        if detection_name == 'parking meter':
            self.parking_meter_pub.publish(out_image)

        # If we see a traffic light, we need:
            # the cropped image of just the traffic light to perform color segmentation for determining red signal or not.
            # the full zed image including the bbox around the parking meter to know how far away it is.
        elif detection_name == 'traffic light':
            self.traffic_light_pub.publish(out_image)
            # Crop the image to the bbox
            assert det.x1 < det.x2 and det.y1 < det.y2, f"why is {det.x1=} < {det.x2=} or {det.y1=} < {det.y2=}?"
            cropped_out_image = out_image[det.y1:det.y2, det.x1:det.x2]
            self.traffic_light_only_pub.publish(cropped_out_image)
        # If we see a person, we need:
            # the full zed image including the bbox around the person to see how far away they are.
        elif detection_name == 'person':
            self.person_pub.publish(out_image)
        else:
            self.get_logger().info(f'detection "{detection_name}" does not match any topic we have set up to publish to')
            return
        
        self.get_logger().info(f'published {detection_name} detection to its respective topic')
    ###############    


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
            #######
            # Added code for final challenge
            # We want to publish each detection individually to handle different behavior for each detection
            self.publish_detection(det, out_image)
            #######
        return out_image


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