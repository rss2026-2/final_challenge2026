import rclpy
from rclpy.node import Node
import numpy as np
from vs_msgs.msg import ConeLocation
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.time import Time
from std_msgs.msg import String, Bool, Float32
from visualization_msgs.msg import Marker
from viz_utils.visualization_tools import VisualizationTools



class ParkingMeter(Node):
    """
    A node that handles parking meter behavior and updates the
    drive command accordingly.
    """

    def __init__(self):
        super().__init__("parking_meter")

        # -- Declared parameters --
        self.declare_parameter('pm_drive_topic', '/vesc/high_level/input/nav_1')
        self.declare_parameter('pm_point_topic', '/pm_relative_point')
        self.declare_parameter('pm_img_topic', '/yolo/annotated_image')
        self.declare_parameter('odom_topic', '/pf/pose/odom')
        self.declare_parameter('pc_drive_topic', '/pc_drive')
        self.declare_parameter('pc_point_topic', '/pc_relative_point')
        self.declare_parameter('pm_status_topic', '/pm_status')
        self.declare_parameter('parked_topic', '/parked')

        self.pm_drive_topic = self.get_parameter('pm_drive_topic').value
        self.pm_point_topic = self.get_parameter('pm_point_topic').value
        self.pm_img_topic = self.get_parameter('pm_img_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.pc_drive_topic = self.get_parameter('pc_drive_topic').value
        self.pc_point_topic = self.get_parameter('pc_point_topic').value
        self.pm_status_topic = self.get_parameter('pm_status_topic').value
        self.parked_topic = self.get_parameter('parked_topic').get_parameter_value().string_value

        # -- Publishers and subscribers --
        # listen to the annotated image that we would like to save
        self.pm_img_sub = self.create_subscription(Image, self.pm_img_topic, self.pm_img_callback, 1)
        self.pm_point_sub = self.create_subscription(Point, self.pm_point_topic, self.pm_point_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.location_callback, 1)
        # the parking controller node will publish the drive command to this topic, we will listen to see if we are parked
        self.pc_drive_sub = self.create_subscription(AckermannDriveStamped, self.pc_drive_topic, self.pc_drive_callback, 1)

        # for getting the steering to the cone
        self.pc_point_pub = self.create_publisher(ConeLocation, self.pc_point_topic, 10) # triggers parking controller node
        self.pm_status_pub = self.create_publisher(String, self.pm_status_topic, 10) # publishes what state we are in
        self.visualize_meter_pub = self.create_publisher(Marker, '/meter_location', 10)

        self.parked_pub = self.create_publisher(Bool, self.parked_topic, 10) # publishes when parked status changes
        self.times_parked_pub = self.create_publisher(Float32, '/times_parked', 10)

        self.create_timer(1/25, self.pm_drive_timer_callback) # timer callback to call the parking controller...which in turn triggers the rest of the logic

        self.pm_drive_command_pub = self.create_publisher(AckermannDriveStamped, self.pm_drive_topic, 10) # where we publish drive command to

        # -- Initialized variables --
        # Variable for cached parking meters to ignore
        self.parked_locations = None
        self.current_parking_meter_locations = []
        # Variable for if we've already finished parking or not (when set to true, do the save img/wait/leave behavior)
        self.currently_parked = False
        self.timestamp_of_last_park = None
        self.parking_start_distance = 4 # at 4 meters to the parking meter we switch from path following to parking
        self.number_of_times_parked = 0  # updated when we get a zero drive command for the first time
        self.number_of_images_saved = 0
        self.br = CvBridge() # used to save the images
        self.location = None # from localization, used to put parking meter in global frame
        self.goal_x, self.goal_y = None, None

        self.debug_counter = 0

        self.get_logger().info("=== Parking Meter Initialized ===")

    def pm_drive_timer_callback(self):
        """
        Timer callback to update the drive command of the parking meter mux.
        """
        # If we haven't cached a parking meter yet, don't run this timer callback
        if self.goal_x is None or self.goal_y is None or self.goal_vec_world_frame is None:
            return
        distance_to_point = np.linalg.norm([self.goal_x, self.goal_y])
        already_parked_here = self.already_parked_near_here(self.goal_vec_world_frame)

        # If we've already parked near the parking meter, or if we're far from the starting distance to start parking, don't send a drive cmd
        if already_parked_here or distance_to_point > self.parking_start_distance:
            # we don't have a different drive command to send, we should just listen to the follower
            return

        # If not currently parked, park at the meter
        # if not self.currently_parked:
        # send the location to the cone parking
        relative_location = ConeLocation()
        relative_location.x_pos = self.goal_x
        relative_location.y_pos = self.goal_y
        self.pc_point_pub.publish(relative_location) # publishes a drive command to whatever topic we tell it to

        # save the location we currently drove to
        # TODO: make sure to cache these in the world frame so I can average and cache
        if self.goal_vec_world_frame is not None:
            self.current_parking_meter_locations.append(self.goal_vec_world_frame)
            # self.get_logger().info(f'Added to current_parking_meter_locations: {self.goal_vec_world_frame}')
        # will average and save all of the locations for this parking meter later

    def pm_point_callback(self, msg):
        """
        Callback function that runs when receiving parking meter point in world frame.
        Caches parking meter location for use in the timer callback function.
        Done this way so that we don't have to rely on laggy YOLO detections to update drive cmd
        """
        if self.location is None:
            self.get_logger().info('No Localization')
            return

        # Get the meter location from the given relative point
        self.goal_x, self.goal_y = self.extract_meter_location(msg)

        self.goal_vec_world_frame = self.vec_in_world_frame(self.goal_x, self.goal_y)
        VisualizationTools.draw_cylinder(self.goal_vec_world_frame[0], self.goal_vec_world_frame[1], self.visualize_meter_pub, self.get_clock().now().to_msg(), 'map', color=(0.5, 1.0, 1.0))

        # TODO: check cache to ensure we haven't already parked here
        # TODO: check to ensure that we are within parking_start_distance away to switch from following to parking

    def pc_drive_callback(self, drive_msg):
        """
        Intercept the AckermannDriveStamped message from the parking controller node.
        If we are parked, updates with that until we decide it is time to move again and saves the image.
        Otherwise pass through the message to the parking meter drive topic
        """
        velocity, time_stamp = drive_msg.drive.speed, drive_msg.header.stamp
        current_position = (self.location.pose.pose.position.x, self.location.pose.pose.position.y)
        if abs(velocity) < 0.05: # if we have parked
            if self.already_parked_near_here(current_position):
                # self.get_logger().info(f'Already parked near this location, shouldn\'t publish drive command.')
                return
            if not self.currently_parked:
                # self.get_logger().info('pc_drive callback first stop')
                 # TODO: make sure this isn't an issue with reversing/it not stopping for the full 5 seconds
                # the first time we get the stop command ie. when we first stop
                self.currently_parked = True
                self.timestamp_of_last_park = self.get_clock().now() #
                self.number_of_times_parked += 1
                # save the image if it has not already been saved -- this is on it's own callback of the image
                # publish to /parked topic so that path planner can update to next path that should be followed
                parked_msg = Bool()
                parked_msg.data = self.currently_parked
                self.parked_pub.publish(parked_msg)
                self.get_logger().info(f"published {parked_msg=} to /parked topic")
                
                times_parked_msg = Float32()
                times_parked_msg.data = float(self.number_of_times_parked)
                self.times_parked_pub.publish(times_parked_msg)
                self.get_logger().info(f'published {self.number_of_times_parked=} to /times_parked topic')
            else:
                self.get_logger().info('pc_drive callback > first stop')
                # check if it has been 5 seconds yet since this is not the first parking command we get
                time_parked = self.get_parking_duration()
                # self.get_logger().info(f'TIME ELAPSED: {time_parked}, {time_stamp=} {self.timestamp_of_last_park}')
                if time_parked > 5.2:
                    self.get_logger().info(f'Moving on from parking, time elapsed: {time_parked}')
                    # we parked long enough and are ready to start moving again
                    self.currently_parked = False
                    self.timestamp_of_last_park = None
                    self.update_parked_locations()

                    parked_msg = Bool()
                    parked_msg.data = self.currently_parked
                    self.parked_pub.publish(parked_msg)
                    self.get_logger().info(f"published {parked_msg=} to /parked topic")

            self.pm_drive_command_pub.publish(drive_msg)
        else:
            # the drive command is still navigating to the point (just do that)
            # Publish to the drive msg associated with parking meter
            self.pm_drive_command_pub.publish(drive_msg)


    def pm_img_callback(self, img_msg):
        """
        If currently parked, caches the image if it hasn't already
        """
        if not self.currently_parked: # only save image if we are parked
            return
        # save the image
        if self.number_of_images_saved < self.number_of_times_parked:
            # TODO: Check that the bounding box is saved there
            # save the image and name it with the current number
            time = self.get_clock().now().to_msg().sec
            current_frame = self.br.imgmsg_to_cv2(img_msg)
            cv2.imwrite(f'images/image_{self.number_of_images_saved}_{time}.jpg', current_frame)
            self.get_logger().info('Saving image ')
            self.number_of_images_saved += 1

    def location_callback(self, msg):
        """
        Caches the odom msg, the current location of the car in the map.
        """
        self.location = msg

    ##### HELPER FUNCTIONS #####

    def already_parked_near_here(self, goal_point):
        """Checks against all cached points of meter locations to see if we are within 3m to any existing points."""
        # if we have no current lcoation or no points to compare to, we haven't already parked here
        if self.location is None: return False
        points = self.parked_locations
        if points is None:
            return False

        threshold = 3.0 # m, how far do we have to be to consider it a new point, need to account for issues with homography and localization

        # Calculate all distances at once
        distances = np.linalg.norm(points - goal_point, axis=1)

        # Check if the minimum distance is greater than the threshold
        return np.any(distances < threshold) # all within some threshold distance to us

    def extract_meter_location(self, msg):
        """relative pose message, find the location of the parking meter within it."""
        return msg.x, msg.y

    def get_parking_duration(self):
        """Returns how long we have been parked form a cached timestamp of the initial parking until the current timestamp"""
        # initial_park = self.timestamp_of_last_park
        parking_duration = self.get_clock().now() - self.timestamp_of_last_park
        # parking_duration = (Time.from_msg(current_time_stamp) - Time.from_msg(initial_park)).nanoseconds / 1e9
        return parking_duration.nanoseconds / 1e9

    def update_parked_locations(self):
        """After we have finished parking, average all of the locations of the last parked location to save memory.
        Keep the first number_of_times_parked - 1 parked locations the same.
        Average all remaining locations into one [x, y] point.
        Result is a (number_of_times_parked, 2) array.
        """
        self.debug_counter += 1
        self.get_logger().info(f'update_parked_locations has run {self.debug_counter} times')
        if not self.current_parking_meter_locations:
            self.get_logger().info("No meter locations recorded during park. Skipping update")
            # stopped but never detected the meter for some reason
            return

        arr_locations = np.array(self.current_parking_meter_locations)
        averaged_location = np.mean(arr_locations, axis=0, keepdims=True)

        # Combine into final array of shape (number_of_times_parked, 2)
        if self.parked_locations is not None:
            self.parked_locations = np.vstack((self.parked_locations, averaged_location))
        else:
            self.parked_locations = averaged_location
        self.current_parking_meter_locations = []
        # self.get_logger().info(f'Updated the parked locations: {self.parked_locations}')

    def vec_in_world_frame(self,x,y):
        if self.location is None: return

        # robot world position
        px = self.location.pose.pose.position.x
        py = self.location.pose.pose.position.y

        # robot orientation quaternion
        q = self.location.pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        # yaw from quaternion
        yaw = np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)
        )

        # rotate local point into world frame
        world_x = px + x * np.cos(yaw) - y * np.sin(yaw)
        world_y = py + x * np.sin(yaw) + y * np.cos(yaw)

        return world_x, world_y


def main(args=None):
    rclpy.init(args=args)
    planner = ParkingMeter()
    rclpy.spin(planner)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
