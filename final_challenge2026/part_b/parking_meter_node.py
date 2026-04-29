import rclpy
from rclpy.node import Node
import numpy as np
from vs_msgs.msg import ConeLocation
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.time import Time
from std_msgs.msg import String



class ParkingMeter(Node):
    """
    A node that handles parking meter behavior and updates the
    drive command accordingly.
    """

    def __init__(self):
        super().__init__("parking_meter")

        # -- Declared parameters --
        self.declare_parameter('pm_drive_topic', '/vesc/high_level/input/nav_1')
        self.declare_parameter('parking_meter_loc_topic', '/parking_meter_loc')
        self.declare_parameter('parking_meter_img_topic', '/parking_meter_img')
        self.declare_parameter('location_topic', '/pf/pose/odom')

        self.pm_drive_topic = self.get_parameter('pm_drive_topic').value
        self.parking_meter_loc_topic = self.get_parameter('parking_meter_loc_topic').value
        self.parking_meter_img_topic = self.get_parameter('parking_meter_img_topic').value
        self.location_topic = self.get_parameter('location_topic').value


        # -- Publishers and subscribers --
        # listen to the annotated image that we would like to save
        self.parking_meter_img_sub = self.create_subscription(Image, self.parking_meter_img_topic, self.parking_meter_img_callback, 1)
        self.parking_meter_loc_sub = self.create_subscription(Pose, self.parking_meter_loc_topic, self.parking_meter_loc_callback, 1)
        # TODO: ^ check what is being returned here
        self.location_sub = self.create_subscription(Odometry, self.location_topic, self.location_callback, 1)

        # self.pm_drive_pub = self.create_publisher(AckermannDriveStamped, self.pm_drive_topic, 10) # the parking controller node will publish the drive command to this topic, we will listen to see if we are parked
        self.pm_drive_sub = self.create_subscription(AckermannDriveStamped, self.pm_drive_topic, self.pc_drive_callback, 1)
        # TODO: update thelaunch to include parking controller with the proper topics

        # for getting the steering to the cone
        self.meter_location_pub = self.create_publisher(ConeLocation, "/relative_cone", 10) # triggers parking controller node

        self.status_updates_pub = self.create_publisher(String, '/parking_meter_status_updates', 10) # publishes what state we are in
        # -- Initialized variables --

        # Variable for cached parking meters to ignore
        self.parked_locations = None
        self.current_parking_meter_locations = []
        # Variable for if we've already finished parking or not (when set to true, do the save img/wait/leave behavior)
        self.currently_parked = False
        self.timestamp_of_last_park = None
        self.parking_start_distance = 3 # at 3 meters to the parking meter we switch from path following to parking
        self.number_of_times_parked = 0 # updated when we get a zero drive command for the first time
        self.number_of_images_saved = 0
        self.br = CvBridge() # used to save the images
        self.location = None # from localization, used to put parking meter in global frame


        self.get_logger().info("=== Parking Meter Initialized ===")

    def parking_meter_loc_callback(self, msg):
        """
        Handles parking meter behavior when we see one
        """
        if self.location is None:
            self.get_logger().info('No Localization')
            return
        # TODO: perform homography on parking meter
        self.goal_x, self.goal_y = self.extract_meter_location(msg)
        self.goal_vec_world_frame = self.vec_in_world_frame(self.goal_x, self.goal_y)
        distance_to_point = np.linalg.norm([self.goal_x, self.goal_y])


        # TODO: check cache to ensure we haven't already parked here
        # TODO: check to ensure that we are within parking_start_distance away to switch from following to parking
        already_parked_here = self.already_parked_near_here(self.goal_vec_world_frame)
        if already_parked_here or distance_to_point > self.parking_start_distance:
            # we don't have a different drive command to send, we should just listen to the follower
            return

        # TODO: park at the meter
        if not self.currently_parked:
            self.publish_status_update(f'Found new cone location to drive to: {(self.goal_x, self.goal_y)}')
            # send the location to the cone parking
            relative_location = ConeLocation()
            relative_location.x_pos = self.goal_x
            relative_location.y_pos = self.goal_y
            self.meter_location_pub.publish(relative_location) # publishes a drive command to whatever topic we tell it to

            # save the location we currently drove to
            # TODO: make sure to cache these in the world frame so I can average and cache
            if self.goal_vec_world_frame is not None:

                self.current_parking_meter_locations.append(self.goal_vec_world_frame)
            # will average and save all of the locations for this parking meter later

        # TODO: take a picture - done in it's own callback

        # TODO: wait 5sec - evaluated each time we get a drive command

        # TODO: stop sending zero drive command and cache parking meter to ignore it - done in it's own callback




    def pc_drive_callback(self, drive_msg):
        """Listens to the AckermannDriveStamped message from the parking controller node (in visual servoing).
           If we are parked, updates with that until we decide it is time to move again and saves the image. """

        velocity, time_stamp = drive_msg.drive.speed, drive_msg.header.stamp
        if abs(velocity) < 0.05: # if we have parked
            if not self.currently_parked:
                 # TODO: make sure this isn't an issue with reversing/it not stopping for the full 5 seconds
                # the first time we get the stop command ie. when we first stop
                self.currently_parked = True
                self.timestamp_of_last_park = time_stamp
                self.number_of_times_parked += 1
            # save the image if it has not already been saved -- this is on it's own callback of the image
            else:
                # check if it has been 5 seconds yet since this is not the first parking command we get
                time_parked = self.get_parking_duration(time_stamp)
                if time_parked > 5.2:
                    # we parked long enough and are ready to start moving again
                    self.currently_parked = False
                    self.timestamp_of_last_park = None
                    self.update_parked_locations()

        else:
            # the drive command is still navigating to the point (just do that)
            # TODO: maybe i need to publish this to the pm? or can cone parking just do that?
            return


    def parking_meter_img_callback(self, img_msg):
        """ If currently parked, caches the image if it hasn't already"""
        if not self.currently_parked: # only save image if we are parked
            return
        # save the image
        if self.number_of_images_saved < self.number_of_times_parked:
            # TODO: Check that the bounding box is saved there
            # save the image and name it with the current number
            current_frame = self.br.imgmsg_to_cv2(img_msg)
            cv2.imwrite(f'image_{self.number_of_images_saved}.jpg', current_frame)
            self.get_logger().info('Saving image ')
            self.number_of_images_saved += 1

    def location_callback(self, msg):
        self.location = msg

    ##### HELPER FUNCTIONS #####

    def already_parked_near_here(self, goal_point):
        """Checks against all cached points of meter locations to see if we are within 3m to any existing points."""
        # if we have no current lcoation or no points to compare to, we haven't already parked here
        if self.location is None: return False
        points = self.parked_locations
        if points is None:
            return False

        threshold = 3 # m, how far do we have to be to consider it a new point, need to account for issues with homography and localization

        # Calculate all distances at once
        distances = np.linalg.norm(points - goal_point, axis=1)

        # Check if the minimum distance is greater than the threshold
        return np.any(distances < threshold) # all within some threshold distance to us

    def extract_meter_location(self, msg):
        """relative pose message, find the location of the parking meter within it."""
        return msg.position.x, msg.position.y

    def get_parking_duration(self, current_time_stamp):
        """Returns how long we have been parked form a cached timestamp of the initial parking until the current timestamp"""
        initial_park = self.timestamp_of_last_park
        parking_duration = (Time.from_msg(current_time_stamp) - Time.from_msg(initial_park)).nanoseconds / 1e9
        return parking_duration

    def update_parked_locations(self):
        """After we have finished parking, average all of the locations of the last parked location to save memory.
        Keep the first number_of_times_parked - 1 parked locations the same.
        Average all remaining locations into one [x, y] point.
        Result is a (number_of_times_parked, 2) array.
        """
        if self.current_parking_meter_locations is None:
            # stopped but never detected the meter for some reason
            return

        arr_locations = np.array(self.current_parking_meter_locations)
        averaged_location = np.mean(arr_locations, axis=0, keepdims=True)

        # Combine into final array of shape (number_of_times_parked, 2)
        if self.parked_locations is not None:
            self.parked_locations = np.vstack((self.parked_locations, averaged_location))
        else:
            self.parked_locations = averaged_location
        self.current_parking_meter_locations = None
        # self.get_logger().info(f'Updated the parked locations: {self.parked_locations}')
        self.publish_status_update(f'Updated the parked locations: {self.parked_locations}')

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

    def publish_status_update(self, text):
        """Helper to publish updates instead of logging them, we can record and echo them. """
        msg = String()
        msg.data = text
        self.status_updates_pub.publish(msg)
        self.get_logger().info(f'Publishing: "{text}"') # can toggle logging verbosity with this
