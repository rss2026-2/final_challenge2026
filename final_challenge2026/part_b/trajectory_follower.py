
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive # added AckermannDrive
import rclpy
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from final_challenge2026.part_b.utils import LineTrajectory

# added:
from visualization_msgs.msg import Marker
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R
from viz_utils.visualization_tools import VisualizationTools

class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        # -- Declared parameters --
        self.declare_parameter('odom_topic', "pf/pose/odom") # /pf/pose/odom - the localization pf pose estimate
        self.declare_parameter('drive_topic', "/drive")
        # added in the last pure pursuit
        # self.declare_parameter("car_length", 0.325) # replaced with self.wheelbase_length
        self.declare_parameter("max_steering_angle", 0.34)
        self.declare_parameter("speed", 0.7)
        self.declare_parameter("lookahead", 0.8)
        self.declare_parameter("error_epsilon", 1.0)
        self.declare_parameter("spin_epsilon", 1.0)
        self.declare_parameter("discretization_length", 0.1)
        
        self.declare_parameter("drive_timer_rate", 15.0)
        self.declare_parameter("spin_timer_rate", 1.0)

        # -- Assigning variables --
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        # self.CAR_LENGTH = self.get_parameter('car_length').get_parameter_value().double_value # replaced with self.wheelbase_length
        self.MAX_STEERING_ANGLE = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.SPEED = self.get_parameter('speed').get_parameter_value().double_value
        self.LOOKAHEAD = self.get_parameter('lookahead').get_parameter_value().double_value
        self.EPSILON = self.get_parameter('error_epsilon').get_parameter_value().double_value
        self.SPIN_EPSILON = self.get_parameter('spin_epsilon').get_parameter_value().double_value
        self.DISCRETIZATION_LENGTH = self.get_parameter('discretization_length').get_parameter_value().double_value
        
        drive_timer_rate = self.get_parameter('drive_timer_rate').get_parameter_value().double_value
        spin_timer_rate = self.get_parameter('spin_timer_rate').get_parameter_value().double_value

        # -- Publishers and subscribers --
        self.pose_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1) # publish drive commands here
        self.line_pub = self.create_publisher(Marker, '/drive_line', 10)
        self.target_pub = self.create_publisher(Marker, '/target_point', 10)
        self.lookahead_pub = self.create_publisher(Marker, '/lookahead_line', 10)

        self.drive_timer = self.create_timer(1/drive_timer_rate, self.drive_timer_callback)
        # Timer to update spin drive cmd facilitating spinning the robot in a circle
        self.spin_timer = self.create_timer(1/spin_timer_rate, self.robot_spin_timer_callback)
        

        # -- Other constant vars --
        self.STEERING_ANGLE_THRESH = 1.2 # initially working with it at 0.9 but it was reversing a lot
        self.WHEELBASE_LENGTH = 0.325 # FILL IN # Need to check this number
        self.LOOKAHEAD = 0.8 + 0.2 * self.SPEED

        # -- Initialized vars --
        # Car odometry
        self.x, self.y, self.theta = None, None, None 
        self.initialized_traj = False
        self.path = None
        self.trajectory = LineTrajectory(self, "/followed_trajectory")
        self.last_closest_idx = 0
        
        # Initialization for drive cmd to spin robot in a circle
        self.spin_drive = AckermannDrive()
        self.spin_drive.speed = 0.5
        self.spin_drive.steering_angle = 0.8 * self.MAX_STEERING_ANGLE
        self.spinning = False


        self.get_logger().info("===== Trajectory follower ready =====")

    def pose_callback(self, odometry_msg):
        """
        Takes in a message of type Odometry which is our pose estimate from localization
        and caches the pose.
        """

        self.x = odometry_msg.pose.pose.position.x
        self.y = odometry_msg.pose.pose.position.y

        orientation = odometry_msg.pose.pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        r = R.from_quat(quat)
        self.theta = r.as_euler('zxy', degrees=False)[0]
        # self.get_logger().info(f'New Pose: {self.x}, {self.y}')

    def trajectory_callback(self, msg):
        """
        Callback function that runs when /trajectory/current is populated with a PoseArray.

        Generates a set of points discretizing the path (represented by key points) into points
        that can be followed using our implementation of pure pursuit.
        """
        # self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True

        stamp = self.get_clock().now().to_msg()

        x, y = zip(*self.trajectory.points)
        VisualizationTools.draw_line(list(x), list(y), self.line_pub, stamp, color=(0.0, 1.0, 0.0))

        # added:  discretizing the path
        new_path = [self.trajectory.points[0]] # initialize with the first point
        new_distances = [0]
        for i in range(1, len(self.trajectory.distances)):
            cummulative_segment_length_to_p2 = self.trajectory.distances[i]
            p1, p2 = self.trajectory.points[i-1], self.trajectory.points[i]
            segment_length = cummulative_segment_length_to_p2 - self.trajectory.distances[i-1]
            if segment_length > self.DISCRETIZATION_LENGTH:
                extra_points = int (segment_length // self.DISCRETIZATION_LENGTH) - 1 # one point less than the number of segments
                new_x_pts = np.linspace(p1[0], p2[0], 2 + extra_points)
                new_y_pts = np.linspace(p1[1], p2[1], 2 + extra_points)
                new_segment_distance = segment_length / (extra_points + 1) # divide segment lengthh by the new number of segments i need
                next_distance = new_segment_distance + self.trajectory.distances[i-1] # starting from the last point
                for x_new, y_new in zip(new_x_pts[1:-1], new_y_pts[1:-1]): # skips p1 and p2
                    new_path.append((x_new, y_new))
                    new_distances.append(next_distance)
                    next_distance += new_segment_distance
            new_path.append(p2)
            new_distances.append(cummulative_segment_length_to_p2)
        # new_path = self.trajectory.points
        self.path = np.array(new_path) # list of x, y tuples --> 2d array

        # set the x and y points for the end point (in the map frame)
        self.end_x, self.end_y = new_path[-1]

        # visualize the path
        x, y = zip(*new_path)
        VisualizationTools.draw_points(list(x), list(y), self.line_pub, stamp, frame="/map") # type=Marker.LINE_STRIP for lines

        self.get_logger().info(f'\n***New Path Recieved: {len(new_path)} points ***')

    def drive_timer_callback(self):
        """
        Timer callback to generate new Drive command by pure pursuit using the cached pose.
        """
        # Message initializations
        drive_cmd = AckermannDriveStamped()
        header = Header()
        stamp = self.get_clock().now().to_msg()
        header.stamp = stamp
        header.frame_id = 'base_link'
        drive_cmd.header = header
        
        # check that we only look to move when we have a trajectory and a pose estimate
        if not self.initialized_traj or self.x is None or self.y is None or self.theta is None:
            return

        # Get the lookahead target point (in map frame)
        target_point, traj_vector = self.get_lookahead_point_traj_vector(self.path)
        
        # Visualize the target point
        VisualizationTools.draw_sphere(target_point[0], target_point[1], self.target_pub, stamp, frame="/map", color=(0.5, 0.0, 0.5), scale=(0.3, 0.3, 0.3))

        # Use the target point to update the drive command using our implementation of pure pursuit
        pure_pursuit_drive_cmd = self.update_control(target_point, traj_vector)
        # Update the command msg
        drive_cmd.drive = pure_pursuit_drive_cmd

        # publish the drive command instead of saving it
        self.drive_pub.publish(drive_cmd)
        
    def robot_spin_timer_callback(self):
        """
        Updates drive command to spin the robot in a circle.
        """
        # Invert the drive speed and steering angle to switch between moving forward and backward
        # With the correct timer rate, achieves a spinning motion on the robot
        self.spin_drive.speed = -self.spin_drive.speed
        self.spin_drive.steering_angle = -self.spin_drive.steering_angle            

    def get_lookahead_point_traj_vector(self, path):
        """
        Returns the first point on the path at least LOOKAHEAD distance away.
        """

        # Put robot xy position into a numpy array
        robot_pos = np.array([self.x, self.y])

        # Calculate the squared distance between robot position and each point on the path
        dists = np.sum((path - robot_pos)**2, axis=1)

        # # Get the index of the closest point
        # closest_idx = np.argmin(dists)
        # Only search forward from last known position, never backwards
        search_dists = dists.copy()
        search_dists[:self.last_closest_idx] = np.inf
        closest_idx = np.argmin(search_dists)
        self.last_closest_idx = closest_idx  # ratchet forward only

        # Only consider values further along the path than the closest point
        future_points = path[closest_idx:]
        future_dists = dists[closest_idx:]


        # Make a mask that is all the squared distances greater than the lookahead distance (squared)
        valid_mask = future_dists >= self.LOOKAHEAD**2

        # Apply the mask to our points
        valid_points = future_points[valid_mask]

        # If there is at least one valid point,
        if len(valid_points) > 0:
            # Return the first point in the array. This is the point closest to the lookahead distance
            # Also return the vector between the first and second points, if it exists
            if len(valid_points) > 1:
                line_traj_vector = np.array(list(valid_points[1])) - np.array(list(valid_points[0]))
                return np.array(list(valid_points[0])), line_traj_vector

            return np.array(list(valid_points[0])), None
        # If there are no valid points,
        else:
            # Just return the last point in the path as a fallback
            # self.get_logger().info(f"No valid points > LOOKAHEAD to follow. Following last point")
            return np.array(list(path[-1])), None

    def update_control(self, target_point, traj_vector=None):
        """
        Returns the ackerman drive command
        """
        drive = AckermannDrive()

        # Check to see if we are too close to the goal
        goal_dist = np.sqrt((self.end_x - self.x)**2 + (self.end_y - self.y)**2)
        
        if self.spinning and goal_dist < self.SPIN_EPSILON:
            # Look around for the parking meter
            # Once we see the parking meter, parking controller (on a higher priority) will override this spinning behavior
            # self.spin_drive is updated on a timer by robot_spin_timer_callback
            drive = self.spin_drive
            return drive
                
        # -- Don't worry about reversing for now --
        # in the case that the cone is behind the car, can also be modified for when we don't see the car
        goal_vector = self.world_to_vehicle(target_point)
        if goal_vector[0] < 0:
            self.get_logger().info(f'Reversing x:{goal_vector[0]}')
            drive.speed = -0.5
            # steer toward the cone while reversing
            drive.steering_angle = float(np.clip(
                -np.sign(goal_vector[1]) * self.MAX_STEERING_ANGLE * 0.6,
                -self.MAX_STEERING_ANGLE,
                self.MAX_STEERING_ANGLE
            ))
            return drive

        # if we are in the stopping range of the end of the trajectory,
        if goal_dist < self.EPSILON:
            # Send zero drive command, and then start spinning to look for parking meter
            drive.speed = 0.0
            drive.steering_angle = 0.0
            self.spinning = True
            return drive

        # calculate with the pure pursuit
        robot_pos = np.array([self.x, self.y])
        # goal_vector = target_point - robot_pos
        # goal_vector = self.world_to_vehicle(target_point)
        new_steering_angle = self.compute_feedback_angle(goal_vector)

        # If the turn we have to make is too tight or the cone is cut off, or the cone is just plainly too close, reverse first
        turning_angle_too_tight = abs(new_steering_angle) > self.MAX_STEERING_ANGLE * self.STEERING_ANGLE_THRESH
        if turning_angle_too_tight:
            self.get_logger().info(f'Reversing turining angle:{new_steering_angle }> {self.MAX_STEERING_ANGLE * self.STEERING_ANGLE_THRESH}')
            drive.speed = -0.5
            reverse_angle = -0.5 * new_steering_angle
            drive.steering_angle = float(np.clip(reverse_angle,
                                    -self.MAX_STEERING_ANGLE,
                                    self.MAX_STEERING_ANGLE))

        else: # if it is in front of us reasonable angle, give it that angle
            new_steering_angle = float(np.clip(new_steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE))
            drive.steering_angle = new_steering_angle
            drive.speed = self.get_speed_by_proximity(goal_dist, new_steering_angle, traj_vector)

        return drive

    def compute_feedback_angle(self, goal_vector):
        """
        Calculate steering angle by pure pursuit steering law
        """
        lookahead_dist = np.linalg.norm(goal_vector)

        # pure pursuit steering law
        delta = np.arctan2(
            2 * self.WHEELBASE_LENGTH * goal_vector[1],
            lookahead_dist**2
        )

        # delta = np.clip(delta, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        # not clipping because we want to check potential reverse behavior
        return delta

    def get_speed_by_proximity(self, distance_to_goal, steering_angle, traj_vector):
        """
        Return speed based on how close it is to the goal
        """

        if traj_vector is not None:
            traj_vector_norm = traj_vector / np.linalg.norm(traj_vector)

            cos_theta = np.dot(traj_vector_norm, np.array([1.0,0.0]))

            angle = np.abs(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

            angle_too_wide = np.degrees(abs(angle) - abs(self.theta)) >= 45.0

            # self.get_logger().info(f"traj angle: {np.degrees(angle)} our angle: {np.degrees(self.theta)}")
        else:
            angle_too_wide = False

        if distance_to_goal <= 1.25 or steering_angle >= self.MAX_STEERING_ANGLE / 2 or angle_too_wide:
            return 0.5
        else:
            return self.SPEED

    def world_to_vehicle(self, point):
        dx = point[0] - self.x
        dy = point[1] - self.y

        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        x_car =  cos_theta * dx + sin_theta * dy
        y_car = -sin_theta * dx + cos_theta * dy

        return np.array([x_car, y_car])

def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
