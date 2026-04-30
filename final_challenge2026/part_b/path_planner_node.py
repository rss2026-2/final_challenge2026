import rclpy

from geometry_msgs.msg import PoseArray, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA



import numpy as np
import math
import heapq
from scipy.spatial.transform import Rotation as R
import cv2

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("grid_search_planner")
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('map_topic', "/map")
        self.declare_parameter('safety_cell_radius', 5)
        self.declare_parameter('max_step_size', 5)
        

        #seperate trajectory publishers for path comparison
        self.declare_parameter('viz_namespace', "/planned_trajectory")
        self.declare_parameter("viz_traj_color", [1.0,1.0,1.0,1.0])
        self.declare_parameter('publish_path', True)
        self.declare_parameter("path_topic", "/trajectory/current")


        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.safety_cell_radius = self.get_parameter('safety_cell_radius').get_parameter_value().integer_value
        self.max_step_size = self.get_parameter('max_step_size').get_parameter_value().integer_value

        self.viz_namespace = self.get_parameter("viz_namespace").get_parameter_value().string_value
        self.viz_traj_color = self.get_parameter("viz_traj_color").get_parameter_value().double_array_value
        self.publish_path = self.get_parameter('publish_path').get_parameter_value().bool_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
    

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 1)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_cb, 10)

        if self.publish_path:
            self.traj_pub = self.create_publisher(PoseArray, self.path_topic, 10)


        self.search_pub = self.create_publisher(MarkerArray,"/search_alg",10)

        self.trajectory = LineTrajectory(node=self, viz_namespace=self.viz_namespace)
        self.map = None
        self.dist_map = None
        self.pose = None

        self.get_logger().info("Awaiting Map")

    def map_cb(self, map_msg):
        """
        Receives an occupancy map and formats it to be used in later functions

        Args:
            map_msg (Occupancy Grid): A map whose data is represented as integers from
            0-100 representing the probability of being occupied. -1 is unknown.
        """
        self.get_logger().info("Successfully Received Map Information")

        map_transform = np.eye(4)
        translation = [
            map_msg.info.origin.position.x,
            map_msg.info.origin.position.y,
            map_msg.info.origin.position.z
        ]
        rotation = R.from_quat([
            map_msg.info.origin.orientation.x,
            map_msg.info.origin.orientation.y,
            map_msg.info.origin.orientation.z,
            map_msg.info.origin.orientation.w,
        ])

        map_transform[:3, 3] = translation
        map_transform[:3,:3] = rotation.as_matrix()

        transform_inverse = np.eye(4)
        transform_inverse[:3, :3] = map_transform[:3,:3].T
        transform_inverse[:3, 3] = -map_transform[:3,:3].T @ map_transform[:3, 3]

        occupancy_grid = np.array(map_msg.data).reshape(map_msg.info.height, map_msg.info.width)

        self.map_occupancy_expansion(occupancy_grid, self.safety_cell_radius)
   
        binary_map = (occupancy_grid == 0).astype(np.uint8)
        self.dist_map = np.array(cv2.distanceTransform(binary_map, cv2.DIST_L2, 5))

        self.map = {
            "res": map_msg.info.resolution,
            "transform": map_transform,
            "transform_inv": transform_inverse,
            "array" : occupancy_grid,
            "width" : map_msg.info.width,
            "height": map_msg.info.height
        }

        self.get_logger().info("Ready to start planning!")

    def map_occupancy_expansion(self, grid, radius, prob_thresh = 0.3):
        """
        Converts the occupancy grid to a binary map based on a probability threshold,
        then expands the boundaries by a certain radius.
        
        Args:
            grid (nd array): A numpy 2D array representing the occupancy grid whose
            values are integers from 0-100 representing the probability of being occupied. 
            -1 is unknown.
            radius (int): The number of cells with which to expand boundaries.
            prob_thresh (float): Used to convert to binary map.

        """
        # find boundaries of map
        threshold = np.zeros(shape = grid.shape, dtype = np.uint8)
        threshold[grid >= int(prob_thresh * 100)] = 255

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            for point in contour:
                # point is [[x, y]], so we need to extract the coordinates
                x, y = point[0]
                cv2.circle(threshold, (x, y), radius, color=255, thickness=-1)

        grid[grid == -1] = 1
        grid[threshold > 0] = 1

    def pose_cb(self, pose_msg):
        """
        Saves the current pose of the racecar in the map.

        Args:
            pose_msg (Odometry): The pose of the racecar received as a message
        """
        self.pose = {
            "position": [
                pose_msg.pose.pose.position.x,
                pose_msg.pose.pose.position.y,
                pose_msg.pose.pose.position.z
            ],
            "orientation": [
                pose_msg.pose.pose.orientation.x,
                pose_msg.pose.pose.orientation.y,
                pose_msg.pose.pose.orientation.z,
                pose_msg.pose.pose.orientation.w,
            ]
        }

    def goal_cb(self, goal_msg):
        """
        Given a goal pose, finds a path to that position from the current pose,
        publishes it as a trajectory, and visualizes in RViz. Returns None if a path
        is not found.

        Args:
            goal_msg (PoseStamped): The pose of the goal position
        """
        if self.map is None:
            self.get_logger().info("Map Information Not Received")
            return

        goal_pose = goal_msg.pose
        start_point = (self.pose["position"][0], self.pose["position"][1])
        end_point = (goal_pose.position.x, goal_pose.position.y)

        path_found = self.plan_path(
            start_point = start_point,
            end_point = end_point,
            visualize = False
        )

        if not path_found:
            self.get_logger().info("No path found to end point")
            return

        self.get_logger().info("Path Generated!")

        if self.publish_path:
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.get_logger().info("Path Published!")

        self.trajectory.publish_viz(traj_color = self.viz_traj_color)
        self.get_logger().info("Path Visualized!")
    
    def find_valid_neighbors(self, curr_cell, step_size):
        """
        Finds the valid unseen neighbors of the given current cell. The step size is bounded 
        by the given max_step_size and the distance to the end_cell (prevents overshooting).

        Args:
            curr_cell (tuple): The (v,u) location of the current cell in the occupancy grid
            end_cell (tuple): The (v,u) location of the end cell in the occupancy grid
            seen (set): All the cells that have been visited before

        """
        map_w, map_h = self.map["width"], self.map["height"]
        neighbors = []
        curr_u, curr_v = curr_cell

        for du,dv in [(1,1), (-1,-1), (1,-1), (-1,1), (-1,0), (1,0), (0,-1), (0,1)]:
            n_cell = (curr_u + (du * step_size), curr_v + (dv*step_size))
            if 0 <= n_cell[0] < map_w and 0 <= n_cell[1] < map_h: #if the cell is within bounds
                if self.map["array"][n_cell[1]][n_cell[0]] == 0:
                    neighbors.append(n_cell)
        
        return neighbors
    
    def plan_path(self, start_point, end_point, visualize = False):
        """
        Given a start and end cell, uses A* search to find a path between these
        locations whilst minimizing path distance and distance from the goal
        and maximizing average and minimum path clearance from obstacles.

        Args:
            start_point: The starting location in the real map
            end_cell: The end location in the real map
            visualize: Whether to visualize the A* algorithm as it runs (makes it much
            slower but is useful for debugging)

        """
        cells = self.real_to_grid_frame(np.array([start_point, end_point]))
        start_cell, end_cell = tuple(cells[0]), tuple(cells[1])
        found_path = None

        queue = [] # le heap
        seen = set()
        edges = [] # collect all edges for visualization

        start_item = (
            math.dist(start_cell,end_cell), #t otal cost
            0, # path_cost
            0, # avg clearance
            float("inf"), # min clearance
            (start_cell,) # path
        )
        heapq.heappush(queue, start_item)
        
        if visualize:
            self.clear_points()

        while queue:
            _, curr_path_cost, avg_clearance, min_clearance, curr_path = heapq.heappop(queue)
            curr_cell = curr_path[-1]

            if curr_cell in seen:
                continue

            seen.add(curr_cell)
            if visualize and len(curr_path) >= 2:
                prev_cell = curr_path[-2]
                prev_point, curr_point = self.grid_to_real_frame([prev_cell, curr_cell])
                
                edges.append((prev_point, curr_point))
                # Publish incrementally to show tree expansion over time
                self.publish_edges(edges)

            if curr_cell == end_cell:
                found_path = curr_path
                break

            curr_dist_from_end = math.dist(curr_cell, end_cell)
            step_size = max( min(int(curr_dist_from_end), self.max_step_size), 1)

            for neighbor in self.find_valid_neighbors(curr_cell, step_size):
                new_path = curr_path + (neighbor,)

                new_path_cost = curr_path_cost + math.dist(curr_cell, neighbor)

                curr_clearance = self.dist_map[neighbor[1],neighbor[0]]

                new_avg_clearance = avg_clearance + (self.dist_map[neighbor[1],neighbor[0]] - avg_clearance) / len(new_path)
                new_min_clearance = min(min_clearance, curr_clearance)
                new_total_cost = new_path_cost + math.dist(neighbor ,end_cell) - new_avg_clearance - new_min_clearance

                heapq.heappush(queue, (new_total_cost, new_path_cost, new_avg_clearance, new_min_clearance, new_path))
        
        if found_path is None:
            return False
        
        shortened_path = self.shorten_cell_path(found_path)
        real_path = self.grid_to_real_frame(shortened_path)

        self.trajectory.clear()
        self.trajectory.addPoints(real_path)
        return True
    
    def publish_edges(self, edges):
        """
        Publish all accumulated edges as a single connected marker.

        Args:
            edges (nd array of tuples): An array of tuples wherein each tuple has two
            cells representing an edge.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = self.viz_namespace
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Add all edge points to the marker
        for start, end in edges:
            start_pt = Point()
            start_pt.x = start[0]
            start_pt.y = start[1]
            start_pt.z = 0.0

            end_pt = Point()
            end_pt.x = end[0]
            end_pt.y = end[1]
            end_pt.z = 0.0

            marker.points.extend([start_pt, end_pt])

        marker.pose.orientation.w = 1.0
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        marker.scale.x = 0.02  # Line width in meters
        marker.scale.z = 0.0
        
        marker_arr = MarkerArray()
        marker_arr.markers.append(marker)
        self.search_pub.publish(marker_arr)
    
    def clear_points(self):
        """
        Clears the drawing of the search algorithm
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "search_alg"
        marker.action = Marker.DELETEALL

        marker_arr = MarkerArray()
        marker_arr.markers.append(marker)
        self.search_pub.publish(marker_arr)

    def shorten_cell_path(self, cell_path):
        """
        Given a path of cells, removes extraneous cells that do not indicate a change
        of direction

        Args:
            cell_path (nd array): An array of tuples representing cells of a valid
            path from the first cell to the last
        """
        curr_heading = None
        new_cell_path = []

        for i in range(1, len(cell_path)):
            prev_cell, new_cell = cell_path[i-1], cell_path[i]

            new_heading = (new_cell[0] - prev_cell[0], new_cell[1] - prev_cell[1])
            if new_heading != curr_heading:
                new_cell_path.append(prev_cell)
                curr_heading = new_heading
        
        if new_cell_path[-1] != cell_path[-1]:
            new_cell_path.append(cell_path[-1])
        return new_cell_path

    def grid_to_real_frame(self, cells):
        """
        Converts a list of cells to their pixel locations in the real map

        Args:
            cells (nd array): A list of (v, u) tuples representing the cell locations in
            the occupancy grid
        """
        cells = np.array([[cx * self.map["res"], cy * self.map["res"],0,1] for cx, cy in cells])
        points = (self.map["transform"] @ cells.T).T

        return np.array([[px,py] for px,py,_,_ in points])
    
    def real_to_grid_frame(self, points):
        """
        Converts a list of pixels to their cell locations in the occupancy grid

        Args:
            points (nd array): A list of (x, y) coords representing the pixel locations in the
            real map
        """
        points = np.array([[px,py,0,1] for px, py in points])
        cells = ((self.map["transform_inv"] @ points.T).T)
        return np.array([[cx / self.map["res"], cy / self.map["res"]] for cx , cy, _, _ in cells], dtype = int)


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
