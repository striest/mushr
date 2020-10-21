import rospy
import numpy as np
from numpy import sin, cos, pi

import matplotlib.pyplot as plt

from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Point, Quaternion
from nav_msgs.msg import MapMetaData, OccupancyGrid

from extended_hybrid_astar.hybrid_astar.astarVREP import *

class HybridAStarPlanner:
    """
    Wrapper class around the hybrid a* code
    Essentially, gathers pose, map and goal and passes it onto the planner
    """
    def __init__(self):
        self._map_metadata = None
        self._map = None
        self.pose = None
        self.start = None
        self.goal = None
        self.traj = ([], [], [])
        self.should_plan = False

    def handle_occupancy_grid(self, msg):
        self._map_metadata = msg.info
        self._map = np.array(msg.data)
        self._map = self._map.astype(float) / 100.
        self._map = np.clip(self._map, 0., 1.)
        self._map = np.reshape(self._map, [self._map_metadata.width, self._map_metadata.height])
        self.prune_map()

    def handle_pose(self, msg):
#        print('handling pose')
        self.pose = msg.pose
        self.start = self.parse_pose(self.pose)

    def handle_goal(self, msg):
#        print('handling goal')
        self.goal = self.parse_pose(msg.pose)

    def handle_reached_goal(self, msg):
        self.should_plan = msg.data

    def plan(self):
#        print('planning...')
        if self.start is None or self.goal is None or not self.should_plan:
            return

        print(self.start)
        print(self.goal)
        start_y, start_x = self.pose_2_map(self.start)
        goal_y, goal_x = self.pose_2_map(self.goal)
        start_theta = self.start[2]
        goal_theta = self.goal[2]
        heightmap = anglemap = np.copy(self._map) #The A* was designed for heighmap as an image.

        print(heightmap.shape)
        import pdb;pdb.set_trace()

        try:
            self.traj = plan_from_VREP(heightmap, start_x, start_y, start_theta, goal_x, goal_y, goal_theta, anglemap)
        except:
            x, y = self.pose_2_map(self.start)
            th = self.start[2]
            self.traj = np.array([[x, y, th]]).transpose()
            print('No valid traj')
            print(self.traj)

    def plan_msg(self):
        out = PoseArray()
        out.header.frame_id = "/map"

        poses = []
        for x, y, yaw in zip(self.traj[0], self.traj[1], self.traj[2]):
            p = Pose()
            pt = self.map_2_pose([y, x])
            p.position = pt
            p.orientation = self.yaw_2_quat(yaw)
            poses.append(p)

        out.poses = poses
        return out

    def prune_map(self, margin=50):
        """
        For efficiency, remove dead regions in the map
        """
        #Note that np.argmax always returs the first idx that is max.
        min_y = np.argmax(np.sum(self._map, axis=1) > 0)# - margin
        min_x = np.argmax(np.sum(self._map, axis=0) > 0)# - margin
        max_y = self._map.shape[0] - np.argmax(np.flip(np.sum(self._map, axis=1), axis=0) > 0)# + margin
        max_x = self._map.shape[1] - np.argmax(np.flip(np.sum(self._map, axis=0), axis=0) > 0)# + margin

        self._map = self._map[min_y:max_y, min_x:max_x]
        res = self._map_metadata.resolution
        self._map_metadata.origin.position.x += min_x * res
        self._map_metadata.origin.position.y += min_y * res

    def parse_pose(self, pose):
        return np.array([pose.position.x, pose.position.y, self.quat_2_yaw(pose.orientation)])

    def pose_2_map(self, pose):
        """
        Finds the cell indexes of the occupancy grid closest to the pose.
        """
        o_map_x = self._map_metadata.origin.position.x
        o_map_y = self._map_metadata.origin.position.y
        o_map_yaw = self.quat_2_yaw(self._map_metadata.origin.orientation)

        assert o_map_yaw == 0., 'TODO: Use the coordinate tranform from the map frame'
        #px = self.pose.position.x * cos(o_map_yaw) - self.pose.position.y * -sin(o_map_yaw)
        #py = self.pose.position.x * sin(o_map_yaw) - self.pose.position.y * cos(o_map_yaw)

        px = pose[0] #note: pose is [x, y, theta], not a pose message.
        py = pose[1]
        
        px = px - o_map_x
        py = py - o_map_y

        px = px//self._map_metadata.resolution
        py = py//self._map_metadata.resolution

        return np.array([px, py])

    def map_2_pose(self, coords):
        """
        Construct a position from map coordinates
        """
        y, x = coords
        o_map_x = self._map_metadata.origin.position.x
        o_map_y = self._map_metadata.origin.position.y
        o_map_yaw = self.quat_2_yaw(self._map_metadata.origin.orientation)
        r_map = self._map_metadata.resolution

        assert o_map_yaw == 0., 'TODO: Use the coordinate tranform from the map frame'

        px = o_map_x + r_map * x
        py = o_map_y + r_map * y
        return Point(x=px, y=py)

    def quat_2_yaw(self, orientation):
        #Gets yaw from pose
        qw = orientation.w
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
        return yaw   

    def yaw_2_quat(self, yaw):
        q = Quaternion()
        q.w = cos(yaw/2)
        q.x = 0.
        q.y = 0.
        q.z = sin(yaw/2)
        return q
