import rospy
import numpy as np
import matplotlib.pyplot as plt
import copy

from skimage.transform import rescale
from numpy import sin, cos, pi
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
        self._heightmap_metadata = None
        self._heightmap = None
        self._combined_map_metadata = None
        self._combined_map = None
        self.pose = None
        self.start = None
        self.goal = None
        self.traj = np.zeros([0, 3])
        self.should_plan = False

    def handle_occupancy_grid(self, msg):
        self._map_metadata = msg.info
        self._map = np.array(msg.data)
        self._map = self._map.astype(float) / 100.
        self._map = np.clip(self._map, 0., 1.)
        self._map = np.reshape(self._map, [self._map_metadata.width, self._map_metadata.height])
        if self._combined_map is None: 
            self._combined_map, self._combined_map_metadata = self.prune_map(self._map, self._map_metadata)
#        self._map, self._map_metadata = self.prune_map(self._map, self._map_metadata)

    def handle_heightmap(self, msg):
#        print('handling heightmap')
        self._heightmap_metadata = msg.info
        self._heightmap = np.array(msg.data)
        self._heightmap = self._heightmap.astype(float) / 100.
        self._heightmap = np.clip(self._heightmap, 0., 1.)
        self._heightmap = np.reshape(self._heightmap, [self._heightmap_metadata.width, self._heightmap_metadata.height])
        if self._map is not None and len(self._map.shape) == 2:#Don't itnerrupt map loading
#            print('combining')
            _combined_map, _combined_map_metadata = self.combine_maps(map1=self._map, map2=self._heightmap, map1_metadata=self._map_metadata, map2_metadata=self._heightmap_metadata)
            self._combined_map, self._combined_map_metadata = self.prune_map(_combined_map, _combined_map_metadata)

    def combine_maps(self, map1, map2, map1_metadata, map2_metadata):
        """
        The algo: stitch map2 into map1, using the resolution of map1
        1. Get xmin/max, ymin/ymax of both maps.
        2. Build two empty occupancy grids of the correct shape.
        3. Put each map into these occupancy grids (scale map2)
        4. Combine maps with max.
        """
        metadata_out = MapMetaData()
        m1_r = map1_metadata.resolution
        m2_r = map2_metadata.resolution
        m1_xmin = map1_metadata.origin.position.x
        m1_ymin = map1_metadata.origin.position.y
        m2_xmin = map2_metadata.origin.position.x
        m2_ymin = map2_metadata.origin.position.y
        m1_w = map1_metadata.width
        m1_h = map1_metadata.height
        m2_w = map2_metadata.width
        m2_h = map2_metadata.height
        m1_xmax = m1_xmin + m1_r*m1_w
        m2_xmax = m2_xmin + m2_r*m2_w
        m1_ymax = m1_ymin + m1_r*m1_h
        m2_ymax = m2_ymin + m2_r*m2_h

        map2_2_map1_scaling = m2_r/m1_r

        metadata_out.resolution = map1_metadata.resolution
        metadata_out.origin.position.x = min(m1_xmin, m2_xmin)
        metadata_out.origin.position.y = min(m1_ymin, m2_ymin)
        out_xmax = max(m1_xmax, m2_xmax)
        out_ymax = max(m1_ymax, m2_ymax)
        metadata_out.width = int( (out_xmax - metadata_out.origin.position.x) / metadata_out.resolution ) + 1
        metadata_out.height = int( (out_ymax - metadata_out.origin.position.y) / metadata_out.resolution ) + 1

        map2_scaled = rescale(map2, map2_2_map1_scaling, multichannel=False, anti_aliasing=False)
        map1_acc = np.zeros([metadata_out.width, metadata_out.height])
        map2_acc = np.zeros([metadata_out.width, metadata_out.height])
        m1_start_x = int( (m1_xmin - metadata_out.origin.position.x) / metadata_out.resolution )
        m1_start_y = int( (m1_ymin - metadata_out.origin.position.y) / metadata_out.resolution )
        m2_start_x = int( (m2_xmin - metadata_out.origin.position.x) / metadata_out.resolution )
        m2_start_y = int( (m2_ymin - metadata_out.origin.position.y) / metadata_out.resolution )

        map1_acc[m1_start_x:m1_start_x+map1.shape[0], m1_start_y:m1_start_y+map1.shape[1]] = np.copy(map1)
        map2_acc[m2_start_y:m2_start_y+map2_scaled.shape[0], m2_start_x:m2_start_x+map2_scaled.shape[1]] = np.copy(map2_scaled)
        out = np.maximum(map1_acc, map2_acc)

        return out, metadata_out

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
        if self.start is None or self.goal is None or not self.should_plan or self._combined_map is None:
            return

        print('planning...')
        print(self.start)
        print(self.goal)
        start_y, start_x = self.pose_2_map(self.start)
        goal_y, goal_x = self.pose_2_map(self.goal)
        start_theta = self.start[2]
        goal_theta = self.goal[2]
        heightmap = anglemap = np.copy(self._combined_map) #The A* was designed for heighmap as an image.

        print(heightmap.shape)
#        import pdb;pdb.set_trace()

        try:
            traj = plan_from_VREP(heightmap, start_x, start_y, start_theta, goal_x, goal_y, goal_theta, anglemap, hmap_threshold=0.035)
            traj = np.stack(traj, axis=1)
            if self.traj.size > 0 and np.allclose(self.traj[-1], traj[-1]):
                print('IS REPLAN')
                if len(traj) < len(self.traj): #There are some wierd plans. This says taht the shortest feasible path is best
                    self.traj = traj
            else: #Always take the new plan if you're planning somewhere new.
                self.traj = traj
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
        for row in self.traj:
            x = row[0]
            y = row[1]
            yaw = row[2]
            p = Pose()
            pt = self.map_2_pose([y, x])
            p.position = pt
            p.orientation = self.yaw_2_quat(yaw)
            poses.append(p)

        out.poses = poses
        return out

    def prune_map(self, _map, _map_metadata, margin=50):
        """
        For efficiency, remove dead regions in the map
        """
        #Note that np.argmax always returs the first idx that is max.
        min_y = np.argmax(np.sum(_map, axis=1) > 0)# - margin
        min_x = np.argmax(np.sum(_map, axis=0) > 0)# - margin
        max_y = _map.shape[0] - np.argmax(np.flip(np.sum(_map, axis=1), axis=0) > 0)# + margin
        max_x = _map.shape[1] - np.argmax(np.flip(np.sum(_map, axis=0), axis=0) > 0)# + margin

        _map_out = _map[min_y:max_y, min_x:max_x]
        res = _map_metadata.resolution
        _metadata_out = copy.deepcopy(_map_metadata)
        _metadata_out.origin.position.x += min_x * res
        _metadata_out.origin.position.y += min_y * res
        _metadata_out.width = int(max_x - min_x)
        _metadata_out.height = int(max_y - min_y)
        return _map_out, _metadata_out

    def parse_pose(self, pose):
        return np.array([pose.position.x, pose.position.y, self.quat_2_yaw(pose.orientation)])

    def pose_2_map(self, pose):
        """
        Finds the cell indexes of the occupancy grid closest to the pose.
        """
        o_map_x = self._combined_map_metadata.origin.position.x
        o_map_y = self._combined_map_metadata.origin.position.y
        o_map_yaw = self.quat_2_yaw(self._combined_map_metadata.origin.orientation)

        assert o_map_yaw == 0., 'TODO: Use the coordinate tranform from the map frame'
        #px = self.pose.position.x * cos(o_map_yaw) - self.pose.position.y * -sin(o_map_yaw)
        #py = self.pose.position.x * sin(o_map_yaw) - self.pose.position.y * cos(o_map_yaw)

        px = pose[0] #note: pose is [x, y, theta], not a pose message.
        py = pose[1]
        
        px = px - o_map_x
        py = py - o_map_y

        px = px//self._combined_map_metadata.resolution
        py = py//self._combined_map_metadata.resolution

        return np.array([px, py])

    def map_2_pose(self, coords):
        """
        Construct a position from map coordinates
        """
        y, x = coords
        o_map_x = self._combined_map_metadata.origin.position.x
        o_map_y = self._combined_map_metadata.origin.position.y
        o_map_yaw = self.quat_2_yaw(self._combined_map_metadata.origin.orientation)
        r_map = self._combined_map_metadata.resolution

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
