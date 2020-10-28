import rospy
import numpy as np
import matplotlib.pyplot as plt
import copy

from skimage.transform import rescale
from numpy import sin, cos, pi

from std_msgs.msg import Float32
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Point, Quaternion
from nav_msgs.msg import MapMetaData, OccupancyGrid

from extended_hybrid_astar.hybrid_astar.astarVREP_rs_cost import *
from extended_hybrid_astar.hybrid_astar_planner import HybridAStarPlanner

class HybridAStarPlannerWithModel(HybridAStarPlanner, object):
    """
    Use hybrid A* with a learned model to get costmaps.
    """
    def __init__(self, network):
        super(HybridAStarPlannerWithModel, self).__init__()
        self.network = network
        self.desired_velocity = 0.

    def vel_msg(self):
        return Float32(data=self.desired_velocity)

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
        heightmap = anglemap = np.copy(self._combined_map) * 4.0

        print(heightmap.shape)
#        import pdb;pdb.set_trace()

        try:
            rx,ry,ryaw,vel,ax,ay = plan_from_VREP(heightmap, start_x, start_y, start_theta, goal_x, goal_y, goal_theta, anglemap, self.network, extended=True, hmap_threshold=0.5)

            vel /= 5.0 #Sam was dumb and put a magic number in his data collection script.

            traj = (ax, ay, np.zeros(len(ax)))
            traj = np.stack(traj, axis=1)
            #if self.traj.size > 0 and np.allclose(self.traj[-1], traj[-1]):
            if False:
                print('IS REPLAN')
                if len(traj) < len(self.traj): #There are some wierd plans. This says taht the shortest feasible path is best
                    self.traj = traj
                    self.desired_velocity = vel
            else: #Always take the new plan if you're planning somewhere new.
                self.traj = traj
                self.desired_velocity = vel
        except:
            x, y = self.pose_2_map(self.start)
            th = self.start[2]
            self.traj = np.array([[x, y, th]]).transpose()
            self.desired_velocity = 0.
            print('No valid traj')
            print(self.traj)
