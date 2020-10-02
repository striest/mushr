import rospy
import numpy as np
from math import sqrt

from geometry_msgs.msg import Pose, PoseStamped, PoseArray

class AStarPlanner:
    """
    Simple A* planning on a grid in order to verify downstream PFC
    """
    def __init__(self, discretization = 1.0):
        self.discretization = discretization
        self.goal = np.array([0., 0.])
        self.start = np.array([0., 0.])
        self.pose = None
        self.path = []

    def handle_pose(self, msg):
        self.pose = msg.pose
        self.start = np.array([self.pose.position.x, self.pose.position.y])

    def handle_goal(self, msg):
        """
        Let's assume that goals are passed in as poses and we just take the x, y
        """
        goal_x = msg.position.x
        goal_y = msg.position.y
        self.goal = np.array([goal_x, goal_y])
        print('Goal set to {}'.format(self.goal))

    def plan(self):
        """
        Simple grid A* with Euclidean diistance
        """
        curr = AStarNode(self.start, g=0., h=self.h(self.start), prev=-1, discretization=self.discretization)
        openlist = [curr]
        closedlist = []
        settled = set()
        openlist_hash = {} #O(1) access for list elements

        while openlist and not self.is_goal(curr):
            curr = openlist.pop(0)
            neighbors = self.expand(curr)
            for neighbor in neighbors:
                if neighbor in settled:
                    continue
                elif tuple(neighbor.pos) in openlist_hash.keys():
                    cmp_node = openlist_hash[tuple(neighbor.pos)]
                    if cmp_node.g > neighbor.g:
                        cmp_node.g = neighbor.g
                        cmp_node.f = neighbor.f
                        cmp_node.prev = curr
                else:
                    openlist.append(neighbor)
                    openlist_hash[tuple(neighbor.pos)] = neighbor

            openlist.sort(key=lambda x: x.f)

        #Extract the path.
        path = [self.goal, curr.pos]
        while curr.prev != -1:
            curr = curr.prev
            path.append(curr.pos)
        path.append(self.start)

        path.reverse()
        self.path = path
        print('path = \n{}'.format(path))

    def get_path(self):
        msg = PoseArray()
        poses = []
        for pos in self.path:
            pose = Pose()
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            poses.append(pose)
        msg.poses = poses
        return msg

    def expand(self, node):
        pos = node.pos
        out = []
        ds = np.array([[1., 0.],[1., 1.],[0., 1.],[-1., 1.],[-1., 0.],[-1., -1.],[0., -1.],[1., -1.]])
        ds *= self.discretization
        for d in ds:
            new_pos = pos + d
            new_cost = sqrt(d[0]**2 + d[1]**2) * self.discretization
            out.append(AStarNode(new_pos, g=node.g+new_cost, h = self.h(new_pos), prev=node, discretization=self.discretization))
        return out
        

    def is_goal(self, node):
        dist_to_goal = self.h(node)
        return dist_to_goal < self.discretization

    def h(self, pos):
        if isinstance(pos, AStarNode):
            return self.h(pos.pos)
        return sqrt((pos[0] - self.goal[0]) ** 2 + (pos[1] - self.goal[1]) ** 2)

class AStarNode:
    """
    nodes for bookkeeping in A*
    """
    def __init__(self, pos, g, h, prev, discretization):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.prev = prev
        self.discretization = discretization

    def __hash__(self):
        return int(self.pos[0]/self.discretization) + 1000000*int(self.pos[1]/self.discretization)
