import rospy

import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi

from geometry_msgs.msg import Pose, PoseArray

class PlannerGUI:
    """
    Lightweight GUI for initial planning experiments on mushr robot
    """
    def __init__(self, x_window = (-10, 10), y_window = (-10, 10), figsize = (8, 8),  max_history_len = 200):
        self.pose = Pose()
        self.pose_history = []
        self.max_history_len = max_history_len

        self.goal = Pose()
        self.path = PoseArray()

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)

        plt.show(block=False)
        self.cnt = 0
        self.x_window = x_window
        self.y_window = y_window
        
    def handle_pose(self, msg):
        self.pose = msg.pose

        if self.cnt % 5 == 0:
            if len(self.pose_history) >= self.max_history_len:
                self.pose_history.pop(0)
            self.pose_history.append(self.pose)
        self.cnt += 1

    def handle_goal(self, msg):
        self.goal = msg

    def handle_path(self, msg):
        self.path = msg

    def draw_history(self):
        for i, pose in enumerate(self.pose_history):
            self.ax.scatter(pose.position.x, pose.position.y, c='b', s=1.)

    def draw_pose(self):
        px = self.pose.position.x
        py = self.pose.position.y
        self.ax.scatter(px, py, c='r', marker='x', label='Current_pose')
        qw = self.pose.orientation.w
        qx = self.pose.orientation.x
        qy = self.pose.orientation.y
        qz = self.pose.orientation.z
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))

        dx = 0.5*cos(yaw)
        dy = 0.5*sin(yaw)

        self.ax.arrow(px, py, dx, dy, head_width=0.05)

    def draw_goal(self):
        self.ax.scatter(self.goal.position.x, self.goal.position.y, c='g', marker='x', label='Goal')

    def draw_path(self):
        xs = [p.position.x for p in self.path.poses]
        ys = [p.position.y for p in self.path.poses]
        self.ax.plot(xs, ys, marker='.', c='g', label='Planned Path')

    def redraw(self):
        print('redrawing...')
        self.ax.clear()
        self.ax.set_title('Planner GUI')

        self.draw_history()
        self.draw_path()
        self.draw_goal()
        self.draw_pose()

        self.ax.set_xlim(*self.x_window)
        self.ax.set_ylim(*self.y_window)
        self.ax.legend()
        self.fig.canvas.draw()
        plt.pause(1e-2)

