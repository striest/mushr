import rospy

import matplotlib.pyplot as plt
import numpy as np

from geometry_msgs.msg import Pose

class PlannerGUI:
    """
    Lightweight GUI for initial planning experiments on mushr robot
    """
    def __init__(self, x_window = (-5, 5), y_window = (-5, 5), figsize = (8, 8),  max_history_len = 200):
        self.pose = Pose()
        self.pose_history = []
        self.max_history_len = max_history_len

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)

        plt.show(block=False)
        self.cnt = 0
        
    def handle_pose(self, msg):
        self.pose = msg.pose
        print 'New pose:', self.pose
        print 'History len:,', len(self.pose_history)

        if self.cnt % 5 == 0:
            if len(self.pose_history) >= self.max_history_len:
                self.pose_history.pop(0)
            self.pose_history.append(self.pose)
        self.cnt += 1

    def redraw(self):
        print('redrawing...')
        self.ax.clear()
        self.ax.set_title('Pose History')
        for i, pose in enumerate(self.pose_history):
            self.ax.scatter(pose.position.x, pose.position.y, c='b')
        self.ax.scatter(self.pose.position.x, self.pose.position.y, c='r', marker='x')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.fig.canvas.draw()
        plt.pause(1e-2)

