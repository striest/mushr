import rospy
import numpy as np

class Executive:
    """
    Tells other nodes when to do things (for now, just when to resample paths)
    """

    def __init__(self, threshold = 0.1, history_len=100):
        self.pose = None
        self.goal = None
        self.reached_goal = True
        self.has_new_goal = False
        self.threshold = threshold
        self.pose_history = []
        self.history_len = history_len
        self.vel_history = []
        self.path_cnt = 0

    def handle_odom(self, msg):
#        print('handling odom')
        vx = msg.twist.twist.linear.x
        self.vel_history.append(vx)
        if len(self.vel_history) > self.history_len:
            self.vel_history.pop(0)
 
    def handle_pose(self, msg):
#        print('handling pose')
        self.pose = msg.pose

        self.pose_history.append(self.pose)
        if len(self.pose_history) > self.history_len:
            self.pose_history.pop(0)
        self.reached_goal = self.check_goal() or self.is_stalled()

        """
        if self.check_goal():
            print('Reached goal?', self.check_goal())
        """

#        print('Stalled?     ', self.is_stalled())

        """
        if self.has_new_goal:
            print('New Goal?    ', self.has_new_goal)

        if self.should_plan:
            print('NEW PLAN')
        """

    def handle_path(self, msg):
        if self.path_cnt > 3:
            self.has_new_goal = False

    @property
    def should_plan(self):
        return self.has_new_goal or (not self.check_goal() and self.is_stalled())

    def handle_goal(self, msg):
        print('handling goal')
        self.path_cnt = 0
        self.goal = msg.pose
        self.reached_goal = False
        self.has_new_goal = True

    def check_goal(self):
        if self.pose is None or self.goal is None or self.reached_goal:
            return True
        px = self.pose.position.x
        py = self.pose.position.y
        gx = self.goal.position.x
        gy = self.goal.position.y
        self.reached_goal = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5 < self.threshold
        return self.reached_goal

    def is_stalled(self):
        b1 = len(self.pose_history) >= self.history_len
        if not b1:
            return False
        
        for i in self.vel_history:
            if abs(i) > 0.01:
                return False

        return True
