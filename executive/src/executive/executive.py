import rospy
import numpy as np

class Executive:
    """
    Tells other nodes when to do things (for now, just when to resample paths)
    """

    def __init__(self, threshold = 0.2, history_len=100):
        self.pose = None
        self.goal = None
        self.reached_goal = True
        self.has_new_goal = False
        self.threshold = threshold
        self.pose_history = []
        self.history_len = history_len

    def handle_pose(self, msg):
        self.pose = msg.pose

        self.pose_history.append(self.pose)
        if len(self.pose_history) > self.history_len:
            self.pose_history.pop(0)
        self.reached_goal = self.check_goal() or self.is_stalled()

        if self.check_goal():
            print('Reached goal?', self.check_goal())

        if self.is_stalled():
            print('Stalled?     ', self.is_stalled())

        if self.has_new_goal:
            print('New Goal?    ', self.has_new_goal)

        if self.should_plan:
            print('NEW PLAN')

    def handle_path(self, msg):
        self.has_new_goal = False

    @property
    def should_plan(self):
        return (self.reached_goal and self.has_new_goal) or ((not self.check_goal) and self.is_stalled())

    def handle_goal(self, msg):
        print('handling goal')
        self.goal = msg.pose
        self.reached_goal = self.check_goal() or self.is_stalled()
        self.has_new_goal = True

    def check_goal(self):
        if self.pose is None or self.goal is None:
            return True
        px = self.pose.position.x
        py = self.pose.position.y
        gx = self.goal.position.x
        gy = self.goal.position.y
        return ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5 < self.threshold

    def is_stalled(self):
        b1 = len(self.pose_history) >= self.history_len
        if not b1:
            return False
        
        for i in range(1, len(self.pose_history)):
            b2 = abs(self.pose_history[0].position.x - self.pose_history[-1].position.x) < 1e-3
            b3 = abs(self.pose_history[0].position.y - self.pose_history[-1].position.y) < 1e-3
            if not(b2 and b3):
                return False

        return True
