#!/usr/bin/env python

import rospy
import random

from geometry_msgs.msg import Pose, PoseStamped

class GoalPublisher:
    def __init__(self, r = 1.5):
        self.resample = True
        self.r = r
        self.goal = Pose()

    def sample_goal(self):
        gx = (random.random() - 0.5) * 2*self.r
        gy = (random.random() - 0.5) * 2*self.r
        goal_pose = Pose()
        goal_pose.position.x = gx
        goal_pose.position.y = gy
        self.goal = goal_pose

    def should_resample(self, pose, thresh=0.2):
        gx = self.goal.position.x
        gy = self.goal.position.y
        ex = pose.position.x
        ey = pose.position.y

        dist = ((ex - gx)**2 + (ey - gy)**2) ** 0.5
        return dist < thresh

    def handle_pose(self, msg):
        p = msg.pose
        self.resample = self.should_resample(p)
        if self.resample:
            self.sample_goal()

def main():
    goal_publisher = GoalPublisher(r=2)
    rospy.init_node('planner_send_goal')
    rate = rospy.Rate(2) #New goal every 5 seconds
    goal_pub = rospy.Publisher('/planner/goal', Pose, queue_size=1)

    pose_sub = rospy.Subscriber('/car/car_pose', PoseStamped, goal_publisher.handle_pose)

    while not rospy.is_shutdown():
        goal_pub.publish(goal_publisher.goal)
        rate.sleep()

if __name__ == '__main__':
    main()
