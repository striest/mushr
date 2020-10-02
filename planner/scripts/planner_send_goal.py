#!/usr/bin/env python

import rospy
import random

from geometry_msgs.msg import Pose

def main():
    rospy.init_node('planner_send_goal')
    rate = rospy.Rate(0.2) #New goal every 5 seconds
    goal_pub = rospy.Publisher('/planner/goal', Pose, queue_size=1)

    while not rospy.is_shutdown():
        gx = (random.random() - 0.5) * 20
        gy = (random.random() - 0.5) * 20
        goal_pose = Pose()
        goal_pose.position.x = gx
        goal_pose.position.y = gy
        goal_pub.publish(goal_pose)
        rate.sleep()

if __name__ == '__main__':
    main()
