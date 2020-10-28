#!/usr/bin/env python

import rospy

from executive.executive import Executive

from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from nav_msgs.msg import Odometry

def main():
    ex = Executive(threshold=0.15)
    rospy.init_node('executive_node')
    #pose_sub = rospy.Subscriber('/car/particle_filter/inferred_pose', PoseStamped, ex.handle_pose)
    pose_sub = rospy.Subscriber('/car/car_pose', PoseStamped, ex.handle_pose)
    goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, ex.handle_goal)
    path_sub = rospy.Subscriber('/car/planner/path', PoseArray, ex.handle_path)
    odom_sub = rospy.Subscriber('car/vesc/odom', Odometry, ex.handle_odom)


    at_goal_pub = rospy.Publisher('/car/executive/reached_goal', Bool, queue_size=1)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        print(ex.should_plan)
        print(ex.vel_history)
        print(len(ex.pose_history))
        at_goal_pub.publish(ex.should_plan)
        ex.path_cnt += 1
        rate.sleep()

if __name__ == '__main__':
     main()
