#!/usr/bin/env python

import rospy

from executive.executive import Executive

from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseStamped, PoseArray

def main():
    ex = Executive()
    rospy.init_node('executive_node')
    #pose_sub = rospy.Subscriber('/car/particle_filter/inferred_pose', PoseStamped, ex.handle_pose)
    pose_sub = rospy.Subscriber('/car/car_pose', PoseStamped, ex.handle_pose)
    goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, ex.handle_goal)
    path_sub = rospy.Subscriber('/car/planner/path', PoseArray, ex.handle_path)


    at_goal_pub = rospy.Publisher('/car/executive/reached_goal', Bool, queue_size=1)
    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        at_goal_pub.publish(ex.should_plan)
        rate.sleep()

if __name__ == '__main__':
     main()
