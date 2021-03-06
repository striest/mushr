#!/usr/bin/env python

import rospy

from planner.planner import AStarPlanner

from geometry_msgs.msg import Pose, PoseStamped, PoseArray

def main():
    planner = AStarPlanner(discretization=0.2)
    rospy.init_node('planner_node')
    pose_sub = rospy.Subscriber('/car/particle_filter/inferred_pose', PoseStamped, planner.handle_pose)
    goal_sub = rospy.Subscriber('/car/planner/goal', Pose, planner.handle_goal)
    path_pub = rospy.Publisher('/car/planner/path', PoseArray, queue_size=1)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        print('New plan...')
        planner.plan()
        plan_msg = planner.get_path()
        path_pub.publish(plan_msg)
        rate.sleep()

if __name__ == '__main__':
     main()
