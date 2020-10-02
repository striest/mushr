#!/usr/bin/env python

import rospy

from planner_gui.gui import PlannerGUI

from geometry_msgs.msg import PoseStamped, Pose, PoseArray

def main():
    gui = PlannerGUI()
    rospy.init_node('planner_gui_node')
    pose_sub = rospy.Subscriber('/car/car_pose', PoseStamped, gui.handle_pose)
    goal_sub = rospy.Subscriber('/planner/goal', Pose, gui.handle_goal)
    path_sub = rospy.Subscriber('/planner/path', PoseArray, gui.handle_path)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        print('looping...')
        gui.redraw()
        rate.sleep()

if __name__ == '__main__':
     main()
