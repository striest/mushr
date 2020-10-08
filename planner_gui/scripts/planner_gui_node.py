#!/usr/bin/env python

import rospy

from planner_gui.gui import PlannerGUI

from geometry_msgs.msg import PoseStamped, Pose, PoseArray

def main():
    gui = PlannerGUI(x_window=(-4, 4), y_window=(-4, 4))
    rospy.init_node('planner_gui_node')

    pose_sub = rospy.Subscriber('/car/particle_filter/inferred_pose', PoseStamped, gui.handle_pose)

    goal_sub = rospy.Subscriber('/car/planner/goal', PoseStamped, gui.handle_goal)
    path_sub = rospy.Subscriber('/car/planner/path', PoseArray, gui.handle_path)

    lookahead_pt_sub = rospy.Subscriber('/car/pfc/lookahead_point', PoseStamped, gui.handle_lookahead_point)
    path_pt_sub = rospy.Subscriber('/car/pfc/path_point', PoseStamped, gui.handle_path_point)

    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        print('looping...')
        gui.redraw()
        rate.sleep()

if __name__ == '__main__':
     main()
