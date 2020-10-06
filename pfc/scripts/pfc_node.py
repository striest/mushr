#!/usr/bin/env python

import rospy

from pfc.pure_pursuit import PurePursuitController

from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from ackermann_msgs.msg import AckermannDriveStamped

def main():
    controller = PurePursuitController(lookahead=1.0, max_v=1.0, kp=0.2)
    rospy.init_node('pfc_node')
    rate = rospy.Rate(10)

    pose_sub = rospy.Subscriber('/car/car_pose', PoseStamped, controller.handle_pose)
    path_sub = rospy.Subscriber('/planner/path', PoseArray, controller.handle_path)

    lookahead_pt_pub = rospy.Publisher('/pfc/lookahead_point', Pose, queue_size=1)
    path_pt_pub = rospy.Publisher('/pfc/path_point', Pose, queue_size=1)
    ctrl_pub = rospy.Publisher('/car/mux/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=1)

    while not rospy.is_shutdown():
        controller.get_action()
        path_pt_pub.publish(controller.path_point_msg())
        lookahead_pt_pub.publish(controller.lookahead_point_msg())
        ctrl_pub.publish(controller.get_action())
        rate.sleep()

if __name__ == '__main__':
    main()
