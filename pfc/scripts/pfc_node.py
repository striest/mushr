#!/usr/bin/env python

import rospy

from pfc.pure_pursuit import PurePursuitController

from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from ackermann_msgs.msg import AckermannDriveStamped

def main():
    controller = PurePursuitController(lookahead=0.5, max_v=1.0, kp=0.4)
    rospy.init_node('pfc_node')
    rate = rospy.Rate(20)

    pose_sub = rospy.Subscriber('/car/particle_filter/inferred_pose', PoseStamped, controller.handle_pose)
    path_sub = rospy.Subscriber('/car/planner/path', PoseArray, controller.handle_path)

    lookahead_pt_pub = rospy.Publisher('/car/pfc/lookahead_point', PoseStamped, queue_size=1)
    path_pt_pub = rospy.Publisher('/car/pfc/path_point', PoseStamped, queue_size=1)
    ctrl_pub = rospy.Publisher('/car/mux/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)

    while not rospy.is_shutdown():
        controller.get_action()
        path_pt_pub.publish(controller.path_point_msg())
        lookahead_pt_pub.publish(controller.lookahead_point_msg())
        ctrl_pub.publish(controller.get_action())
        rate.sleep()

if __name__ == '__main__':
    main()
