#!/usr/bin/env python

import rospy

from pfc.pure_pursuit import PurePursuitController, PurePursuitFixedVelocityController

from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

def main():
    #controller = PurePursuitController(lookahead=1.2, max_v=1.0, kp=0.4)
    controller = PurePursuitFixedVelocityController(lookahead=1.5, v=0.5, kp=0.4)
    rospy.init_node('pfc_node')
    rate = rospy.Rate(20)

    pose_sub = rospy.Subscriber('/car/particle_filter/inferred_pose', PoseStamped, controller.handle_pose)
    path_sub = rospy.Subscriber('/car/planner/path', PoseArray, controller.handle_path)
    odom_sub = rospy.Subscriber('/car/vesc/odom', Odometry, controller.handle_odom)
    exec_sub = rospy.Subscriber('/car/executive/reached_goal', Bool, controller.handle_exec)

    lookahead_pt_pub = rospy.Publisher('/car/pfc/lookahead_point', PoseStamped, queue_size=1)
    path_pt_pub = rospy.Publisher('/car/pfc/path_point', PoseStamped, queue_size=1)
    ctrl_pub = rospy.Publisher('/car/mux/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)

    for i in range(40):
        rate.sleep()

    while not rospy.is_shutdown():
        controller.get_action()
        path_pt_pub.publish(controller.path_point_msg())
        lookahead_pt_pub.publish(controller.lookahead_point_msg())
        ctrl_pub.publish(controller.get_action())
        rate.sleep()

if __name__ == '__main__':
    main()
