#!/usr/bin/env python

import rospy

from pfc.ackermann_motion_model import AckermannMotionModel

from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class AckermannPathGen:
    def __init__(self, model):
        self.pose = Pose()
        self.model = model

    def handle_pose(self, msg):
        self.pose = msg.pose

    def get_path(self):
        return self.model.sample_ackermann_path(self.pose, v_lim=(-2., 2.), x_lim=(-0.5, 3.0), y_lim=(-1.5, 1.5), n_steps=20, resample_every=20, max_steer=0.25)
        

def main():
    model = AckermannMotionModel()
    pathgen = AckermannPathGen(model)
    rospy.init_node('ackermann_test')
    freq = 16.
    rate = rospy.Rate(1/freq)
    cmd = AckermannDrive(speed=1.0, steering_angle=0.3)

    pose_sub = rospy.Subscriber('/car/particle_filter/inferred_pose', PoseStamped, pathgen.handle_pose)
    path_pub = rospy.Publisher('/planner/path', PoseArray, queue_size=1)


    while not rospy.is_shutdown():
        path_pub.publish(pathgen.get_path())
        
        rate.sleep()

if __name__ == '__main__':
    main()
